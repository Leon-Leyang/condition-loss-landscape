import random
from dataclasses import asdict, dataclass

import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Subset


@dataclass
class Config:
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    seed: int = 0

    data_root: str = "./data"
    subset_size: int = 1000
    batch_size: int = 1000   # full-batch GD on the subset
    num_workers: int = 2

    num_classes: int = 10
    # Loss supported: "mse", "cross_entropy"
    loss_type: str = "mse"
    lr: float = 2 / 150      # try 2/200, 2/150, 2/100
    weight_decay: float = 0.0
    momentum: float = 0.0    # pure GD-style
    epochs: int = 5000

    sharpness_every: int = 20
    power_iters: int = 10

    log_path: str = "resnet_cifar10_subset1000_eos_log.pt"
    loss_plot_path: str = "loss_plot.png"
    sharpness_plot_path: str = "sharpness_plot.png"


cfg = Config()


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def build_loader(cfg: Config) -> DataLoader:
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(
            mean=(0.4914, 0.4822, 0.4465),
            std=(0.2470, 0.2435, 0.2616),
        ),
    ])

    train_dataset = torchvision.datasets.CIFAR10(
        root=cfg.data_root,
        train=True,
        download=True,
        transform=transform,
    )

    if cfg.num_classes > 10:
        raise ValueError("CIFAR-10 only has 10 classes (0..9).")

    # Build a deterministic, class-balanced subset using only classes 0..num_classes-1.
    # Total size is exactly cfg.subset_size (remainder distributed over the lowest class ids).
    rng = random.Random(cfg.seed)
    targets = train_dataset.targets  # list[int] of length 50000

    per_class = cfg.subset_size // cfg.num_classes
    remainder = cfg.subset_size % cfg.num_classes

    chosen_indices = []
    for c in range(cfg.num_classes):
        count = per_class + (1 if c < remainder else 0)
        cls_indices = [i for i, t in enumerate(targets) if t == c]
        if count > len(cls_indices):
            raise ValueError(
                f"Requested {count} samples for class {c}, "
                f"but class has only {len(cls_indices)} samples."
            )
        chosen_indices.extend(rng.sample(cls_indices, count))

    # Mix classes so batch composition isn't class-blocked (important if batch_size < subset_size).
    rng.shuffle(chosen_indices)

    subset = Subset(train_dataset, chosen_indices)

    loader = DataLoader(
        subset,
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=cfg.num_workers,
        pin_memory=True,
    )
    return loader


def build_model(cfg: Config) -> nn.Module:
    model = torchvision.models.resnet18(num_classes=cfg.num_classes)

    # CIFAR-10 adjustment
    model.conv1 = nn.Conv2d(
        3, 64, kernel_size=3, stride=1, padding=1, bias=False
    )
    model.maxpool = nn.Identity()

    return model.to(cfg.device)


def get_params(model: nn.Module):
    return [p for p in model.parameters() if p.requires_grad]


def normalize_vector_list(vecs, eps: float = 1e-12):
    norm_sq = sum((v * v).sum() for v in vecs)
    norm = torch.sqrt(norm_sq + eps)
    return [v / norm for v in vecs]


def hessian_vector_product(loss, params, vec):
    grads = torch.autograd.grad(
        loss,
        params,
        create_graph=True,
        retain_graph=True,
    )
    gv = sum((g * v).sum() for g, v in zip(grads, vec))
    hvp = torch.autograd.grad(
        gv,
        params,
        retain_graph=True,
    )
    return hvp


def compute_loss_from_logits(logits: torch.Tensor, y: torch.Tensor, cfg: Config):
    loss_type = cfg.loss_type.lower().strip()
    if loss_type == "mse":
        # Squared loss between logits and one-hot labels.
        y_onehot = F.one_hot(y, num_classes=cfg.num_classes).to(dtype=logits.dtype)
        return F.mse_loss(logits, y_onehot, reduction="mean")
    if loss_type in {"cross_entropy", "ce"}:
        return F.cross_entropy(logits, y)
    raise ValueError(f"Unknown loss_type={cfg.loss_type!r}. Use 'mse' or 'cross_entropy'.")


def top_hessian_eigenvalue(model, x, y, cfg: Config, power_iters=10):
    # Temporarily switch to train mode (to match training dynamics),
    # but restore original mode and BatchNorm buffers afterward so the
    # model itself is not changed by this computation.
    was_training = model.training

    bn_layers = [m for m in model.modules() if isinstance(m, nn.BatchNorm2d)]
    bn_state = [
        (
            bn.running_mean.detach().clone(),
            bn.running_var.detach().clone(),
            bn.num_batches_tracked.detach().clone(),
        )
        for bn in bn_layers
    ]

    model.train()
    params = [p for p in model.parameters() if p.requires_grad]

    logits = model(x)
    loss = compute_loss_from_logits(logits, y, cfg)

    vec = [torch.randn_like(p) for p in params]
    vec = normalize_vector_list(vec)

    for _ in range(power_iters):
        hvp = hessian_vector_product(loss, params, vec)
        vec = normalize_vector_list([h.detach() for h in hvp])

    hvp = hessian_vector_product(loss, params, vec)
    eigval = sum((v * h).sum() for v, h in zip(vec, hvp)).item()

    # Restore original BatchNorm buffers and training mode.
    for bn, (running_mean, running_var, num_batches_tracked) in zip(
        bn_layers, bn_state
    ):
        bn.running_mean.data.copy_(running_mean)
        bn.running_var.data.copy_(running_var)
        bn.num_batches_tracked.data.copy_(num_batches_tracked)

    model.train(was_training)

    return float(eigval)


@torch.no_grad()
def compute_loss_acc(model: nn.Module, x: torch.Tensor, y: torch.Tensor, cfg: Config):
    model.eval()
    logits = model(x)
    loss = compute_loss_from_logits(logits, y, cfg).item()
    acc = (logits.argmax(dim=1) == y).float().mean().item()
    return loss, acc


def plot_loss(history: dict, save_path: str):
    plt.figure(figsize=(7, 5))
    plt.plot(history["step"], history["loss"])
    plt.xlabel("Iteration")
    plt.ylabel("Training Loss")
    plt.title("ResNet18 on CIFAR-10 subset (1000): Loss")
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()


def plot_sharpness(history: dict, lr: float, save_path: str):
    sharp_steps = []
    sharp_vals = []

    for s, sh in zip(history["step"], history["sharpness"]):
        if sh is not None:
            sharp_steps.append(s)
            sharp_vals.append(sh)

    plt.figure(figsize=(7, 5))
    plt.plot(sharp_steps, sharp_vals, label="Top Hessian Eigenvalue")
    plt.axhline(
        y=2.0 / lr,
        linestyle="--",
        label=f"2/eta = {2.0 / lr:.2f}",
    )
    plt.xlabel("Iteration")
    plt.ylabel("Sharpness")
    plt.title("ResNet18 on CIFAR-10 subset (1000): Sharpness")
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()


def main():
    set_seed(cfg.seed)

    print(f"Device: {cfg.device}")
    print(f"Learning rate eta = {cfg.lr}")
    print(f"Theoretical EOS threshold 2/eta = {2.0 / cfg.lr:.4f}")

    loader = build_loader(cfg)
    model = build_model(cfg)

    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=cfg.lr,
        momentum=cfg.momentum,
        weight_decay=cfg.weight_decay,
    )

    x_full, y_full = next(iter(loader))
    x_full = x_full.to(cfg.device, non_blocking=True)
    y_full = y_full.to(cfg.device, non_blocking=True)

    history = {
        "step": [],
        "loss": [],
        "acc": [],
        "sharpness": [],
    }

    for step in range(cfg.epochs + 1):
        model.train()
        optimizer.zero_grad(set_to_none=True)

        logits = model(x_full)
        loss = compute_loss_from_logits(logits, y_full, cfg)
        loss.backward()
        optimizer.step()

        train_loss, train_acc = compute_loss_acc(model, x_full, y_full, cfg)

        sharpness = None
        if step % cfg.sharpness_every == 0:
            sharpness = top_hessian_eigenvalue(
                model=model,
                x=x_full,
                y=y_full,
                cfg=cfg,
                power_iters=cfg.power_iters,
            )

        history["step"].append(step)
        history["loss"].append(train_loss)
        history["acc"].append(train_acc)
        history["sharpness"].append(sharpness)

        if sharpness is None:
            print(
                f"step={step:4d} | "
                f"loss={train_loss:.6f} | "
                f"acc={train_acc:.4f}"
            )
        else:
            print(
                f"step={step:4d} | "
                f"loss={train_loss:.6f} | "
                f"acc={train_acc:.4f} | "
                f"sharpness={sharpness:.6f} | "
                f"eta*sharpness={cfg.lr * sharpness:.6f}"
            )

    torch.save({"cfg": asdict(cfg), "history": history}, cfg.log_path)
    print(f"\nSaved log to: {cfg.log_path}")

    plot_loss(history, cfg.loss_plot_path)
    print(f"Saved loss plot to: {cfg.loss_plot_path}")

    plot_sharpness(history, cfg.lr, cfg.sharpness_plot_path)
    print(f"Saved sharpness plot to: {cfg.sharpness_plot_path}")


if __name__ == "__main__":
    main()