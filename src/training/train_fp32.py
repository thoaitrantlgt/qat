from __future__ import annotations

import argparse
import time
from pathlib import Path
from typing import Any

import torch
import yaml
from torch import nn
from torch.optim import SGD, AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, MultiStepLR

from src.datasets.cifar import build_cifar_loaders
from src.datasets.cifar import CIFAR_STATS
from src.models.resnet_cifar import build_cifar_resnet
from src.quantization.lsq import clamp_lsq_scales
from src.utils.checkpoint import save_checkpoint
from src.utils.logging import save_json
from src.utils.metrics import AverageMeter, accuracy
from src.utils.seed import set_seed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a FP32 CIFAR ResNet baseline.")
    parser.add_argument("--config", type=str, required=True, help="Path to YAML config.")
    parser.add_argument("--resume", type=str, default=None, help="Optional checkpoint to resume from.")
    parser.add_argument("--epochs", type=int, default=None, help="Override number of epochs.")
    parser.add_argument("--dry-run", action="store_true", help="Build the model and exit without training.")
    return parser.parse_args()


def load_config(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    config["config_path"] = str(path)
    return config


def build_optimizer(config: dict[str, Any], model: nn.Module) -> torch.optim.Optimizer:
    opt_config = config["optimizer"]
    name = opt_config.get("name", "sgd").lower()
    lr = float(opt_config.get("lr", 0.1))
    weight_decay = float(opt_config.get("weight_decay", 5e-4))
    quantizer_lr_multiplier = float(opt_config.get("quantizer_lr_multiplier", 1.0))
    quantizer_weight_decay = float(opt_config.get("quantizer_weight_decay", weight_decay))
    quantizer_params = []
    other_params = []
    for param_name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        if param_name.endswith(".scale") and "quantizer" in param_name:
            quantizer_params.append(parameter)
        else:
            other_params.append(parameter)
    params: Any
    if quantizer_params and (
        quantizer_lr_multiplier != 1.0 or quantizer_weight_decay != weight_decay
    ):
        params = [
            {"params": other_params, "lr": lr, "weight_decay": weight_decay},
            {
                "params": quantizer_params,
                "lr": lr * quantizer_lr_multiplier,
                "weight_decay": quantizer_weight_decay,
            },
        ]
    else:
        params = model.parameters()
    if name == "sgd":
        return SGD(
            params,
            lr=lr,
            momentum=float(opt_config.get("momentum", 0.9)),
            weight_decay=weight_decay,
            nesterov=bool(opt_config.get("nesterov", False)),
        )
    if name == "adamw":
        return AdamW(params, lr=lr, weight_decay=weight_decay)
    raise ValueError(f"Unsupported optimizer '{name}'.")


def build_scheduler(config: dict[str, Any], optimizer: torch.optim.Optimizer, epochs: int):
    sched_config = config.get("scheduler", {"name": "cosine"})
    name = sched_config.get("name", "cosine").lower()
    if name == "cosine":
        return CosineAnnealingLR(optimizer, T_max=epochs)
    if name == "multistep":
        return MultiStepLR(
            optimizer,
            milestones=list(sched_config.get("milestones", [100, 150])),
            gamma=float(sched_config.get("gamma", 0.1)),
        )
    if name in {"none", "constant"}:
        return None
    raise ValueError(f"Unsupported scheduler '{name}'.")


def resolve_device(config: dict[str, Any]) -> torch.device:
    requested = str(config["training"].get("device", "auto")).lower()
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(requested)


def train_one_epoch(
    model: nn.Module,
    loader: torch.utils.data.DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    epoch: int,
    log_interval: int,
    grad_clip: float | None = None,
) -> dict[str, float]:
    model.train()
    loss_meter = AverageMeter()
    acc_meter = AverageMeter()
    start = time.time()

    for step, (images, targets) in enumerate(loader, start=1):
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        logits = model(images)
        loss = criterion(logits, targets)
        if not torch.isfinite(loss):
            print(f"warning: non-finite loss at epoch={epoch} step={step}; skipping batch.")
            continue
        loss.backward()
        if grad_clip is not None and grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=float(grad_clip))
        optimizer.step()
        clamp_lsq_scales(model)

        batch_size = targets.size(0)
        top1 = accuracy(logits, targets, topk=(1,))[0].item()
        loss_meter.update(loss.item(), batch_size)
        acc_meter.update(top1, batch_size)

        if log_interval > 0 and step % log_interval == 0:
            print(
                f"epoch={epoch} step={step}/{len(loader)} "
                f"loss={loss_meter.avg:.4f} top1={acc_meter.avg:.2f}"
            )

    return {
        "loss": loss_meter.avg,
        "top1": acc_meter.avg,
        "seconds": time.time() - start,
    }


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: torch.utils.data.DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> dict[str, float]:
    model.eval()
    loss_meter = AverageMeter()
    acc_meter = AverageMeter()

    for images, targets in loader:
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        logits = model(images)
        loss = criterion(logits, targets)
        batch_size = targets.size(0)
        top1 = accuracy(logits, targets, topk=(1,))[0].item()
        loss_meter.update(loss.item(), batch_size)
        acc_meter.update(top1, batch_size)

    return {"loss": loss_meter.avg, "top1": acc_meter.avg}


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    seed = int(config.get("seed", 42))
    set_seed(seed)

    device = resolve_device(config)
    if args.dry_run:
        dataset_name = config["dataset"]["name"].lower().replace("-", "")
        num_classes = int(CIFAR_STATS[dataset_name]["num_classes"])
        model = build_cifar_resnet(config["model"]["name"], num_classes=num_classes).to(device)
        total_params = sum(parameter.numel() for parameter in model.parameters())
        print(
            f"dry-run ok: model={config['model']['name']} num_classes={num_classes} "
            f"params={total_params} device={device}"
        )
        return

    train_loader, val_loader, num_classes = build_cifar_loaders(config)
    model = build_cifar_resnet(config["model"]["name"], num_classes=num_classes).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = build_optimizer(config, model)

    epochs = int(args.epochs or config["training"].get("epochs", 200))
    scheduler = build_scheduler(config, optimizer, epochs)
    start_epoch = 0
    best_top1 = 0.0
    history: list[dict[str, Any]] = []

    if args.resume is not None:
        checkpoint = torch.load(args.resume, map_location=device)
        model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        if scheduler is not None and checkpoint.get("scheduler") is not None:
            scheduler.load_state_dict(checkpoint["scheduler"])
        start_epoch = int(checkpoint.get("epoch", -1)) + 1
        best_top1 = float(checkpoint.get("best_top1", 0.0))

    output_config = config.get("output", {})
    checkpoint_path = Path(output_config.get("checkpoint", "outputs/checkpoints/resnet32_cifar100_fp32_best.pt"))
    last_checkpoint_path = checkpoint_path.with_name(checkpoint_path.stem + "_last.pt")
    log_path = Path(output_config.get("log", "outputs/logs/fp32_cifar100_resnet32.json"))
    log_interval = int(config["training"].get("log_interval", 100))
    grad_clip = float(config["training"].get("grad_clip", 0.0))

    total_start = time.time()
    for epoch in range(start_epoch, epochs):
        train_metrics = train_one_epoch(
            model=model,
            loader=train_loader,
            criterion=criterion,
            optimizer=optimizer,
            device=device,
            epoch=epoch,
            log_interval=log_interval,
            grad_clip=grad_clip if grad_clip > 0 else None,
        )
        val_metrics = evaluate(model, val_loader, criterion, device)
        if scheduler is not None:
            scheduler.step()

        lr = optimizer.param_groups[0]["lr"]
        row = {
            "epoch": epoch,
            "lr": lr,
            "train_loss": train_metrics["loss"],
            "train_top1": train_metrics["top1"],
            "val_loss": val_metrics["loss"],
            "val_top1": val_metrics["top1"],
            "epoch_seconds": train_metrics["seconds"],
        }
        history.append(row)
        print(
            f"epoch={epoch} lr={lr:.6f} train_loss={row['train_loss']:.4f} "
            f"train_top1={row['train_top1']:.2f} val_loss={row['val_loss']:.4f} "
            f"val_top1={row['val_top1']:.2f}"
        )

        is_best = val_metrics["top1"] > best_top1
        if is_best:
            best_top1 = val_metrics["top1"]

        payload = {
            "epoch": epoch,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict() if scheduler is not None else None,
            "best_top1": best_top1,
            "config": config,
        }
        save_checkpoint(last_checkpoint_path, payload)

        if is_best:
            save_checkpoint(checkpoint_path, payload)

        save_json(
            log_path,
            {
                "config": config,
                "seed": seed,
                "best_top1": best_top1,
                "latest_epoch": epoch,
                "history": history,
                "total_seconds": time.time() - total_start,
            },
        )

    print(f"Best Top-1: {best_top1:.2f}. Checkpoint: {checkpoint_path}")


if __name__ == "__main__":
    main()
