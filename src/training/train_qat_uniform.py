from __future__ import annotations

import argparse
import time
from pathlib import Path
from typing import Any

import torch
import yaml
from torch import nn

from src.agents.state_builder import AgentStateBuilder, summarize_agent_states
from src.datasets.cifar import CIFAR_STATS, build_cifar_loaders
from src.models.resnet_cifar_qat import build_cifar_resnet_qat
from src.quantization.bitops import collect_resource_stats
from src.quantization.policy_applier import set_uniform_bit_widths
from src.training.train_fp32 import build_optimizer, build_scheduler, evaluate, load_config, resolve_device, train_one_epoch
from src.utils.checkpoint import save_checkpoint
from src.utils.logging import save_json
from src.utils.seed import set_seed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a uniform QAT CIFAR ResNet baseline.")
    parser.add_argument("--config", type=str, required=True, help="Path to YAML config.")
    parser.add_argument("--fp32-checkpoint", type=str, default=None, help="Optional FP32 checkpoint to warm start from.")
    parser.add_argument("--epochs", type=int, default=None, help="Override number of epochs.")
    parser.add_argument("--dry-run", action="store_true", help="Build the model and exit without training.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    seed = int(config.get("seed", 42))
    set_seed(seed)

    device = resolve_device(config)
    dataset_name = config["dataset"]["name"].lower().replace("-", "")
    num_classes = int(CIFAR_STATS[dataset_name]["num_classes"])
    qat_config = config.get("qat", {})
    weight_bits = int(qat_config.get("weight_bits", 8))
    activation_bits = int(qat_config.get("activation_bits", 8))
    first_last_bits = qat_config.get("first_last_bits")
    warmup_epochs = int(qat_config.get("warmup_epochs", 0))
    if "qat" not in config:
        print("warning: config has no 'qat' section; defaulting to W8A8.")

    model = build_cifar_resnet_qat(
        config["model"]["name"],
        num_classes=num_classes,
        w_bits=weight_bits,
        a_bits=activation_bits,
    ).to(device)

    if args.fp32_checkpoint is not None:
        fp32_path = Path(args.fp32_checkpoint)
        if fp32_path.exists():
            checkpoint = torch.load(fp32_path, map_location=device)
            missing, unexpected = model.load_state_dict(checkpoint["model"], strict=False)
            print(f"loaded fp32 checkpoint: missing={len(missing)} unexpected={len(unexpected)}")
            fp32_checkpoint_info: dict[str, Any] | None = {
                "path": str(fp32_path),
                "missing": len(missing),
                "unexpected": len(unexpected),
            }
        else:
            print(f"warning: fp32 checkpoint not found at {fp32_path}; training from scratch.")
            fp32_checkpoint_info = {
                "path": str(fp32_path),
                "missing": None,
                "unexpected": None,
                "not_found": True,
            }
    else:
        fp32_checkpoint_info = None

    if args.dry_run:
        set_uniform_bit_widths(model, weight_bits=weight_bits, activation_bits=activation_bits)
        if first_last_bits is not None:
            model.conv1.set_bits(int(first_last_bits), int(first_last_bits))
            model.fc.set_bits(int(first_last_bits), int(first_last_bits))
        total_params = sum(parameter.numel() for parameter in model.parameters())
        print(
            f"dry-run ok: model={config['model']['name']} num_classes={num_classes} "
            f"params={total_params} device={device} w_bits={weight_bits} "
            f"a_bits={activation_bits}"
        )
        return

    train_loader, val_loader, _ = build_cifar_loaders(config)
    criterion = nn.CrossEntropyLoss()
    optimizer = build_optimizer(config, model)

    epochs = int(args.epochs or config["training"].get("epochs", 200))
    scheduler = build_scheduler(config, optimizer, epochs)
    start_epoch = 0
    best_top1 = 0.0
    history: list[dict[str, Any]] = []

    output_config = config.get("output", {})
    checkpoint_path = Path(output_config.get("checkpoint", "outputs/checkpoints/resnet32_cifar100_qat_best.pt"))
    last_checkpoint_path = checkpoint_path.with_name(checkpoint_path.stem + "_last.pt")
    log_path = Path(output_config.get("log", "outputs/logs/qat_cifar100_resnet32.json"))
    log_interval = int(config["training"].get("log_interval", 100))
    grad_clip = float(config["training"].get("grad_clip", 0.0))

    def apply_precision(epoch: int) -> None:
        if warmup_epochs > 0 and epoch < warmup_epochs:
            set_uniform_bit_widths(model, weight_bits=8, activation_bits=8)
            if first_last_bits is not None:
                model.conv1.set_bits(int(first_last_bits), int(first_last_bits))
                model.fc.set_bits(int(first_last_bits), int(first_last_bits))
            return

        set_uniform_bit_widths(model, weight_bits=weight_bits, activation_bits=activation_bits)
        if first_last_bits is not None:
            model.conv1.set_bits(int(first_last_bits), int(first_last_bits))
            model.fc.set_bits(int(first_last_bits), int(first_last_bits))

    def is_target_precision_epoch(epoch: int) -> bool:
        return epoch >= warmup_epochs or (weight_bits == 8 and activation_bits == 8)

    apply_precision(start_epoch)
    warmstart_metrics = (
        evaluate(model, val_loader, criterion, device)
        if fp32_checkpoint_info and not fp32_checkpoint_info.get("not_found")
        else None
    )
    if warmstart_metrics is not None:
        print(
            f"warmstart_eval top1={warmstart_metrics['top1']:.2f} "
            f"loss={warmstart_metrics['loss']:.4f}"
        )

    total_start = time.time()
    for epoch in range(start_epoch, epochs):
        apply_precision(epoch)
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

        resource_stats = collect_resource_stats(model)
        state_builder = AgentStateBuilder(resource_stats)
        agent_states = state_builder.build(model, {"epoch": epoch, "total_epochs": epochs})
        agent_state_summary = summarize_agent_states(agent_states)
        lr = optimizer.param_groups[0]["lr"]
        row = {
            "epoch": epoch,
            "lr": lr,
            "train_loss": train_metrics["loss"],
            "train_top1": train_metrics["top1"],
            "val_loss": val_metrics["loss"],
            "val_top1": val_metrics["top1"],
            "epoch_seconds": train_metrics["seconds"],
            "weight_bits": weight_bits,
            "activation_bits": activation_bits,
            "agent_state_summary": agent_state_summary,
        }
        history.append(row)
        print(
            f"epoch={epoch} lr={lr:.6f} train_loss={row['train_loss']:.4f} "
            f"train_top1={row['train_top1']:.2f} val_loss={row['val_loss']:.4f} "
            f"val_top1={row['val_top1']:.2f} agent_states={agent_state_summary['num_agents']}x{agent_state_summary['state_dim']}"
        )

        is_best = is_target_precision_epoch(epoch) and val_metrics["top1"] > best_top1
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
                "fp32_checkpoint": fp32_checkpoint_info,
                "warmstart_metrics": warmstart_metrics,
                "best_top1": best_top1,
                "latest_epoch": epoch,
                "history": history,
                "total_seconds": time.time() - total_start,
            },
        )

    print(f"Best Top-1: {best_top1:.2f}. Checkpoint: {checkpoint_path}")


if __name__ == "__main__":
    main()
