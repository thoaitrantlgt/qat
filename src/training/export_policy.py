from __future__ import annotations

import argparse
import csv
import json
import time
from pathlib import Path
from typing import Any

import torch
from torch import nn

from src.datasets.cifar import CIFAR_STATS, build_cifar_loaders
from src.models.resnet_cifar_qat import build_cifar_resnet_qat
from src.quantization.bitops import collect_resource_stats
from src.quantization.policy_applier import apply_block_policy
from src.training.train_fp32 import build_optimizer, build_scheduler, evaluate, load_config, resolve_device, train_one_epoch
from src.training.train_marl_qat import apply_first_last_bits
from src.utils.checkpoint import save_checkpoint
from src.utils.logging import save_json
from src.utils.seed import set_seed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export and optionally fine-tune a static mixed-precision policy.")
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--policy", type=str, required=True)
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument("--output-json", type=str, default="outputs/policies/static_policy_export.json")
    parser.add_argument("--output-csv", type=str, default="outputs/tables/static_policy_metrics.csv")
    parser.add_argument("--comparison-csv", type=str, default=None)
    parser.add_argument("--name", type=str, default="multiagent_static")
    parser.add_argument("--output-checkpoint", type=str, default="outputs/checkpoints/static_policy_finetuned.pt")
    parser.add_argument("--fine-tune-epochs", type=int, default=0)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def load_policy(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    policy = payload.get("policy", payload)
    if not isinstance(policy, dict):
        raise ValueError(f"Policy file {path} does not contain a policy object.")
    return {"payload": payload, "policy": policy}


def normalize_export_policy(policy: dict[str, Any]) -> dict[str, dict[str, int]]:
    normalized = {}
    for block_name, entry in policy.items():
        if not isinstance(entry, dict):
            raise ValueError(f"Policy entry for {block_name} must be an object.")
        weight_bits = entry.get("weight_bits", entry.get("w_bits"))
        activation_bits = entry.get("activation_bits", entry.get("a_bits"))
        if weight_bits is None or activation_bits is None:
            raise ValueError(f"Policy entry for {block_name} is missing bit-width fields.")
        normalized[str(block_name)] = {
            "w_bits": int(weight_bits),
            "a_bits": int(activation_bits),
        }
    return normalized


def export_payload(
    config: dict[str, Any],
    policy: dict[str, dict[str, int]],
    stats: dict[str, Any],
    metrics: dict[str, float],
    source_policy: str,
    source_checkpoint: str | None,
    reward: float | None = None,
) -> dict[str, Any]:
    model_size_ratio = stats["total_model_bits"] / stats["fp32_model_bits"] if stats.get("fp32_model_bits") else 1.0
    return {
        "model": config["model"]["name"],
        "dataset": config["dataset"]["name"],
        "policy": policy,
        "bitops_ratio": stats["bitops_ratio"],
        "model_size_ratio": model_size_ratio,
        "compression_ratio": stats["compression_ratio"],
        "reward": reward,
        "metrics": metrics,
        "source_policy": source_policy,
        "source_checkpoint": source_checkpoint,
    }


def append_metrics_csv(path: str | Path, payload: dict[str, Any]) -> None:
    csv_path = Path(path)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    row = {
        "model": payload["model"],
        "dataset": payload["dataset"],
        "top1": payload["metrics"]["top1"],
        "loss": payload["metrics"]["loss"],
        "bitops_ratio": payload["bitops_ratio"],
        "model_size_ratio": payload["model_size_ratio"],
        "compression_ratio": payload["compression_ratio"],
        "reward": payload["reward"] if payload["reward"] is not None else "",
    }
    write_header = not csv_path.exists()
    with csv_path.open("a", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(row.keys()))
        if write_header:
            writer.writeheader()
        writer.writerow(row)


def append_comparison_csv(path: str | Path, payload: dict[str, Any], stats: dict[str, Any], name: str) -> None:
    csv_path = Path(path)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    row = {
        "name": name,
        "dataset": payload["dataset"],
        "model": payload["model"],
        "top1": payload["metrics"]["top1"],
        "loss": payload["metrics"]["loss"],
        "model_size_mb": stats["model_size_mb"],
        "compression_ratio": stats["compression_ratio"],
        "avg_weight_bits": stats["avg_weight_bits"],
        "avg_activation_bits": stats["avg_activation_bits"],
        "bitops_ratio": stats["bitops_ratio"],
        "w8a8_bitops_ratio": stats["w8a8_bitops_ratio"],
        "total_bitops": stats["total_bitops"],
        "total_params": stats["total_params"],
        "layer_count": stats["layer_count"],
        "block_count": stats["block_count"],
    }
    write_header = not csv_path.exists()
    with csv_path.open("a", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(row.keys()))
        if write_header:
            writer.writeheader()
        writer.writerow(row)


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    set_seed(int(config.get("seed", 42)))
    device = resolve_device(config)

    dataset_name = config["dataset"]["name"].lower().replace("-", "")
    num_classes = int(CIFAR_STATS[dataset_name]["num_classes"])
    qat_config = config.get("qat", {})
    model = build_cifar_resnet_qat(
        config["model"]["name"],
        num_classes=num_classes,
        w_bits=int(qat_config.get("weight_bits", 8)),
        a_bits=int(qat_config.get("activation_bits", 8)),
    ).to(device)

    if args.checkpoint is not None:
        checkpoint = torch.load(args.checkpoint, map_location=device)
        state_dict = checkpoint.get("model", checkpoint)
        missing, unexpected = model.load_state_dict(state_dict, strict=False)
        print(f"loaded checkpoint: missing={len(missing)} unexpected={len(unexpected)}")

    policy_file = load_policy(args.policy)
    normalized_policy = normalize_export_policy(policy_file["policy"])
    apply_block_policy(model, normalized_policy)
    first_last_bits = qat_config.get("first_last_bits")
    apply_first_last_bits(model, int(first_last_bits) if first_last_bits is not None else None)

    if args.dry_run:
        stats = collect_resource_stats(model)
        print(
            f"dry-run ok: model={config['model']['name']} dataset={config['dataset']['name']} "
            f"blocks={len(normalized_policy)} bitops_ratio={stats['bitops_ratio']:.4f}"
        )
        return

    train_loader, val_loader, _ = build_cifar_loaders(config)
    criterion = nn.CrossEntropyLoss()
    optimizer = build_optimizer(config, model)
    scheduler = build_scheduler(config, optimizer, max(args.fine_tune_epochs, 1))
    start = time.time()

    for epoch in range(args.fine_tune_epochs):
        train_one_epoch(
            model=model,
            loader=train_loader,
            criterion=criterion,
            optimizer=optimizer,
            device=device,
            epoch=epoch,
            log_interval=int(config["training"].get("log_interval", 100)),
        )
        if scheduler is not None:
            scheduler.step()
        apply_block_policy(model, normalized_policy)
        apply_first_last_bits(model, int(first_last_bits) if first_last_bits is not None else None)

    metrics = evaluate(model, val_loader, criterion, device)
    stats = collect_resource_stats(model)
    source_reward = None
    if isinstance(policy_file["payload"], dict):
        reward_value = policy_file["payload"].get("reward")
        if isinstance(reward_value, dict):
            source_reward = float(reward_value.get("smoothed_reward", reward_value.get("global_reward", 0.0)))
        elif reward_value is not None:
            source_reward = float(reward_value)
    payload = export_payload(
        config=config,
        policy=normalized_policy,
        stats=stats,
        metrics=metrics,
        source_policy=args.policy,
        source_checkpoint=args.checkpoint,
        reward=source_reward,
    )
    payload["fine_tune_epochs"] = args.fine_tune_epochs
    payload["seconds"] = time.time() - start

    save_json(args.output_json, payload)
    append_metrics_csv(args.output_csv, payload)
    if args.comparison_csv is not None:
        append_comparison_csv(args.comparison_csv, payload, stats, args.name)
    save_checkpoint(
        args.output_checkpoint,
        {
            "model": model.state_dict(),
            "policy_export": payload,
            "config": config,
        },
    )
    print(
        f"exported static policy: top1={metrics['top1']:.2f} "
        f"bitops_ratio={payload['bitops_ratio']:.4f} json={args.output_json}"
    )


if __name__ == "__main__":
    main()
