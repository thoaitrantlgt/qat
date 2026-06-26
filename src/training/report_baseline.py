from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Any

import torch
import yaml
from torch import nn

from src.datasets.cifar import build_cifar_loaders
from src.training.model_factory import build_model_from_config
from src.training.train_fp32 import evaluate, resolve_device
from src.utils.logging import save_json
from src.utils.model_stats import collect_model_stats
from src.utils.seed import set_seed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate and summarize a baseline checkpoint.")
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--name", type=str, default=None)
    parser.add_argument("--json", type=str, default="outputs/tables/baselines.json")
    parser.add_argument("--csv", type=str, default="outputs/tables/baselines.csv")
    return parser.parse_args()


def load_config(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    config["config_path"] = str(path)
    return config


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    seed = int(config.get("seed", 42))
    set_seed(seed)

    device = resolve_device(config)
    _, val_loader, num_classes = build_cifar_loaders(config)
    model = build_model_from_config(config, num_classes=num_classes).to(device)
    checkpoint = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(checkpoint["model"], strict=False if "qat" in config else True)

    metrics = evaluate(model, val_loader, nn.CrossEntropyLoss(), device)
    stats = collect_model_stats(model)
    summary = {
        "name": args.name or Path(args.checkpoint).stem,
        "config": config,
        "checkpoint": args.checkpoint,
        "metrics": metrics,
        "stats": stats,
        "best_top1_in_checkpoint": checkpoint.get("best_top1"),
        "seed": seed,
    }
    save_json(args.json, summary)

    csv_path = Path(args.csv)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    row = {
        "name": summary["name"],
        "dataset": config["dataset"]["name"],
        "model": config["model"]["name"],
        "top1": metrics["top1"],
        "loss": metrics["loss"],
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
    with csv_path.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(row.keys()))
        if write_header:
            writer.writeheader()
        writer.writerow(row)

    print(
        f"{row['name']}: top1={row['top1']:.2f} loss={row['loss']:.4f} "
        f"model_size_mb={row['model_size_mb']:.3f} bitops_ratio={row['bitops_ratio']:.4f}"
    )


if __name__ == "__main__":
    main()
