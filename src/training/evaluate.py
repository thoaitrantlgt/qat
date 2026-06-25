from __future__ import annotations

import argparse
from pathlib import Path

import torch
import yaml
from torch import nn

from src.datasets.cifar import build_cifar_loaders
from src.models.resnet_cifar import build_cifar_resnet
from src.training.train_fp32 import evaluate, resolve_device
from src.utils.logging import save_json
from src.utils.seed import set_seed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate a FP32 CIFAR ResNet checkpoint.")
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--output", type=str, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    with Path(args.config).open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    config["config_path"] = args.config
    set_seed(int(config.get("seed", 42)))

    _, val_loader, num_classes = build_cifar_loaders(config)
    device = resolve_device(config)
    model = build_cifar_resnet(config["model"]["name"], num_classes=num_classes).to(device)
    checkpoint = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(checkpoint["model"])

    metrics = evaluate(model, val_loader, nn.CrossEntropyLoss(), device)
    payload = {
        "checkpoint": args.checkpoint,
        "config": config,
        "metrics": metrics,
        "best_top1_in_checkpoint": checkpoint.get("best_top1"),
    }
    print(f"loss={metrics['loss']:.4f} top1={metrics['top1']:.2f}")
    if args.output is not None:
        save_json(args.output, payload)


if __name__ == "__main__":
    main()
