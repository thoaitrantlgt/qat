from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import torch
from torch import nn
from torch.utils.data import DataLoader, Subset

from src.quantization.bitops import collect_resource_stats
from src.utils.metrics import AverageMeter, accuracy


@dataclass(frozen=True)
class RewardConfig:
    lambda_bitops: float = 0.1
    lambda_local: float = 0.01
    lambda_accuracy_drop: float = 0.0
    accuracy_target: float | None = None
    smoothing_alpha: float = 0.9
    val_subset_size: int = 2048
    clip_min: float = -100.0
    clip_max: float = 100.0


@dataclass
class RewardResult:
    validation_accuracy: float
    validation_loss: float
    bitops_ratio: float
    global_reward: float
    smoothed_reward: float
    local_rewards: list[float]
    block_bitops_ratios: list[float]
    num_samples: int

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class RewardMovingAverage:
    def __init__(self, alpha: float = 0.9) -> None:
        self.alpha = float(alpha)
        self.value: float | None = None

    def update(self, reward: float) -> float:
        reward = float(reward)
        if self.value is None:
            self.value = reward
        else:
            self.value = self.alpha * self.value + (1.0 - self.alpha) * reward
        return self.value

    def reset(self) -> None:
        self.value = None


def clip_reward(value: float, clip_min: float = -100.0, clip_max: float = 100.0) -> float:
    return max(float(clip_min), min(float(clip_max), float(value)))


def build_fixed_subset_loader(loader: DataLoader, subset_size: int = 2048) -> DataLoader:
    dataset = loader.dataset
    sample_count = min(int(subset_size), len(dataset))
    if sample_count >= len(dataset):
        subset = dataset
    else:
        subset = Subset(dataset, list(range(sample_count)))

    return DataLoader(
        subset,
        batch_size=loader.batch_size,
        shuffle=False,
        num_workers=loader.num_workers,
        pin_memory=loader.pin_memory,
        drop_last=False,
    )


@torch.no_grad()
def evaluate_validation_subset(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    criterion: nn.Module | None = None,
) -> dict[str, float]:
    was_training = model.training
    model.eval()
    criterion = criterion or nn.CrossEntropyLoss()
    loss_meter = AverageMeter()
    acc_meter = AverageMeter()
    sample_count = 0

    for images, targets in loader:
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        logits = model(images)
        loss = criterion(logits, targets)
        batch_size = targets.size(0)
        top1 = accuracy(logits, targets, topk=(1,))[0].item()
        loss_meter.update(loss.item(), batch_size)
        acc_meter.update(top1, batch_size)
        sample_count += int(batch_size)

    if was_training:
        model.train()

    return {"loss": loss_meter.avg, "top1": acc_meter.avg, "num_samples": float(sample_count)}


def compute_rewards(
    validation_accuracy: float,
    resource_stats: dict[str, Any],
    config: RewardConfig | None = None,
    moving_average: RewardMovingAverage | None = None,
) -> RewardResult:
    config = config or RewardConfig()
    bitops_ratio = float(resource_stats.get("bitops_ratio", 1.0))
    accuracy_drop = 0.0
    if config.accuracy_target is not None:
        accuracy_drop = max(0.0, float(config.accuracy_target) - float(validation_accuracy))
    global_reward = (
        validation_accuracy
        - config.lambda_bitops * bitops_ratio
        - config.lambda_accuracy_drop * accuracy_drop
    )
    global_reward = clip_reward(global_reward, config.clip_min, config.clip_max)
    smoothed_reward = moving_average.update(global_reward) if moving_average is not None else global_reward

    residual_blocks = [block for block in resource_stats.get("blocks", []) if str(block.get("kind")) == "residual"]
    block_bitops_ratios = [float(block.get("bitops_ratio", 0.0)) for block in residual_blocks]
    local_rewards = [
        clip_reward(global_reward - config.lambda_local * block_ratio, config.clip_min, config.clip_max)
        for block_ratio in block_bitops_ratios
    ]

    return RewardResult(
        validation_accuracy=float(validation_accuracy),
        validation_loss=0.0,
        bitops_ratio=bitops_ratio,
        global_reward=global_reward,
        smoothed_reward=smoothed_reward,
        local_rewards=local_rewards,
        block_bitops_ratios=block_bitops_ratios,
        num_samples=0,
    )


class RewardEvaluator:
    def __init__(
        self,
        config: RewardConfig | None = None,
        criterion: nn.Module | None = None,
        moving_average: RewardMovingAverage | None = None,
    ) -> None:
        self.config = config or RewardConfig()
        self.criterion = criterion or nn.CrossEntropyLoss()
        self.moving_average = moving_average or RewardMovingAverage(alpha=self.config.smoothing_alpha)

    def evaluate(
        self,
        model: nn.Module,
        val_loader: DataLoader,
        device: torch.device,
        resource_stats: dict[str, Any] | None = None,
    ) -> RewardResult:
        subset_loader = build_fixed_subset_loader(val_loader, subset_size=self.config.val_subset_size)
        metrics = evaluate_validation_subset(model, subset_loader, device=device, criterion=self.criterion)
        resource_stats = resource_stats or collect_resource_stats(model)
        result = compute_rewards(
            validation_accuracy=float(metrics["top1"]),
            resource_stats=resource_stats,
            config=self.config,
            moving_average=self.moving_average,
        )
        result.validation_loss = float(metrics["loss"])
        result.num_samples = int(metrics["num_samples"])
        return result
