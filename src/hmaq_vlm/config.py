from __future__ import annotations

from dataclasses import asdict, dataclass, field, fields, is_dataclass
from pathlib import Path
from types import UnionType
from typing import Any, TypeVar, get_args, get_origin, get_type_hints

import yaml


@dataclass(frozen=True)
class ModelConfig:
    vision_model: str = "timm/vit_small_patch16_224.augreg_in21k_ft_in1k"
    vision_revision: str = "7e2c55630205e1266030f18370f4c6ed1a514b52"
    language_model: str = "openai-community/gpt2"
    language_revision: str = "607a30d783dfa663caf39e06633721c8d4cfcd7e"
    image_size: int = 224
    max_new_tokens: int = 30
    gradient_checkpointing: bool = True


@dataclass(frozen=True)
class DataConfig:
    coco_dataset: str = "yerevann/coco-karpathy"
    coco_revision: str = "448fdb1bc7b2d09e46881c4541a14d796a3d41e8"
    flickr_images: str | None = None
    flickr_annotations: str | None = None
    workers: int = 4
    train_sample_limit: int | None = None
    validation_sample_limit: int | None = None


@dataclass(frozen=True)
class TrainConfig:
    micro_batch_size: int = 4
    gradient_accumulation: int = 8
    epochs: int = 10
    projector_warmup_epochs: int = 3
    gradient_clip: float = 1.0
    projector_lr: float = 1.0e-4
    vision_lr: float = 1.0e-5
    language_lr: float = 5.0e-6
    amp: str = "fp16"

    @property
    def effective_batch_size(self) -> int:
        return self.micro_batch_size * self.gradient_accumulation


@dataclass(frozen=True)
class SearchConfig:
    candidate_budget: int = 100
    timing_budget: int = 50
    ppo_clip: float = 0.2
    gae_lambda: float = 0.95
    optimization_epochs: int = 4
    actor_lr: float = 3.0e-4
    critic_lr: float = 1.0e-3
    entropy: float = 0.01
    vision_min_budget: float = 0.10
    projector_min_budget: float = 0.05
    language_min_budget: float = 0.30


@dataclass(frozen=True)
class ExperimentConfig:
    seed: int = 11
    model: ModelConfig = field(default_factory=ModelConfig)
    data: DataConfig = field(default_factory=DataConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    search: SearchConfig = field(default_factory=SearchConfig)
    artifacts_dir: str = "artifacts"
    schema_version: str = "1.0"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


T = TypeVar("T")


def _matches(value: Any, annotation: Any) -> bool:
    origin = get_origin(annotation)
    if origin is UnionType:
        return any(_matches(value, option) for option in get_args(annotation))
    if annotation is float:
        return isinstance(value, (int, float)) and not isinstance(value, bool)
    if annotation is int:
        return isinstance(value, int) and not isinstance(value, bool)
    return isinstance(value, annotation) if isinstance(annotation, type) else True


def _construct(cls: type[T], values: dict[str, Any], prefix: str = "") -> T:
    allowed = {item.name: item for item in fields(cls)}
    unknown = sorted(set(values) - set(allowed))
    if unknown:
        name = f"{prefix}.{unknown[0]}" if prefix else unknown[0]
        raise ValueError(f"unknown config key: {name}")
    hints = get_type_hints(cls)
    kwargs: dict[str, Any] = {}
    for name, value in values.items():
        target = hints.get(name)
        if isinstance(value, dict) and isinstance(target, type) and is_dataclass(target):
            value = _construct(target, value, f"{prefix}.{name}".strip("."))
        if target is not None and not _matches(value, target):
            key = f"{prefix}.{name}".strip(".")
            raise ValueError(f"config key {key} has invalid type: expected {target}, got {type(value).__name__}")
        kwargs[name] = value
    return cls(**kwargs)


def load_config(path: str | Path) -> ExperimentConfig:
    raw = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
    if not isinstance(raw, dict):
        raise ValueError("config root must be a mapping")
    config = _construct(ExperimentConfig, raw)
    if config.seed < 0 or config.train.micro_batch_size < 1 or config.train.gradient_accumulation < 1 or config.train.epochs < 1:
        raise ValueError("seed must be non-negative and training counts must be positive")
    sample_limits = (config.data.train_sample_limit, config.data.validation_sample_limit)
    if any(value is not None and value < 1 for value in sample_limits):
        raise ValueError("data sample limits must be positive when set")
    if not 0 <= config.train.projector_warmup_epochs <= config.train.epochs or config.train.gradient_clip <= 0 or config.train.amp != "fp16":
        raise ValueError("invalid training schedule, clipping, or AMP mode")
    if not 0 <= config.model.max_new_tokens <= 30 or config.model.image_size < 1:
        raise ValueError("invalid model generation or image size")
    if config.search.candidate_budget < 1 or not 0 <= config.search.timing_budget <= config.search.candidate_budget:
        raise ValueError("invalid search budgets")
    minimums = (config.search.vision_min_budget, config.search.projector_min_budget, config.search.language_min_budget)
    if any(value < 0 for value in minimums) or sum(minimums) >= 1:
        raise ValueError("invalid modality minimum budgets")
    return config
