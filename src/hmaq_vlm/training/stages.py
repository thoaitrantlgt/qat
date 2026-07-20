from __future__ import annotations

from torch import nn


def set_trainable_stage(model: nn.Module, epoch: int, warmup_epochs: int = 3) -> None:
    warmup = epoch <= warmup_epochs
    for name, parameter in model.named_parameters():
        parameter.requires_grad = (not warmup) or name.startswith("projector.")
