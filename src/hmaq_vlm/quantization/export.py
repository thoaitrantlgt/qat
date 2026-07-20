from __future__ import annotations

import io
from pathlib import Path
from typing import Any

import torch
from torch import nn

from hmaq_vlm.reproducibility import atomic_write_bytes
from .inject import QuantizedConv1D, QuantizedLinear
from .policy import MixedPrecisionPolicy
from .registry import QuantGroup


CONTROLLER_PREFIXES = ("actor", "critic", "coordinator")


def _module_at(model: nn.Module, path: str) -> nn.Module:
    module = model
    for part in path.split("."):
        module = getattr(module, part)
    return module


def export_static_checkpoint(path: str | Path, model: nn.Module, policy: MixedPrecisionPolicy, metadata: dict[str, Any], registry: list[QuantGroup]) -> None:
    if set(policy.actions) != {group.name for group in registry}:
        raise ValueError("static export policy does not cover the registry exactly")
    for group in registry:
        module = _module_at(model, group.module_path)
        if not isinstance(module, (QuantizedLinear, QuantizedConv1D)) or module.action != policy.actions[group.name]:
            raise ValueError(f"model is not quantized with the frozen policy at {group.name}")
    state = {name: value.detach().cpu() for name, value in model.state_dict().items() if not name.startswith(CONTROLLER_PREFIXES)}
    payload = {"schema_version": "1.0", "model_state": state, "policy": policy.to_dict(), "metadata": metadata}
    buffer = io.BytesIO()
    torch.save(payload, buffer)
    atomic_write_bytes(path, buffer.getvalue())


def load_static_checkpoint(path: str | Path) -> dict[str, Any]:
    return torch.load(Path(path), map_location="cpu", weights_only=False)
