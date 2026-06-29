from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import torch
from torch import nn

from src.quantization.bitops import collect_resource_stats
from src.quantization.quant_layers import QuantConv2d, QuantLinear
from src.utils.logging import save_json


STATE_FEATURES = (
    "block_id_normalized",
    "in_channels",
    "out_channels",
    "spatial_size",
    "current_weight_bits",
    "current_activation_bits",
    "weight_std",
    "activation_range_ema",
    "gradient_norm",
    "block_bitops_ratio",
    "remaining_bitops_budget",
    "epoch_ratio",
)


@dataclass(frozen=True)
class AgentBlockMetadata:
    block_id: int
    block_name: str
    layer_names: list[str]
    in_channels: int
    out_channels: int
    spatial_size: int
    macs: int
    params: int
    fp32_bitops: int


def _safe_div(numerator: float, denominator: float, default: float = 0.0) -> float:
    if denominator == 0:
        return default
    return float(numerator) / float(denominator)


def _clamp01(value: float) -> float:
    if value != value or value in {float("inf"), float("-inf")}:
        return 0.0
    return max(0.0, min(1.0, float(value)))


def _shape_area(shape: list[int], default: int = 1) -> int:
    if len(shape) >= 4:
        return int(shape[-1]) * int(shape[-2])
    return default


def _dict_value(values: dict[str, Any], block_name: str, default: float = 0.0) -> float:
    if block_name in values:
        return float(values[block_name])
    short_name = block_name.replace(".", "_")
    if short_name in values:
        return float(values[short_name])
    return default


def _extract_stat_map(train_stats: dict[str, Any], *keys: str) -> dict[str, Any]:
    for key in keys:
        value = train_stats.get(key)
        if isinstance(value, dict):
            return value
    return {}


def _epoch_ratio(train_stats: dict[str, Any]) -> float:
    if "epoch_ratio" in train_stats:
        return _clamp01(float(train_stats["epoch_ratio"]))
    epoch = float(train_stats.get("epoch", 0.0))
    total_epochs = float(train_stats.get("total_epochs", train_stats.get("epochs", 1.0)))
    return _clamp01(_safe_div(epoch, max(total_epochs - 1.0, 1.0)))


def _remaining_bitops_budget(resource_stats: dict[str, Any], train_stats: dict[str, Any]) -> float:
    if "remaining_bitops_budget" in train_stats:
        return _clamp01(float(train_stats["remaining_bitops_budget"]))

    bitops_ratio = float(resource_stats.get("bitops_ratio", 1.0))
    if "bitops_budget_ratio" in train_stats:
        budget_ratio = float(train_stats["bitops_budget_ratio"])
        return _clamp01(_safe_div(budget_ratio - bitops_ratio, budget_ratio))

    if "bitops_budget" in train_stats:
        budget = float(train_stats["bitops_budget"])
        total_bitops = float(resource_stats.get("total_bitops", 0.0))
        return _clamp01(_safe_div(budget - total_bitops, budget))

    return 1.0


def _residual_blocks(resource_stats: dict[str, Any]) -> list[dict[str, Any]]:
    return [block for block in resource_stats.get("blocks", []) if str(block.get("kind")) == "residual"]


def _layer_lookup(resource_stats: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return {str(layer["name"]): layer for layer in resource_stats.get("layers", [])}


def _make_metadata(resource_stats: dict[str, Any]) -> list[AgentBlockMetadata]:
    layers_by_name = _layer_lookup(resource_stats)
    input_area = _shape_area(resource_stats.get("input_size", []), default=32 * 32)
    metadata = []

    for index, block in enumerate(_residual_blocks(resource_stats)):
        layer_names = [str(name) for name in block.get("layer_names", [])]
        block_layers = [layers_by_name[name] for name in layer_names if name in layers_by_name]
        conv_layers = [layer for layer in block_layers if layer.get("kind") == "conv"]
        first_layer = conv_layers[0] if conv_layers else {}
        last_layer = conv_layers[-1] if conv_layers else {}
        spatial_size = _shape_area(block.get("output_shape", []), default=input_area)
        metadata.append(
            AgentBlockMetadata(
                block_id=index,
                block_name=str(block.get("block_name", f"block_{index}")),
                layer_names=layer_names,
                in_channels=int(first_layer.get("in_channels") or 0),
                out_channels=int(last_layer.get("out_channels") or 0),
                spatial_size=int(spatial_size),
                macs=int(block.get("macs", 0)),
                params=int(block.get("params", 0)),
                fp32_bitops=int(block.get("fp32_bitops", 0)),
            )
        )
    return metadata


def _block_modules(model: nn.Module) -> dict[str, nn.Module]:
    return {name: module for name, module in model.named_modules()}


def _module_weight_std(module: nn.Module) -> float:
    values = []
    for submodule in module.modules():
        if isinstance(submodule, (QuantConv2d, QuantLinear, nn.Conv2d, nn.Linear)):
            weight = getattr(submodule, "weight", None)
            if isinstance(weight, torch.Tensor) and weight.numel() > 1:
                values.append(weight.detach().float().flatten())
    if not values:
        return 0.0
    return float(torch.cat(values).std(unbiased=False).item())


def _module_gradient_norm(module: nn.Module) -> float:
    squared_norm = 0.0
    for parameter in module.parameters(recurse=True):
        if parameter.grad is not None:
            squared_norm += float(parameter.grad.detach().float().norm(2).item() ** 2)
    return squared_norm ** 0.5


def _block_bits(resource_stats: dict[str, Any], block_name: str) -> tuple[float, float]:
    for block in _residual_blocks(resource_stats):
        if block.get("block_name") == block_name:
            return float(block.get("avg_weight_bits", 32.0)), float(block.get("avg_activation_bits", 32.0))
    return 32.0, 32.0


class AgentStateBuilder:
    def __init__(self, resource_stats: dict[str, Any]) -> None:
        self.resource_stats = resource_stats
        self.metadata = _make_metadata(resource_stats)
        self.feature_names = STATE_FEATURES
        self.max_block_id = max(len(self.metadata) - 1, 1)
        self.max_channels = max([item.in_channels for item in self.metadata] + [item.out_channels for item in self.metadata] + [1])
        self.input_area = _shape_area(resource_stats.get("input_size", []), default=32 * 32)

    def build(self, model: nn.Module, train_stats: dict[str, Any] | None = None) -> torch.Tensor:
        train_stats = train_stats or {}
        modules = _block_modules(model)
        activation_ranges = _extract_stat_map(train_stats, "activation_range_ema", "activation_ranges")
        gradient_norms = _extract_stat_map(train_stats, "gradient_norms", "gradient_norm")

        raw_weight_std = {
            item.block_name: _module_weight_std(modules[item.block_name])
            for item in self.metadata
            if item.block_name in modules
        }
        raw_gradient_norm = {
            item.block_name: _dict_value(gradient_norms, item.block_name, _module_gradient_norm(modules[item.block_name]) if item.block_name in modules else 0.0)
            for item in self.metadata
        }

        max_weight_std = max(raw_weight_std.values(), default=1.0) or 1.0
        max_activation_range = max([float(value) for value in activation_ranges.values()], default=1.0) or 1.0
        max_gradient_norm = max(raw_gradient_norm.values(), default=1.0) or 1.0
        remaining_budget = _remaining_bitops_budget(self.resource_stats, train_stats)
        epoch_ratio = _epoch_ratio(train_stats)

        rows = []
        for item in self.metadata:
            weight_bits, activation_bits = _block_bits(self.resource_stats, item.block_name)
            activation_range = _dict_value(activation_ranges, item.block_name)
            gradient_norm = raw_gradient_norm.get(item.block_name, 0.0)
            block = next(block for block in _residual_blocks(self.resource_stats) if block.get("block_name") == item.block_name)
            rows.append(
                [
                    _safe_div(item.block_id, self.max_block_id),
                    _safe_div(item.in_channels, self.max_channels),
                    _safe_div(item.out_channels, self.max_channels),
                    _safe_div(item.spatial_size, self.input_area),
                    _safe_div(weight_bits, 32.0),
                    _safe_div(activation_bits, 32.0),
                    _safe_div(raw_weight_std.get(item.block_name, 0.0), max_weight_std),
                    _safe_div(activation_range, max_activation_range),
                    _safe_div(gradient_norm, max_gradient_norm),
                    float(block.get("bitops_ratio", 0.0)),
                    remaining_budget,
                    epoch_ratio,
                ]
            )

        device = next(model.parameters()).device
        states = torch.tensor(rows, dtype=torch.float32, device=device)
        return torch.nan_to_num(states, nan=0.0, posinf=1.0, neginf=0.0).clamp_(0.0, 1.0)

    def metadata_dicts(self) -> list[dict[str, Any]]:
        return [asdict(item) for item in self.metadata]


def build_agent_states(
    model: nn.Module,
    resource_tracker: dict[str, Any] | AgentStateBuilder | None = None,
    train_stats: dict[str, Any] | None = None,
) -> torch.Tensor:
    if isinstance(resource_tracker, AgentStateBuilder):
        return resource_tracker.build(model, train_stats=train_stats)
    resource_stats = resource_tracker or collect_resource_stats(model)
    return AgentStateBuilder(resource_stats).build(model, train_stats=train_stats)


def summarize_agent_states(states: torch.Tensor) -> dict[str, Any]:
    cpu_states = states.detach().float().cpu()
    return {
        "num_agents": int(cpu_states.shape[0]),
        "state_dim": int(cpu_states.shape[1]) if cpu_states.dim() == 2 else 0,
        "feature_names": list(STATE_FEATURES),
        "min": float(cpu_states.min().item()) if cpu_states.numel() else 0.0,
        "max": float(cpu_states.max().item()) if cpu_states.numel() else 0.0,
        "mean": float(cpu_states.mean().item()) if cpu_states.numel() else 0.0,
    }


def save_agent_state_debug(path: str | Path, states: torch.Tensor, builder: AgentStateBuilder) -> None:
    save_json(
        path,
        {
            "summary": summarize_agent_states(states),
            "blocks": builder.metadata_dicts(),
        },
    )
