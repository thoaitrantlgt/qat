from __future__ import annotations

from collections.abc import Iterable
from collections.abc import Mapping
from collections.abc import Sequence
import re

from torch import nn

from src.agents.action_space import DEFAULT_ACTION_BITS
from src.quantization.quant_layers import QuantConv2d, QuantLinear


ACTION_BITS = DEFAULT_ACTION_BITS
RESIDUAL_BLOCK_RE = re.compile(r"^layer\d+\.\d+$")


def iter_quant_layers(model: nn.Module) -> Iterable[nn.Module]:
    for module in model.modules():
        if isinstance(module, (QuantConv2d, QuantLinear)):
            yield module


def set_uniform_bit_widths(model: nn.Module, weight_bits: int, activation_bits: int) -> None:
    for module in iter_quant_layers(model):
        module.set_bits(weight_bits, activation_bits)


def set_layer_bit_widths(model: nn.Module, layer_bits: dict[str, tuple[int, int]]) -> None:
    for name, module in model.named_modules():
        if name in layer_bits and isinstance(module, (QuantConv2d, QuantLinear)):
            weight_bits, activation_bits = layer_bits[name]
            module.set_bits(weight_bits, activation_bits)


def iter_residual_blocks(model: nn.Module) -> Iterable[tuple[str, nn.Module]]:
    for name, module in model.named_modules():
        if RESIDUAL_BLOCK_RE.match(name):
            yield name, module


def _normalize_bit_pair(action, action_bits: Sequence[tuple[int, int]] | None = None) -> tuple[int, int]:
    choices = tuple(action_bits or ACTION_BITS)
    if isinstance(action, int):
        if action < 0 or action >= len(choices):
            raise ValueError(f"Invalid action index {action}; expected 0..{len(choices) - 1}.")
        return choices[action]

    if isinstance(action, Mapping):
        weight_bits = action.get("weight_bits", action.get("w_bits"))
        activation_bits = action.get("activation_bits", action.get("a_bits"))
        if weight_bits is None or activation_bits is None:
            raise KeyError("policy entries must contain weight_bits/activation_bits or w_bits/a_bits")
        return int(weight_bits), int(activation_bits)

    if isinstance(action, Sequence) and len(action) == 2:
        return int(action[0]), int(action[1])

    raise TypeError(f"Unsupported block action {action!r}; use an action index or (weight_bits, activation_bits).")


def _set_block_bits(block: nn.Module, weight_bits: int, activation_bits: int) -> int:
    changed = 0
    for module in block.modules():
        if isinstance(module, (QuantConv2d, QuantLinear)):
            module.set_bits(weight_bits, activation_bits)
            changed += 1
    return changed


def apply_block_policy(
    model: nn.Module,
    block_actions,
    action_bits: Sequence[tuple[int, int]] | None = None,
) -> list[dict[str, int | str]]:
    blocks = list(iter_residual_blocks(model))
    if isinstance(block_actions, Mapping):
        action_items = [(name, module, block_actions[name]) for name, module in blocks if name in block_actions]
    else:
        if hasattr(block_actions, "detach"):
            actions = block_actions.detach().cpu().reshape(-1).tolist()
        else:
            actions = list(block_actions)
        if len(actions) != len(blocks):
            raise ValueError(f"Expected {len(blocks)} block actions, got {len(actions)}.")
        action_items = [(name, module, action) for (name, module), action in zip(blocks, actions)]

    applied = []
    for block_name, block, action in action_items:
        weight_bits, activation_bits = _normalize_bit_pair(action, action_bits=action_bits)
        changed = _set_block_bits(block, weight_bits, activation_bits)
        applied.append(
            {
                "block_name": block_name,
                "weight_bits": weight_bits,
                "activation_bits": activation_bits,
                "quant_layers": changed,
            }
        )
    return applied
