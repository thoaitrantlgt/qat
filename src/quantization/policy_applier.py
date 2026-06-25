from __future__ import annotations

from collections.abc import Iterable

from torch import nn

from src.quantization.quant_layers import QuantConv2d, QuantLinear


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

