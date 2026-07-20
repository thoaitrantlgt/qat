from __future__ import annotations

from contextlib import contextmanager

import torch
from torch import nn
from torch.nn import functional as F

from .lsq import LSQFakeQuantizer
from .policy import MixedPrecisionPolicy, PrecisionAction
from .registry import QuantGroup


def _resolve(model: nn.Module, path: str) -> tuple[nn.Module, str]:
    parts = path.split(".")
    parent = model
    for part in parts[:-1]:
        parent = getattr(parent, part)
    return parent, parts[-1]


class QuantizedLinear(nn.Module):
    def __init__(self, linear: nn.Linear, action: PrecisionAction) -> None:
        super().__init__()
        self.linear = linear
        self.action = action
        self.weight_quantizer = LSQFakeQuantizer(action.weight_bits, per_channel=True, channels=linear.out_features, channel_axis=0)
        self.activation_quantizer = LSQFakeQuantizer(action.activation_bits)

    def set_action(self, action: PrecisionAction) -> None:
        if action == self.action:
            return
        self.action = action
        device = self.linear.weight.device
        self.weight_quantizer = LSQFakeQuantizer(action.weight_bits, per_channel=True, channels=self.linear.out_features, channel_axis=0).to(device)
        self.activation_quantizer = LSQFakeQuantizer(action.activation_bits).to(device)

    def calibrate(self, activation: torch.Tensor | None = None) -> None:
        self.weight_quantizer.calibrate(self.linear.weight.detach())
        if activation is not None:
            self.activation_quantizer.calibrate(activation.detach())

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        value = self.activation_quantizer(value)
        weight = self.weight_quantizer(self.linear.weight)
        return F.linear(value, weight, self.linear.bias)


class QuantizedConv1D(nn.Module):
    """LSQ wrapper for Hugging Face GPT-style Conv1D ([in_features, out_features])."""

    def __init__(self, conv: nn.Module, action: PrecisionAction) -> None:
        super().__init__()
        self.conv = conv
        self.action = action
        self.weight_quantizer = LSQFakeQuantizer(action.weight_bits, per_channel=True, channels=conv.nf, channel_axis=1)
        self.activation_quantizer = LSQFakeQuantizer(action.activation_bits)

    def set_action(self, action: PrecisionAction) -> None:
        if action == self.action:
            return
        self.action = action
        device = self.conv.weight.device
        self.weight_quantizer = LSQFakeQuantizer(action.weight_bits, per_channel=True, channels=self.conv.nf, channel_axis=1).to(device)
        self.activation_quantizer = LSQFakeQuantizer(action.activation_bits).to(device)

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        value = self.activation_quantizer(value)
        weight = self.weight_quantizer(self.conv.weight)
        return F.linear(value, weight.transpose(0, 1), self.conv.bias)


def _validate(registry: list[QuantGroup], policy: MixedPrecisionPolicy) -> None:
    expected = {group.name for group in registry}
    actual = set(policy.actions)
    if expected != actual:
        raise ValueError(f"policy must provide exactly one action per quantization group; missing={sorted(expected-actual)}, extra={sorted(actual-expected)}")


def inject_quantizers(model: nn.Module, registry: list[QuantGroup], policy: MixedPrecisionPolicy) -> nn.Module:
    _validate(registry, policy)
    for group in registry:
        parent, attribute = _resolve(model, group.module_path)
        module = getattr(parent, attribute)
        if isinstance(module, (QuantizedLinear, QuantizedConv1D)):
            module.set_action(policy.actions[group.name])
        elif group.module_type == "linear" and isinstance(module, nn.Linear):
            setattr(parent, attribute, QuantizedLinear(module, policy.actions[group.name]))
        elif group.module_type == "conv1d" and module.__class__.__name__ == "Conv1D":
            setattr(parent, attribute, QuantizedConv1D(module, policy.actions[group.name]))
        else:
            raise TypeError(f"registered module changed type: {group.module_path}")
    return model


@contextmanager
def temporary_policy(model: nn.Module, registry: list[QuantGroup], policy: MixedPrecisionPolicy):
    originals = {}
    for group in registry:
        parent, attribute = _resolve(model, group.module_path)
        module = getattr(parent, attribute)
        if isinstance(module, (QuantizedLinear, QuantizedConv1D)):
            originals[group.module_path] = (module, module.action, {name: value.detach().clone() for name, value in module.state_dict().items()})
        else:
            originals[group.module_path] = (module, None, None)
    inject_quantizers(model, registry, policy)
    try:
        yield model
    finally:
        for path, (module, action, state) in originals.items():
            if action is not None:
                module.set_action(action)
                module.load_state_dict(state)
            parent, attribute = _resolve(model, path)
            setattr(parent, attribute, module)
