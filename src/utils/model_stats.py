from __future__ import annotations

from dataclasses import asdict, dataclass

import torch
from torch import nn

from src.quantization.quant_layers import QuantConv2d, QuantLinear


@dataclass
class LayerStat:
    name: str
    kind: str
    params: int
    w_bits: int
    a_bits: int
    macs: int
    activations: int
    model_bits: int
    bitops: int


def _get_bits(module: nn.Module) -> tuple[int, int]:
    weight_bits = int(getattr(module, "w_bits", 32))
    activation_bits = int(getattr(module, "a_bits", 32))
    return weight_bits, activation_bits


def _param_count(module: nn.Module) -> int:
    total = 0
    weight = getattr(module, "weight", None)
    bias = getattr(module, "bias", None)
    if isinstance(weight, torch.Tensor):
        total += weight.numel()
    if isinstance(bias, torch.Tensor):
        total += bias.numel()
    return total


def _conv_macs(module: nn.Module, output: torch.Tensor) -> int:
    if output.dim() != 4:
        return 0
    batch, out_channels, out_h, out_w = output.shape
    if isinstance(module, QuantConv2d):
        kernel_h, kernel_w = module.kernel_size
        in_channels = module.in_channels
        groups = module.groups
    else:
        kernel_h, kernel_w = module.kernel_size if isinstance(module.kernel_size, tuple) else (module.kernel_size, module.kernel_size)
        in_channels = module.in_channels
        groups = module.groups
    return int(batch * out_channels * out_h * out_w * (in_channels // groups) * kernel_h * kernel_w)


def _linear_macs(module: nn.Module, output: torch.Tensor) -> int:
    if output.dim() != 2:
        output = output.reshape(output.shape[0], -1)
    batch = int(output.shape[0])
    out_features = int(output.shape[-1])
    in_features = int(getattr(module, "in_features", 0))
    return int(batch * in_features * out_features)


def collect_model_stats(model: nn.Module, input_size: tuple[int, int, int] = (3, 32, 32)) -> dict:
    was_training = model.training
    model.eval()

    layer_stats: list[LayerStat] = []
    handles = []

    def make_hook(name: str, module: nn.Module):
        def hook(_module: nn.Module, _inputs: tuple[torch.Tensor, ...], output):
            tensor = output[0] if isinstance(output, (tuple, list)) else output
            if not isinstance(tensor, torch.Tensor):
                return

            if isinstance(module, (QuantConv2d, nn.Conv2d)):
                macs = _conv_macs(module, tensor)
                kind = "conv"
            elif isinstance(module, (QuantLinear, nn.Linear)):
                macs = _linear_macs(module, tensor)
                kind = "linear"
            else:
                return

            w_bits, a_bits = _get_bits(module)
            params = _param_count(module)
            weight_params = int(getattr(module, "weight", torch.empty(0)).numel())
            bias_params = int(getattr(module, "bias", torch.empty(0)).numel()) if getattr(module, "bias", None) is not None else 0
            model_bits = weight_params * w_bits + bias_params * 32
            bitops = macs * w_bits * a_bits
            layer_stats.append(
                LayerStat(
                    name=name,
                    kind=kind,
                    params=params,
                    w_bits=w_bits,
                    a_bits=a_bits,
                    macs=macs,
                    activations=int(tensor.numel()),
                    model_bits=model_bits,
                    bitops=bitops,
                )
            )

        return hook

    for name, module in model.named_modules():
        if isinstance(module, (QuantConv2d, QuantLinear, nn.Conv2d, nn.Linear)):
            handles.append(module.register_forward_hook(make_hook(name, module)))

    device = next(model.parameters()).device
    with torch.no_grad():
        dummy = torch.zeros((1, *input_size), device=device)
        model(dummy)

    for handle in handles:
        handle.remove()
    if was_training:
        model.train()

    total_params = sum(stat.params for stat in layer_stats)
    total_model_bits = sum(stat.model_bits for stat in layer_stats)
    total_macs = sum(stat.macs for stat in layer_stats)
    total_bitops = sum(stat.bitops for stat in layer_stats)
    total_weight_params = sum(int(getattr(module, "weight", torch.empty(0)).numel()) for module in model.modules() if isinstance(module, (QuantConv2d, QuantLinear, nn.Conv2d, nn.Linear)))
    total_weight_bits = sum(
        int(getattr(module, "weight", torch.empty(0)).numel()) * _get_bits(module)[0]
        for module in model.modules()
        if isinstance(module, (QuantConv2d, QuantLinear, nn.Conv2d, nn.Linear))
    )
    total_activation_elements = sum(stat.activations for stat in layer_stats)
    total_activation_bits = sum(stat.activations * stat.a_bits for stat in layer_stats)

    fp32_model_bits = total_params * 32
    fp32_bitops = total_macs * 32 * 32
    model_size_mb = total_model_bits / 8 / 1024 / 1024
    fp32_model_size_mb = fp32_model_bits / 8 / 1024 / 1024
    compression_ratio = fp32_model_bits / total_model_bits if total_model_bits else 1.0
    avg_weight_bits = total_weight_bits / total_weight_params if total_weight_params else 32.0
    avg_activation_bits = total_activation_bits / total_activation_elements if total_activation_elements else 32.0
    bitops_ratio = total_bitops / fp32_bitops if fp32_bitops else 1.0

    return {
        "input_size": list(input_size),
        "total_params": total_params,
        "total_macs": total_macs,
        "total_model_bits": total_model_bits,
        "total_bitops": total_bitops,
        "fp32_model_bits": fp32_model_bits,
        "fp32_bitops": fp32_bitops,
        "model_size_mb": model_size_mb,
        "fp32_model_size_mb": fp32_model_size_mb,
        "compression_ratio": compression_ratio,
        "avg_weight_bits": avg_weight_bits,
        "avg_activation_bits": avg_activation_bits,
        "bitops_ratio": bitops_ratio,
        "layers": [asdict(stat) for stat in layer_stats],
    }
