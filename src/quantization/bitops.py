from __future__ import annotations

from dataclasses import asdict, dataclass
import re
from collections import OrderedDict
from collections.abc import Callable

import torch
from torch import nn

from src.quantization.quant_layers import QuantConv2d, QuantLinear


@dataclass
class LayerResource:
    name: str
    block_name: str
    kind: str
    input_shape: list[int]
    output_shape: list[int]
    kernel_size: list[int] | None
    in_channels: int | None
    out_channels: int | None
    in_features: int | None
    out_features: int | None
    groups: int
    params: int
    weight_params: int
    bias_params: int
    activations: int
    w_bits: int
    a_bits: int
    macs: int
    model_bits: int
    bitops: int


@dataclass
class BlockResource:
    block_id: int
    block_name: str
    kind: str
    layer_names: list[str]
    input_shape: list[int]
    output_shape: list[int]
    layer_count: int
    params: int
    weight_params: int
    bias_params: int
    activations: int
    macs: int
    model_bits: int
    bitops: int
    fp32_model_bits: int
    fp32_bitops: int
    w8a8_bitops: int
    compression_ratio: float
    bitops_ratio: float
    w8a8_bitops_ratio: float
    avg_weight_bits: float
    avg_activation_bits: float


BLOCK_RE = re.compile(r"^(layer\d+\.\d+)(?:\..+)?$")


def _shape_list(value: torch.Tensor | tuple[int, ...] | list[int]) -> list[int]:
    if isinstance(value, torch.Tensor):
        return [int(dim) for dim in value.shape]
    return [int(dim) for dim in value]


def _layer_kind(module: nn.Module) -> str:
    if isinstance(module, (QuantConv2d, nn.Conv2d)):
        return "conv"
    if isinstance(module, (QuantLinear, nn.Linear)):
        return "linear"
    return module.__class__.__name__.lower()


def _block_name(name: str) -> str:
    if name == "conv1":
        return "stem"
    if name == "fc":
        return "head"
    match = BLOCK_RE.match(name)
    if match:
        return match.group(1)
    if "." in name:
        return name.rsplit(".", 1)[0]
    return name


def _get_bits(module: nn.Module) -> tuple[int, int]:
    return int(getattr(module, "w_bits", 32)), int(getattr(module, "a_bits", 32))


def _param_counts(module: nn.Module) -> tuple[int, int, int]:
    weight = getattr(module, "weight", None)
    bias = getattr(module, "bias", None)
    weight_params = int(weight.numel()) if isinstance(weight, torch.Tensor) else 0
    bias_params = int(bias.numel()) if isinstance(bias, torch.Tensor) else 0
    return weight_params + bias_params, weight_params, bias_params


def _conv_macs(module: nn.Module, input_tensor: torch.Tensor, output_tensor: torch.Tensor) -> int:
    if output_tensor.dim() != 4:
        return 0
    batch, out_channels, out_h, out_w = output_tensor.shape
    kernel_h, kernel_w = (
        module.kernel_size if isinstance(module.kernel_size, tuple) else (module.kernel_size, module.kernel_size)
    )
    in_channels = int(getattr(module, "in_channels", input_tensor.shape[1] if input_tensor.dim() > 1 else 0))
    groups = int(getattr(module, "groups", 1))
    return int(batch * out_channels * out_h * out_w * (in_channels // groups) * kernel_h * kernel_w)


def _linear_macs(module: nn.Module, input_tensor: torch.Tensor, output_tensor: torch.Tensor) -> int:
    if output_tensor.dim() < 2:
        return 0
    batch = int(output_tensor.shape[0])
    in_features = int(getattr(module, "in_features", input_tensor.reshape(batch, -1).shape[-1]))
    out_features = int(output_tensor.shape[-1])
    return int(batch * in_features * out_features)


def _record_to_dict(record: LayerResource | BlockResource) -> dict:
    return asdict(record)


def collect_resource_stats(model: nn.Module, input_size: tuple[int, int, int] = (3, 32, 32)) -> dict:
    was_training = model.training
    model.eval()

    layer_resources: list[LayerResource] = []
    block_resources: OrderedDict[str, dict] = OrderedDict()
    handles = []

    def ensure_block(name: str) -> dict:
        record = block_resources.get(name)
        if record is None:
            record = {
                "block_name": name,
                "kind": "residual" if name.startswith("layer") else name,
                "layer_names": [],
                "input_shape": None,
                "output_shape": None,
                "params": 0,
                "weight_params": 0,
                "bias_params": 0,
                "activations": 0,
                "macs": 0,
                "model_bits": 0,
                "bitops": 0,
                "fp32_model_bits": 0,
                "fp32_bitops": 0,
                "w8a8_bitops": 0,
                "weight_bits_total": 0,
                "activation_bits_total": 0,
                "block_id": len(block_resources),
            }
            block_resources[name] = record
        return record

    def make_hook(name: str, module: nn.Module) -> Callable:
        def hook(_module: nn.Module, inputs: tuple[torch.Tensor, ...], output):
            input_tensor = inputs[0] if inputs else None
            output_tensor = output[0] if isinstance(output, (tuple, list)) else output
            if not isinstance(input_tensor, torch.Tensor) or not isinstance(output_tensor, torch.Tensor):
                return

            kind = _layer_kind(module)
            if kind == "conv":
                macs = _conv_macs(module, input_tensor, output_tensor)
                in_channels = int(getattr(module, "in_channels", 0))
                out_channels = int(getattr(module, "out_channels", 0))
                in_features = None
                out_features = None
                kernel_size = [
                    int(dim)
                    for dim in (module.kernel_size if isinstance(module.kernel_size, tuple) else (module.kernel_size, module.kernel_size))
                ]
                groups = int(getattr(module, "groups", 1))
            elif kind == "linear":
                macs = _linear_macs(module, input_tensor, output_tensor)
                in_channels = None
                out_channels = None
                in_features = int(getattr(module, "in_features", input_tensor.reshape(input_tensor.shape[0], -1).shape[-1]))
                out_features = int(getattr(module, "out_features", output_tensor.shape[-1]))
                kernel_size = None
                groups = 1
            else:
                return

            w_bits, a_bits = _get_bits(module)
            params, weight_params, bias_params = _param_counts(module)
            activations = int(output_tensor.numel())
            model_bits = weight_params * w_bits + bias_params * 32
            bitops = macs * w_bits * a_bits
            block_name = _block_name(name)
            layer_resource = LayerResource(
                name=name,
                block_name=block_name,
                kind=kind,
                input_shape=_shape_list(input_tensor),
                output_shape=_shape_list(output_tensor),
                kernel_size=kernel_size,
                in_channels=in_channels,
                out_channels=out_channels,
                in_features=in_features,
                out_features=out_features,
                groups=groups,
                params=params,
                weight_params=weight_params,
                bias_params=bias_params,
                activations=activations,
                w_bits=w_bits,
                a_bits=a_bits,
                macs=macs,
                model_bits=model_bits,
                bitops=bitops,
            )
            layer_resources.append(layer_resource)

            block = ensure_block(block_name)
            block["layer_names"].append(name)
            block["input_shape"] = block["input_shape"] or _shape_list(input_tensor)
            block["output_shape"] = _shape_list(output_tensor)
            block["params"] += params
            block["weight_params"] += weight_params
            block["bias_params"] += bias_params
            block["activations"] += activations
            block["macs"] += macs
            block["model_bits"] += model_bits
            block["bitops"] += bitops
            block["fp32_model_bits"] += params * 32
            block["fp32_bitops"] += macs * 32 * 32
            block["w8a8_bitops"] += macs * 8 * 8
            block["weight_bits_total"] += weight_params * w_bits
            block["activation_bits_total"] += activations * a_bits

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

    total_params = sum(resource.params for resource in layer_resources)
    total_weight_params = sum(resource.weight_params for resource in layer_resources)
    total_bias_params = sum(resource.bias_params for resource in layer_resources)
    total_activations = sum(resource.activations for resource in layer_resources)
    total_macs = sum(resource.macs for resource in layer_resources)
    total_model_bits = sum(resource.model_bits for resource in layer_resources)
    total_bitops = sum(resource.bitops for resource in layer_resources)
    total_weight_bits = sum(resource.weight_params * resource.w_bits for resource in layer_resources)
    total_activation_bits = sum(resource.activations * resource.a_bits for resource in layer_resources)

    fp32_model_bits = total_params * 32
    fp32_bitops = total_macs * 32 * 32
    w8a8_bitops = total_macs * 8 * 8
    model_size_mb = total_model_bits / 8 / 1024 / 1024
    fp32_model_size_mb = fp32_model_bits / 8 / 1024 / 1024
    compression_ratio = fp32_model_bits / total_model_bits if total_model_bits else 1.0
    avg_weight_bits = total_weight_bits / total_weight_params if total_weight_params else 32.0
    avg_activation_bits = total_activation_bits / total_activations if total_activations else 32.0
    bitops_ratio = total_bitops / fp32_bitops if fp32_bitops else 1.0
    w8a8_bitops_ratio = total_bitops / w8a8_bitops if w8a8_bitops else 1.0

    blocks: list[dict] = []
    for block_name, block in block_resources.items():
        block_model_bits = int(block["model_bits"])
        block_bitops = int(block["bitops"])
        block_fp32_model_bits = int(block["fp32_model_bits"])
        block_fp32_bitops = int(block["fp32_bitops"])
        block_w8a8_bitops = int(block["w8a8_bitops"])
        block_weight_params = int(block["weight_params"])
        block_activations = int(block["activations"])
        block_weight_bits_total = int(block["weight_bits_total"])
        block_activation_bits_total = int(block["activation_bits_total"])
        blocks.append(
            _record_to_dict(
                BlockResource(
                    block_id=int(block["block_id"]),
                    block_name=block_name,
                    kind=str(block["kind"]),
                    layer_names=list(block["layer_names"]),
                    input_shape=list(block["input_shape"] or []),
                    output_shape=list(block["output_shape"] or []),
                    layer_count=len(block["layer_names"]),
                    params=int(block["params"]),
                    weight_params=block_weight_params,
                    bias_params=int(block["bias_params"]),
                    activations=block_activations,
                    macs=int(block["macs"]),
                    model_bits=block_model_bits,
                    bitops=block_bitops,
                    fp32_model_bits=block_fp32_model_bits,
                    fp32_bitops=block_fp32_bitops,
                    w8a8_bitops=block_w8a8_bitops,
                    compression_ratio=(block_fp32_model_bits / block_model_bits if block_model_bits else 1.0),
                    bitops_ratio=(block_bitops / block_fp32_bitops if block_fp32_bitops else 1.0),
                    w8a8_bitops_ratio=(block_bitops / block_w8a8_bitops if block_w8a8_bitops else 1.0),
                    avg_weight_bits=(block_weight_bits_total / block_weight_params if block_weight_params else 32.0),
                    avg_activation_bits=(block_activation_bits_total / block_activations if block_activations else 32.0),
                )
            )
        )

    return {
        "input_size": list(input_size),
        "total_params": total_params,
        "total_weight_params": total_weight_params,
        "total_bias_params": total_bias_params,
        "total_activations": total_activations,
        "total_macs": total_macs,
        "total_model_bits": total_model_bits,
        "total_bitops": total_bitops,
        "fp32_model_bits": fp32_model_bits,
        "fp32_bitops": fp32_bitops,
        "w8a8_bitops": w8a8_bitops,
        "model_size_mb": model_size_mb,
        "fp32_model_size_mb": fp32_model_size_mb,
        "compression_ratio": compression_ratio,
        "avg_weight_bits": avg_weight_bits,
        "avg_activation_bits": avg_activation_bits,
        "bitops_ratio": bitops_ratio,
        "w8a8_bitops_ratio": w8a8_bitops_ratio,
        "layer_count": len(layer_resources),
        "block_count": len(blocks),
        "layers": [_record_to_dict(resource) for resource in layer_resources],
        "blocks": blocks,
    }
