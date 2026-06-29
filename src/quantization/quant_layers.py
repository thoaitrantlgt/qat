from __future__ import annotations

import math

import torch
from torch import nn
from torch.nn import functional as F

from src.quantization.lsq import LSQActivationQuantizer, LSQWeightQuantizer


class QuantConv2d(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size,
        stride=1,
        padding=0,
        dilation=1,
        groups: int = 1,
        bias: bool = True,
        w_bits: int = 8,
        a_bits: int = 8,
        per_channel_weight: bool = True,
        input_signed: bool = True,
    ) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size if isinstance(kernel_size, tuple) else (kernel_size, kernel_size)
        self.stride = stride if isinstance(stride, tuple) else (stride, stride)
        self.padding = padding if isinstance(padding, tuple) else (padding, padding)
        self.dilation = dilation if isinstance(dilation, tuple) else (dilation, dilation)
        self.groups = groups
        self.w_bits = int(w_bits)
        self.a_bits = int(a_bits)

        weight_shape = (out_channels, in_channels // groups, *self.kernel_size)
        self.weight = nn.Parameter(torch.empty(weight_shape))
        self.bias = nn.Parameter(torch.empty(out_channels)) if bias else None

        self.weight_quantizer = LSQWeightQuantizer(
            bits=self.w_bits,
            per_channel=per_channel_weight,
            channel_dim=0,
            num_channels=out_channels if per_channel_weight else None,
        )
        self.act_quantizer = LSQActivationQuantizer(bits=self.a_bits, signed=input_signed)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in = self.in_channels * self.kernel_size[0] * self.kernel_size[1] / self.groups
            bound = 1 / math.sqrt(fan_in)
            nn.init.uniform_(self.bias, -bound, bound)

    def set_bits(self, w_bits: int, a_bits: int) -> None:
        self.w_bits = int(w_bits)
        self.a_bits = int(a_bits)
        self.weight_quantizer.set_bits(self.w_bits)
        self.act_quantizer.set_bits(self.a_bits)

    def get_bits(self) -> tuple[int, int]:
        return self.w_bits, self.a_bits

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_q = self.act_quantizer(x)
        w_q = self.weight_quantizer(self.weight)
        return F.conv2d(
            x_q,
            w_q,
            self.bias,
            stride=self.stride,
            padding=self.padding,
            dilation=self.dilation,
            groups=self.groups,
        )


class QuantLinear(nn.Module):
    def __init__(self, in_features: int, out_features: int, bias: bool = True, w_bits: int = 8, a_bits: int = 8) -> None:
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.w_bits = int(w_bits)
        self.a_bits = int(a_bits)

        self.weight = nn.Parameter(torch.empty(out_features, in_features))
        self.bias = nn.Parameter(torch.empty(out_features)) if bias else None
        self.weight_quantizer = LSQWeightQuantizer(bits=self.w_bits, per_channel=True, channel_dim=0, num_channels=out_features)
        self.act_quantizer = LSQActivationQuantizer(bits=self.a_bits, signed=True)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            bound = 1 / math.sqrt(self.in_features)
            nn.init.uniform_(self.bias, -bound, bound)

    def set_bits(self, w_bits: int, a_bits: int) -> None:
        self.w_bits = int(w_bits)
        self.a_bits = int(a_bits)
        self.weight_quantizer.set_bits(self.w_bits)
        self.act_quantizer.set_bits(self.a_bits)

    def get_bits(self) -> tuple[int, int]:
        return self.w_bits, self.a_bits

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_q = self.act_quantizer(x)
        w_q = self.weight_quantizer(self.weight)
        return F.linear(x_q, w_q, self.bias)
