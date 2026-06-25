from __future__ import annotations

import math

import torch
from torch import nn

from src.quantization.fake_quant import fake_quantize_uniform


class LSQFakeQuantizer(nn.Module):
    def __init__(
        self,
        bits: int,
        signed: bool = True,
        per_channel: bool = False,
        channel_dim: int = 0,
        num_channels: int | None = None,
        learnable: bool = True,
    ) -> None:
        super().__init__()
        if bits < 2:
            raise ValueError("bits must be >= 2")
        self.bits = int(bits)
        self.signed = bool(signed)
        self.per_channel = bool(per_channel)
        self.channel_dim = int(channel_dim)
        self.learnable = bool(learnable)
        if self.per_channel and num_channels is None:
            raise ValueError("num_channels is required when per_channel=True")

        self.qmin, self.qmax = self._compute_range()
        self.register_buffer("initialized", torch.tensor(False))
        scale_shape = (int(num_channels),) if self.per_channel else (1,)
        self.scale = nn.Parameter(torch.ones(scale_shape), requires_grad=self.learnable)

    def set_bits(self, bits: int) -> None:
        bits = int(bits)
        if bits == self.bits:
            return
        self.bits = bits
        self.qmin, self.qmax = self._compute_range()
        self.initialized.fill_(False)

    def _compute_range(self) -> tuple[int, int]:
        if self.signed:
            qmin = -(2 ** (self.bits - 1))
            qmax = 2 ** (self.bits - 1) - 1
        else:
            qmin = 0
            qmax = 2**self.bits - 1
        return qmin, qmax

    def _reduce_dims(self, x: torch.Tensor) -> tuple[int, ...]:
        if not self.per_channel:
            return tuple(range(x.dim()))
        return tuple(dim for dim in range(x.dim()) if dim != self.channel_dim)

    def _init_scale(self, x: torch.Tensor) -> None:
        with torch.no_grad():
            if self.per_channel:
                reduce_dims = self._reduce_dims(x)
                magnitude = x.detach().abs().mean(dim=reduce_dims)
                denom = max(math.sqrt(max(self.qmax, 1)), 1.0)
                scale = torch.clamp(magnitude / denom, min=torch.finfo(x.dtype).eps)
                self.scale.data.copy_(scale.reshape_as(self.scale))
            else:
                magnitude = x.detach().abs().mean()
                denom = max(math.sqrt(max(self.qmax, 1)), 1.0)
                scale = torch.clamp(magnitude / denom, min=torch.finfo(x.dtype).eps)
                self.scale.data.copy_(scale.reshape_as(self.scale))
            self.initialized.fill_(True)

    def _reshape_scale(self, x: torch.Tensor) -> torch.Tensor:
        if self.per_channel:
            shape = [1] * x.dim()
            shape[self.channel_dim] = -1
            return self.scale.view(*shape)
        return self.scale

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not bool(self.initialized):
            self._init_scale(x)

        scale = torch.clamp(self._reshape_scale(x), min=torch.finfo(x.dtype).eps)
        return fake_quantize_uniform(x, scale, self.qmin, self.qmax)


class LSQWeightQuantizer(LSQFakeQuantizer):
    def __init__(
        self,
        bits: int,
        per_channel: bool = True,
        channel_dim: int = 0,
        num_channels: int | None = None,
    ) -> None:
        super().__init__(
            bits=bits,
            signed=True,
            per_channel=per_channel,
            channel_dim=channel_dim,
            num_channels=num_channels,
            learnable=True,
        )


class LSQActivationQuantizer(LSQFakeQuantizer):
    def __init__(self, bits: int, signed: bool = True) -> None:
        super().__init__(bits=bits, signed=signed, per_channel=False, channel_dim=0, learnable=True)
