from __future__ import annotations

import math

import torch
from torch import nn


def _ste_round(value: torch.Tensor) -> torch.Tensor:
    return value + (value.round() - value).detach()


class LSQFakeQuantizer(nn.Module):
    def __init__(self, bits: int, *, per_channel: bool = False, channels: int | None = None, channel_axis: int = 0) -> None:
        super().__init__()
        if bits not in (2, 4, 8, 16):
            raise ValueError("unsupported bit width")
        if per_channel and (channels is None or channels < 1):
            raise ValueError("channels is required for per-channel quantization")
        self.bits = bits
        self.per_channel = per_channel
        self.channel_axis = channel_axis
        shape = (channels,) if per_channel else (1,)
        self.scale = nn.Parameter(torch.ones(shape))
        self.register_buffer("calibrated", torch.tensor(False))
        self.register_buffer("collecting", torch.tensor(False))
        self.register_buffer("observer_sum", torch.zeros(shape))
        self.register_buffer("observer_count", torch.zeros(shape))

    def begin_calibration(self) -> None:
        self.observer_sum.zero_()
        self.observer_count.zero_()
        self.collecting.fill_(True)

    def _observe(self, value: torch.Tensor) -> None:
        with torch.no_grad():
            if self.per_channel:
                dimensions = tuple(index for index in range(value.ndim) if index != self.channel_axis)
                observed_sum = value.detach().abs().sum(dim=dimensions)
                count = value.numel() // value.shape[self.channel_axis]
                self.observer_sum.add_(observed_sum.to(self.observer_sum))
                self.observer_count.add_(count)
            else:
                self.observer_sum.add_(value.detach().abs().sum().to(self.observer_sum))
                self.observer_count.add_(value.numel())

    def end_calibration(self) -> None:
        if not bool(self.collecting):
            raise RuntimeError("calibration was not started")
        if not bool((self.observer_count > 0).all()):
            self.collecting.fill_(False)
            raise RuntimeError("calibration received no observations")
        if self.bits != 16:
            qp = 2 ** (self.bits - 1) - 1
            estimate = (self.observer_sum / self.observer_count.clamp_min(1)) * 2 / math.sqrt(qp)
            with torch.no_grad():
                self.scale.copy_(estimate.clamp_min(torch.finfo(self.scale.dtype).eps))
        self.collecting.fill_(False)
        self.calibrated.fill_(True)

    def calibrate(self, value: torch.Tensor) -> None:
        self.begin_calibration()
        self._observe(value)
        self.end_calibration()

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        if bool(self.collecting):
            self._observe(value)
            return value
        if self.bits == 16:
            return value
        qn, qp = -(2 ** (self.bits - 1)), 2 ** (self.bits - 1) - 1
        scale = self.scale.abs().clamp_min(torch.finfo(value.dtype).eps)
        if self.per_channel:
            shape = [1] * value.ndim
            shape[self.channel_axis] = scale.numel()
            scale = scale.view(shape)
        return _ste_round((value / scale).clamp(qn, qp)) * scale
