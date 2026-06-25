from __future__ import annotations

import torch
from torch.autograd import Function


class UniformFakeQuantizeSTE(Function):
    @staticmethod
    def forward(ctx, x: torch.Tensor, scale: torch.Tensor, qmin: int, qmax: int) -> torch.Tensor:
        scale = torch.clamp(scale, min=torch.finfo(x.dtype).eps)
        x_scaled = x / scale
        x_clipped = torch.clamp(x_scaled, qmin, qmax)
        x_rounded = torch.round(x_clipped)
        ctx.save_for_backward(x_scaled, x_rounded, scale)
        ctx.qmin = qmin
        ctx.qmax = qmax
        return x_rounded * scale

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        x_scaled, x_rounded, scale = ctx.saved_tensors
        pass_through = (x_scaled >= ctx.qmin) & (x_scaled <= ctx.qmax)
        grad_x = grad_output * pass_through.to(dtype=grad_output.dtype)

        grad_scale = None
        if ctx.needs_input_grad[1]:
            grad_scale = ((x_rounded - x_scaled) * grad_output).sum_to_size(scale.shape)

        return grad_x, grad_scale, None, None


def fake_quantize_uniform(
    x: torch.Tensor,
    scale: torch.Tensor,
    qmin: int,
    qmax: int,
) -> torch.Tensor:
    return UniformFakeQuantizeSTE.apply(x, scale, qmin, qmax)

