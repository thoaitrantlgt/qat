from __future__ import annotations

from collections.abc import Callable, Iterable
from typing import Any

import torch
from torch import nn

from hmaq_vlm.losses import consistency_losses
from hmaq_vlm.quantization import QuantizedConv1D, QuantizedLinear


def build_optimizer(model: nn.Module, *, projector_lr: float, vision_lr: float, language_lr: float, weight_decay: float = 0.01) -> torch.optim.Optimizer:
    groups = []
    for prefix, lr in (("projector.", projector_lr), ("vision_encoder.", vision_lr), ("language_model.", language_lr)):
        parameters = [parameter for name, parameter in model.named_parameters() if name.startswith(prefix) and parameter.requires_grad]
        if parameters:
            groups.append({"params": parameters, "lr": lr})
    if not groups:
        raise ValueError("model has no trainable VLM parameters")
    return torch.optim.AdamW(groups, weight_decay=weight_decay)


def calibrate_quantizers(model: nn.Module, batches: Iterable[dict[str, torch.Tensor]], forward_fn: Callable[[nn.Module, dict[str, torch.Tensor]], Any]) -> None:
    quantized = [module for module in model.modules() if isinstance(module, (QuantizedLinear, QuantizedConv1D))]
    if not quantized:
        raise ValueError("model has no injected quantizers")
    for module in quantized:
        base = module.linear if isinstance(module, QuantizedLinear) else module.conv
        module.weight_quantizer.calibrate(base.weight.detach())
        module.activation_quantizer.begin_calibration()
    was_training = model.training
    model.eval()
    try:
        with torch.no_grad():
            for batch in batches:
                forward_fn(model, batch)
    finally:
        for module in quantized:
            module.activation_quantizer.end_calibration()
        model.train(was_training)
    if not all(bool(module.weight_quantizer.calibrated) and bool(module.activation_quantizer.calibrated) for module in quantized):
        raise RuntimeError("activation calibration did not reach every quantizer")


def caption_training_step(
    model: nn.Module,
    batch: dict[str, torch.Tensor],
    optimizer: torch.optim.Optimizer,
    *,
    teacher: nn.Module | None = None,
    prefix_weight: float = 1.0,
    kl_weight: float = 1.0,
    temperature: float = 2.0,
    gradient_clip: float = 1.0,
) -> dict[str, float]:
    model.train()
    optimizer.zero_grad(set_to_none=True)
    use_amp = torch.cuda.is_available()
    with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=use_amp):
        output = model(**batch)
        if output.loss is None:
            raise ValueError("caption training requires labels")
        total = output.loss
        prefix_loss = output.loss.new_zeros(())
        kl_loss = output.loss.new_zeros(())
        if teacher is not None:
            teacher.eval()
            with torch.no_grad():
                teacher_output = teacher(**batch)
            logit_mask = output.labels[:, 1:].ne(-100)
            prefix_loss, kl_loss = consistency_losses(
                output.visual_prefix,
                teacher_output.visual_prefix,
                output.logits[:, :-1],
                teacher_output.logits[:, :-1],
                temperature,
                logit_mask,
            )
            total = total + prefix_weight * prefix_loss + kl_weight * kl_loss
    if not bool(torch.isfinite(total)):
        raise FloatingPointError("non-finite QAT loss; policy rejected without precision fallback")
    total.backward()
    gradient_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clip)
    if not bool(torch.isfinite(gradient_norm)):
        optimizer.zero_grad(set_to_none=True)
        raise FloatingPointError("non-finite gradient norm; policy rejected without precision fallback")
    optimizer.step()
    return {"loss": float(total.detach()), "caption_loss": float(output.loss.detach()), "prefix_mse": float(prefix_loss.detach()), "logit_kl": float(kl_loss.detach()), "gradient_norm": float(gradient_norm.detach())}
