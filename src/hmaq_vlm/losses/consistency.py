from __future__ import annotations

import torch
from torch.nn import functional as F


def consistency_losses(
    student_prefix: torch.Tensor,
    teacher_prefix: torch.Tensor,
    student_logits: torch.Tensor,
    teacher_logits: torch.Tensor,
    temperature: float = 2.0,
    logit_mask: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    if temperature <= 0:
        raise ValueError("temperature must be positive")
    teacher_prefix = teacher_prefix.detach()
    teacher_logits = teacher_logits.detach()
    prefix_mse = F.mse_loss(F.normalize(student_prefix, dim=-1), F.normalize(teacher_prefix, dim=-1))
    log_student = F.log_softmax(student_logits / temperature, dim=-1)
    log_teacher = F.log_softmax(teacher_logits / temperature, dim=-1)
    prob_teacher = log_teacher.exp()
    per_position = (prob_teacher * (log_teacher - log_student)).sum(dim=-1)
    if logit_mask is not None:
        if logit_mask.shape != per_position.shape:
            raise ValueError("logit mask must match batch and sequence dimensions")
        selected = per_position.masked_select(logit_mask.bool())
        logit_kl = selected.mean() * (temperature**2) if selected.numel() else per_position.sum() * 0
    else:
        logit_kl = per_position.mean() * (temperature**2)
    return prefix_mse, logit_kl.clamp_min(0)
