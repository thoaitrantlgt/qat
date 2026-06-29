from __future__ import annotations

from dataclasses import asdict, dataclass

import torch
from torch import nn


@dataclass(frozen=True)
class PolicyLearningConfig:
    entropy_beta: float = 0.01
    max_grad_norm: float = 5.0
    advantage_clip: float = 10.0


@dataclass
class PolicyUpdateResult:
    loss: float
    policy_loss: float
    entropy: float
    entropy_loss: float
    advantage_mean: float
    advantage_std: float

    def to_dict(self) -> dict[str, float]:
        return asdict(self)


class ReinforcePolicyLearner:
    def __init__(
        self,
        actor: nn.Module,
        optimizer: torch.optim.Optimizer,
        config: PolicyLearningConfig | None = None,
    ) -> None:
        self.actor = actor
        self.optimizer = optimizer
        self.config = config or PolicyLearningConfig()

    def update(
        self,
        log_probs: torch.Tensor,
        entropy: torch.Tensor,
        local_rewards: list[float],
        baseline: float,
    ) -> PolicyUpdateResult:
        rewards = torch.tensor(local_rewards, dtype=torch.float32, device=log_probs.device)
        if rewards.numel() != log_probs.numel():
            raise ValueError(f"Expected {log_probs.numel()} local rewards, got {rewards.numel()}.")

        advantages = rewards - float(baseline)
        advantages = torch.nan_to_num(advantages, nan=0.0, posinf=0.0, neginf=0.0)
        advantages = advantages.clamp(-self.config.advantage_clip, self.config.advantage_clip)

        policy_loss = -(log_probs * advantages.detach()).mean()
        entropy_mean = entropy.mean()
        entropy_loss = -float(self.config.entropy_beta) * entropy_mean
        loss = policy_loss + entropy_loss
        if not torch.isfinite(loss):
            raise FloatingPointError("policy loss became non-finite")

        self.optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.actor.parameters(), max_norm=self.config.max_grad_norm)
        self.optimizer.step()

        return PolicyUpdateResult(
            loss=float(loss.detach().item()),
            policy_loss=float(policy_loss.detach().item()),
            entropy=float(entropy_mean.detach().item()),
            entropy_loss=float(entropy_loss.detach().item()),
            advantage_mean=float(advantages.detach().mean().item()),
            advantage_std=float(advantages.detach().std(unbiased=False).item()) if advantages.numel() > 1 else 0.0,
        )


def ppo_clipped_loss(
    new_log_probs: torch.Tensor,
    old_log_probs: torch.Tensor,
    advantages: torch.Tensor,
    clip_eps: float = 0.2,
) -> torch.Tensor:
    ratio = torch.exp(new_log_probs - old_log_probs)
    unclipped = ratio * advantages
    clipped = torch.clamp(ratio, 1.0 - clip_eps, 1.0 + clip_eps) * advantages
    return -torch.minimum(unclipped, clipped).mean()
