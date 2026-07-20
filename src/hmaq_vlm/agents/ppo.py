from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn


@dataclass(frozen=True)
class PPOHyperparameters:
    clip: float = 0.2
    gae_lambda: float = 0.95
    optimization_epochs: int = 4
    actor_lr: float = 3.0e-4
    critic_lr: float = 1.0e-3
    entropy: float = 0.01


class SharedActor(nn.Module):
    def __init__(self, state_dim: int, actions: int = 16) -> None:
        super().__init__()
        self.network = nn.Sequential(nn.Linear(state_dim, 64), nn.Tanh(), nn.Linear(64, actions))

    def forward(self, state: torch.Tensor, action_mask: torch.Tensor | None = None) -> torch.Tensor:
        logits = self.network(state)
        return logits.masked_fill(~action_mask.bool(), -torch.inf) if action_mask is not None else logits


class CentralCritic(nn.Module):
    def __init__(self, state_dim: int) -> None:
        super().__init__()
        self.network = nn.Sequential(nn.Linear(state_dim, 64), nn.Tanh(), nn.Linear(64, 1))

    def forward(self, global_state: torch.Tensor) -> torch.Tensor:
        return self.network(global_state).squeeze(-1)
