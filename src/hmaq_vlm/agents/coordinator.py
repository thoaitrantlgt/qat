from __future__ import annotations

import torch
from torch import nn


class ModalityCoordinator(nn.Module):
    """Differentiably distributes the residual budget after hard modality minima."""

    def __init__(self, input_dim: int, hidden_dim: int = 32, minimums: tuple[float, float, float] = (0.10, 0.05, 0.30)) -> None:
        super().__init__()
        if sum(minimums) >= 1 or any(value < 0 for value in minimums):
            raise ValueError("modality minimums must be non-negative and sum below one")
        self.network = nn.Sequential(nn.Linear(input_dim, hidden_dim), nn.Tanh(), nn.Linear(hidden_dim, 3))
        self.register_buffer("minimums", torch.tensor(minimums))

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        residual = 1.0 - self.minimums.sum()
        return self.minimums + residual * self.network(state).softmax(dim=-1)
