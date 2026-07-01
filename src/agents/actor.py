from __future__ import annotations

from collections.abc import Sequence

import torch
from torch import nn
from torch.distributions import Categorical

from src.agents.action_space import DEFAULT_ACTION_BITS, DEFAULT_BIT_CHOICES, build_action_bits

BIT_CHOICES = DEFAULT_BIT_CHOICES
ACTION_BITS = DEFAULT_ACTION_BITS


def decode_actions(
    actions: torch.Tensor | Sequence[int] | int,
    action_bits: Sequence[tuple[int, int]] | None = None,
) -> list[tuple[int, int]]:
    choices = tuple(action_bits or ACTION_BITS)
    if isinstance(actions, torch.Tensor):
        flat_actions = actions.detach().cpu().reshape(-1).tolist()
    elif isinstance(actions, int):
        flat_actions = [actions]
    else:
        flat_actions = list(actions)

    decoded = []
    for action in flat_actions:
        action_index = int(action)
        if action_index < 0 or action_index >= len(choices):
            raise ValueError(f"Invalid action index {action_index}; expected 0..{len(choices) - 1}.")
        decoded.append(choices[action_index])
    return decoded


def encode_bits(
    bits: tuple[int, int] | Sequence[int],
    action_bits: Sequence[tuple[int, int]] | None = None,
) -> int:
    choices = tuple(action_bits or ACTION_BITS)
    bit_pair = (int(bits[0]), int(bits[1]))
    if bit_pair not in choices:
        raise ValueError(f"Unsupported bit pair {bit_pair}; expected one of {choices}.")
    return choices.index(bit_pair)


class SharedActor(nn.Module):
    def __init__(
        self,
        state_dim: int,
        hidden_dim: int = 128,
        num_actions: int | None = None,
        bit_choices: Sequence[int] | None = None,
        weight_bit_choices: Sequence[int] | None = None,
        activation_bit_choices: Sequence[int] | None = None,
        action_bits: Sequence[tuple[int, int]] | None = None,
    ) -> None:
        super().__init__()
        self.state_dim = int(state_dim)
        self.hidden_dim = int(hidden_dim)
        self.action_bits = tuple(
            action_bits
            or build_action_bits(
                bit_choices=bit_choices,
                weight_bit_choices=weight_bit_choices,
                activation_bit_choices=activation_bit_choices,
            )
        )
        self.num_actions = int(num_actions or len(self.action_bits))
        if self.num_actions != len(self.action_bits):
            raise ValueError(
                f"num_actions={self.num_actions} does not match action_bits={len(self.action_bits)}."
            )
        self.net = nn.Sequential(
            nn.Linear(self.state_dim, self.hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(self.hidden_dim, self.num_actions),
        )

    def forward(self, states: torch.Tensor) -> torch.Tensor:
        if states.dim() != 2:
            raise ValueError(f"states must have shape [num_agents, state_dim], got {tuple(states.shape)}.")
        if states.shape[-1] != self.state_dim:
            raise ValueError(f"expected state_dim={self.state_dim}, got {states.shape[-1]}.")
        return self.net(states)

    def distribution(self, states: torch.Tensor) -> Categorical:
        return Categorical(logits=self.forward(states))

    def sample(self, states: torch.Tensor, deterministic: bool = False) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        distribution = self.distribution(states)
        if deterministic:
            actions = torch.argmax(distribution.logits, dim=-1)
        else:
            actions = distribution.sample()
        log_probs = distribution.log_prob(actions)
        entropy = distribution.entropy()
        return actions, log_probs, entropy

    def evaluate_actions(self, states: torch.Tensor, actions: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        distribution = self.distribution(states)
        return distribution.log_prob(actions), distribution.entropy()


def build_actor(
    state_dim: int,
    hidden_dim: int = 128,
    bit_choices: Sequence[int] | None = None,
    weight_bit_choices: Sequence[int] | None = None,
    activation_bit_choices: Sequence[int] | None = None,
) -> SharedActor:
    return SharedActor(
        state_dim=state_dim,
        hidden_dim=hidden_dim,
        bit_choices=bit_choices,
        weight_bit_choices=weight_bit_choices,
        activation_bit_choices=activation_bit_choices,
    )
