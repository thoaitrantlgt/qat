from __future__ import annotations

import random
from dataclasses import dataclass

import torch
from torch import nn
from torch.nn import functional as F

from hmaq_vlm.agents import CentralCritic, ModalityCoordinator, PPOHyperparameters, SharedActor
from hmaq_vlm.quantization import ACTION_SPACE, MixedPrecisionPolicy, PrecisionAction
from .environment import CandidateResult, SearchEnvironment


METHODS = ("random", "greedy", "ppo", "mappo", "hierarchical_mappo")


@dataclass(frozen=True)
class SearchRun:
    method: str
    candidates: tuple[CandidateResult, ...]
    audit: tuple[dict[str, object], ...]
    candidate_budget: int
    timing_budget: int
    schema_version: str = "1.0"


def _modality(name: str) -> int:
    lowered = name.lower()
    if "vision" in lowered:
        return 0
    if "projector" in lowered:
        return 1
    return 2


def _module_type(name: str) -> str:
    lowered = name.lower()
    if any(token in lowered for token in ("attn", "attention", "qkv", "c_attn")):
        return "attention"
    if any(token in lowered for token in ("mlp", "fc", "c_fc", "c_proj")):
        return "mlp"
    return "linear"


class _RLLearner:
    def __init__(self, method: str, groups: tuple[str, ...], seed: int, hyper: PPOHyperparameters) -> None:
        torch.manual_seed(seed)
        self.method = method
        self.groups = groups
        self.hyper = hyper
        self.state_dim = len(groups) + 3
        keys = ["shared"] if method == "ppo" else sorted({_module_type(group) for group in groups})
        self.actors = nn.ModuleDict({key: SharedActor(self.state_dim, len(ACTION_SPACE)) for key in keys})
        self.coordinator = ModalityCoordinator(5) if method == "hierarchical_mappo" else None
        critic_dim = len(groups) + (3 if self.coordinator is not None else 0)
        self.critic = CentralCritic(critic_dim)
        actor_parameters = list(self.actors.parameters()) + (list(self.coordinator.parameters()) if self.coordinator is not None else [])
        self.actor_optimizer = torch.optim.Adam(actor_parameters, lr=hyper.actor_lr)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=hyper.critic_lr)
        self.generator = torch.Generator().manual_seed(seed)
        self.last_reward = 0.0

    def _actor_for(self, group: str) -> SharedActor:
        return self.actors["shared" if self.method == "ppo" else _module_type(group)]

    def _states(self, progress: float, selected: list[int] | None = None) -> tuple[list[torch.Tensor], torch.Tensor | None, list[torch.Tensor]]:
        if self.coordinator is not None:
            context = torch.tensor([[progress, self.last_reward, 1.0, 0.0, 0.0]])
            budgets = self.coordinator(context).squeeze(0)
        else:
            budgets = None
        states, masks = [], []
        action_costs = torch.tensor([(action.weight_bits * action.activation_bits) / (256 * len(self.groups)) for action in ACTION_SPACE])
        remaining = budgets.detach().clone() if budgets is not None else None
        remaining_groups = [sum(_modality(group) == modality for group in self.groups) for modality in range(3)]
        minimum_cost = float(action_costs.min())
        for index, group in enumerate(self.groups):
            one_hot = F.one_hot(torch.tensor(index), len(self.groups)).float()
            budget_features = budgets if budgets is not None else F.one_hot(torch.tensor(_modality(group)), 3).float()
            states.append(torch.cat((one_hot, budget_features)))
            if budgets is None:
                mask = torch.ones(len(ACTION_SPACE), dtype=torch.bool)
            else:
                modality = _modality(group)
                available = remaining[modality] - (remaining_groups[modality] - 1) * minimum_cost
                mask = action_costs <= available + 1e-9
                if selected is not None:
                    mask[selected[index]] = True
                    remaining[modality] = (remaining[modality] - action_costs[selected[index]]).clamp_min(0)
                remaining_groups[modality] -= 1
            masks.append(mask)
        return states, budgets, masks

    def sample(self, progress: float) -> tuple[MixedPrecisionPolicy, torch.Tensor, list[int], list[list[int]], list[float] | None]:
        states, budgets, masks = self._states(progress)
        indices, old_log_probs = [], []
        remaining = budgets.detach().clone() if budgets is not None else None
        action_costs = torch.tensor([(action.weight_bits * action.activation_bits) / (256 * len(self.groups)) for action in ACTION_SPACE])
        remaining_groups = [sum(_modality(group) == modality for group in self.groups) for modality in range(3)]
        minimum_cost = float(action_costs.min())
        actual_masks = []
        for group, state, initial_mask in zip(self.groups, states, masks, strict=True):
            if remaining is None:
                mask = initial_mask
            else:
                modality = _modality(group)
                available = remaining[modality] - (remaining_groups[modality] - 1) * minimum_cost
                mask = action_costs <= available + 1e-9
            logits = self._actor_for(group)(state, mask)
            probabilities = logits.softmax(dim=-1)
            index = int(torch.multinomial(probabilities.detach(), 1, generator=self.generator))
            indices.append(index)
            old_log_probs.append(logits.log_softmax(dim=-1)[index].detach())
            actual_masks.append(mask)
            if remaining is not None:
                remaining[_modality(group)] = (remaining[_modality(group)] - action_costs[index]).clamp_min(0)
                remaining_groups[_modality(group)] -= 1
        policy = MixedPrecisionPolicy({group: ACTION_SPACE[index] for group, index in zip(self.groups, indices, strict=True)})
        return policy, torch.stack(old_log_probs).sum(), indices, [mask.int().tolist() for mask in actual_masks], budgets.detach().tolist() if budgets is not None else None

    def update(self, progress: float, indices: list[int], old_log_prob: torch.Tensor, reward: float) -> dict[str, float]:
        target = torch.tensor(max(-10.0, min(10.0, reward)))
        last_actor = last_critic = last_entropy = 0.0
        for _ in range(self.hyper.optimization_epochs):
            states, budgets, masks = self._states(progress, indices)
            log_probs, entropies = [], []
            for group, state, mask, index in zip(self.groups, states, masks, indices, strict=True):
                logits = self._actor_for(group)(state, mask)
                distribution = torch.distributions.Categorical(logits=logits)
                log_probs.append(distribution.log_prob(torch.tensor(index)))
                entropies.append(distribution.entropy())
            new_log_prob = torch.stack(log_probs).sum()
            entropy = torch.stack(entropies).mean()
            action_costs = torch.tensor([(ACTION_SPACE[index].weight_bits + ACTION_SPACE[index].activation_bits) / 32 for index in indices])
            critic_state = torch.cat((action_costs, budgets.detach())) if budgets is not None else action_costs
            value = self.critic(critic_state)
            advantage = target - value.detach()
            ratio = (new_log_prob - old_log_prob).exp()
            clipped = ratio.clamp(1 - self.hyper.clip, 1 + self.hyper.clip)
            actor_loss = -torch.minimum(ratio * advantage, clipped * advantage) - self.hyper.entropy * entropy
            critic_loss = F.mse_loss(value, target)
            self.actor_optimizer.zero_grad(set_to_none=True)
            actor_loss.backward()
            self.actor_optimizer.step()
            self.critic_optimizer.zero_grad(set_to_none=True)
            critic_loss.backward()
            self.critic_optimizer.step()
            last_actor, last_critic, last_entropy = float(actor_loss.detach()), float(critic_loss.detach()), float(entropy.detach())
        self.last_reward = float(target)
        return {"policy_loss": last_actor, "critic_loss": last_critic, "entropy": last_entropy}


def _greedy_policy(groups: tuple[str, ...], index: int, best: MixedPrecisionPolicy | None) -> MixedPrecisionPolicy:
    if best is None:
        return MixedPrecisionPolicy({group: PrecisionAction(16, 16) for group in groups})
    actions = dict(best.actions)
    group = groups[(index - 1) % len(groups)]
    current = actions[group]
    cheaper = [action for action in ACTION_SPACE if action.weight_bits * action.activation_bits < current.weight_bits * current.activation_bits]
    actions[group] = max(cheaper, key=lambda action: action.weight_bits * action.activation_bits) if cheaper else current
    return MixedPrecisionPolicy(actions)


def run_search(method: str, environment: SearchEnvironment, *, budget: int = 100, timing_budget: int = 50, seed: int = 11) -> SearchRun:
    if method not in METHODS:
        raise ValueError(f"unknown search method: {method}")
    if budget < 1 or timing_budget < 0 or timing_budget > budget:
        raise ValueError("invalid search budgets")
    rng = random.Random(f"{method}:{seed}")
    learner = _RLLearner(method, environment.groups, seed, PPOHyperparameters()) if method in ("ppo", "mappo", "hierarchical_mappo") else None
    candidates: list[CandidateResult] = []
    audit: list[dict[str, object]] = []
    best_policy = None
    best_reward = float("-inf")
    for index in range(budget):
        progress = index / max(1, budget - 1)
        budgets = None
        if method == "random":
            policy = MixedPrecisionPolicy({name: rng.choice(ACTION_SPACE) for name in environment.groups})
            masks = [[1] * len(ACTION_SPACE) for _ in environment.groups]
        elif method == "greedy":
            policy = _greedy_policy(environment.groups, index, best_policy)
            masks = [[1] * len(ACTION_SPACE) for _ in environment.groups]
        else:
            assert learner is not None
            policy, old_log_prob, indices, masks, budgets = learner.sample(progress)
        result = environment.evaluate(policy)
        if method == "greedy" and result.valid and result.reward > best_reward:
            best_policy, best_reward = policy, result.reward
        losses = learner.update(progress, indices, old_log_prob, result.reward) if learner is not None else {}
        entry: dict[str, object] = {
            "index": index,
            "policy_hash": result.policy_hash,
            "raw_reward": result.reward,
            "reward_components": result.metrics,
            "action_mask": {name: mask for name, mask in zip(environment.groups, masks, strict=True)},
            "failure": result.failure,
            "cached": result.cached,
            "search_cost": {"candidate_evaluations": index + 1, "timing_diagnostics": min(index + 1, timing_budget)},
            **losses,
        }
        if index < timing_budget:
            timing = environment.measure_diagnostic(policy)
            if timing is not None:
                entry["server_timing"] = timing
        if learner is not None:
            entry["learner_updates"] = (index + 1) * learner.hyper.optimization_epochs
        if budgets is not None:
            entry["coordinator_budgets"] = dict(zip(("vision", "projector", "language"), budgets, strict=True))
            aggregate = {"vision": 0.0, "projector": 0.0, "language": 0.0}
            names = ("vision", "projector", "language")
            for group, action in policy.actions.items():
                aggregate[names[_modality(group)]] += action.weight_bits * action.activation_bits / (256 * len(environment.groups))
            entry["aggregate_costs"] = aggregate
            entry["budget_projection"] = {"minimums": {"vision": 0.10, "projector": 0.05, "language": 0.30}, "projected": all(aggregate[name] <= entry["coordinator_budgets"][name] + 1e-7 for name in names)}
        candidates.append(result)
        audit.append(entry)
    return SearchRun(method, tuple(candidates), tuple(audit), budget, timing_budget)
