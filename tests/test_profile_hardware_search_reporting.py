from __future__ import annotations

import json
from pathlib import Path

import torch
import numpy as np
from torch import nn

from hmaq_vlm.agents import ModalityCoordinator
from hmaq_vlm.hardware import MeasurementProtocol, ServerTimingBackend
from hmaq_vlm.profiling import SensitivityProfiler
from hmaq_vlm.quantization import MixedPrecisionPolicy, PrecisionAction, build_quant_group_registry
from hmaq_vlm.quantization.export import export_static_checkpoint, load_static_checkpoint
from hmaq_vlm.reporting import build_result_artifacts, pareto_frontier, select_policies
from hmaq_vlm.search import SearchEnvironment, run_search


class ProfileModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.vision_encoder = nn.Sequential(nn.Linear(4, 4))
        self.projector = nn.Sequential(nn.Linear(4, 4))
        self.language_model = nn.Module()
        self.language_model.transformer = nn.Sequential(nn.Linear(4, 4))
        self.language_model.lm_head = nn.Linear(4, 5)


def test_profiler_restores_model_and_records_every_metric() -> None:
    model = ProfileModel()
    registry = build_quant_group_registry(model)
    before = {name: value.clone() for name, value in model.state_dict().items()}
    model.vision_encoder.eval()
    modes = {name: module.training for name, module in model.named_modules()}
    rng = torch.get_rng_state().clone()
    numpy_rng = np.random.get_state()

    def probe(_model, group, action):
        _ = np.random.rand()
        return {
            "caption_loss_delta": action.weight_bits / 100,
            "prefix_distortion": 0.1,
            "logit_kl": 0.2,
            "activation_range": 1.0,
            "gradient_norm": 0.3,
            "parameters": group.parameters,
            "bitops": group.parameters * action.weight_bits * action.activation_bits,
            "model_size_bytes": group.parameters * action.weight_bits / 8,
        }

    rows = SensitivityProfiler(model, registry, probe).profile(actions=(PrecisionAction(4, 8), PrecisionAction(16, 16)))
    assert len(rows) == len(registry) * 2
    assert all(row["schema_version"] == "1.0" for row in rows)
    assert all(torch.equal(before[name], value) for name, value in model.state_dict().items())
    assert modes == {name: module.training for name, module in model.named_modules()}
    assert torch.equal(rng, torch.get_rng_state())
    assert np.array_equal(numpy_rng[1], np.random.get_state()[1])
    assert not any("QuantizedLinear" in type(module).__name__ for module in model.modules())


def test_server_timing_is_explicitly_diagnostic_only() -> None:
    measurement = ServerTimingBackend().measure(lambda: torch.ones(8).sin(), MeasurementProtocol(warmups=1, repeats=5))
    assert measurement.backend == "server_fake_quant"
    assert measurement.diagnostic_only is True
    assert measurement.server_fake_quant_ms["p95"] >= measurement.server_fake_quant_ms["p50"] >= 0
    payload = measurement.to_dict()
    assert "jetson_latency_ms" not in payload
    assert payload["claim_scope"] == "diagnostic_only_no_jetson_claims"


def test_search_methods_share_budget_cache_and_hierarchical_audit(tmp_path: Path) -> None:
    groups = ["vision.0", "projector.0", "language.0"]
    calls = {"count": 0, "timing": 0}

    def evaluator(policy: MixedPrecisionPolicy):
        calls["count"] += 1
        mean_bits = sum(action.weight_bits + action.activation_bits for action in policy.actions.values()) / (2 * len(groups))
        return {"cider": 1.0 - (16 - mean_bits) / 100, "bitops_ratio": mean_bits / 16, "model_size_ratio": mean_bits / 16, "prefix_distortion": 0.01, "logit_kl": 0.02}

    def timing(_policy):
        calls["timing"] += 1
        return {"server_fake_quant_ms": {"p50": 1.0, "p95": 1.1}, "diagnostic_only": True}

    environment = SearchEnvironment(groups, evaluator, timing_evaluator=timing, cache_path=tmp_path / "cache.json", model_hash="m", dataset_hash="d", runtime_hash="r", protocol_hash="p")
    same = MixedPrecisionPolicy({name: PrecisionAction(8, 8) for name in groups})
    assert environment.evaluate(same).cached is False
    assert environment.evaluate(same).cached is True
    assert calls["count"] == 1
    for method in ("random", "greedy", "ppo", "mappo", "hierarchical_mappo"):
        run = run_search(method, environment, budget=4, timing_budget=2, seed=11)
        assert len(run.candidates) == 4
        assert run.candidate_budget == 4 and run.timing_budget == 2
        assert all("action_mask" in entry for entry in run.audit)
        assert all("server_timing" in entry for entry in run.audit[:2])
        if method == "hierarchical_mappo":
            assert all("coordinator_budgets" in entry for entry in run.audit)
            for entry in run.audit:
                assert all(entry["aggregate_costs"][name] <= entry["coordinator_budgets"][name] + 1e-7 for name in ("vision", "projector", "language"))
        if method in ("ppo", "mappo", "hierarchical_mappo"):
            assert all(entry["learner_updates"] >= 1 for entry in run.audit)
            assert any("policy_loss" in entry for entry in run.audit)
    assert calls["timing"] == 10


def test_coordinator_respects_minimums_and_is_differentiable() -> None:
    coordinator = ModalityCoordinator(input_dim=5)
    budgets = coordinator(torch.randn(2, 5))
    assert torch.all(budgets[:, 0] >= 0.10)
    assert torch.all(budgets[:, 1] >= 0.05)
    assert torch.all(budgets[:, 2] >= 0.30)
    assert torch.allclose(budgets.sum(dim=-1), torch.ones(2))
    budgets.sum().backward()
    assert all(parameter.grad is not None for parameter in coordinator.parameters())


def test_static_export_pareto_selection_and_reports(tmp_path: Path) -> None:
    model = ProfileModel()
    registry = build_quant_group_registry(model)
    policy = MixedPrecisionPolicy({group.name: PrecisionAction(8, 8) for group in registry})
    checkpoint = tmp_path / "model.pt"
    import pytest
    from hmaq_vlm.quantization import inject_quantizers

    with pytest.raises(ValueError, match="quantized"):
        export_static_checkpoint(checkpoint, model, policy, {"seed": 11}, registry)
    inject_quantizers(model, registry, policy)
    export_static_checkpoint(checkpoint, model, policy, {"seed": 11}, registry)
    payload = load_static_checkpoint(checkpoint)
    assert set(payload) == {"schema_version", "model_state", "policy", "metadata"}
    assert not any(key.startswith(("actor", "critic", "coordinator")) for key in payload["model_state"])
    rows = [
        {"method": "fp16", "cider": 1.00, "bitops_ratio": 1.0, "model_size_ratio": 1.0},
        {"method": "fast", "cider": 0.95, "bitops_ratio": 0.3, "model_size_ratio": 0.4},
        {"method": "small", "cider": 0.92, "bitops_ratio": 0.5, "model_size_ratio": 0.2},
        {"method": "dominated", "cider": 0.80, "bitops_ratio": 0.8, "model_size_ratio": 0.8},
    ]
    assert "dominated" not in {row["method"] for row in pareto_frontier(rows)}
    selected = select_policies(rows, cost_target=0.5, quality_drop_target=0.1)
    assert set(selected) == {"lowest_cost", "highest_quality_under_cost", "smallest_under_quality_drop"}
    paths = build_result_artifacts(rows, tmp_path / "reports")
    assert {path.suffix for path in paths} >= {".json", ".csv", ".tex"}
    assert json.loads((tmp_path / "reports" / "metrics.json").read_text())[0]["method"] == "fp16"
