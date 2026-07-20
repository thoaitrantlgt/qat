from __future__ import annotations

from types import SimpleNamespace
from pathlib import Path
import copy

import torch
from torch import nn

from hmaq_vlm.hardware import MeasurementProtocol, ServerTimingBackend
from hmaq_vlm.models import HMAQVLM
from hmaq_vlm.losses import consistency_losses
from hmaq_vlm.quantization import MixedPrecisionPolicy, PrecisionAction, build_quant_group_registry, inject_quantizers, temporary_policy
from hmaq_vlm.quantization.export import export_static_checkpoint
from hmaq_vlm.reporting import build_result_artifacts
from hmaq_vlm.reproducibility import atomic_write_json, seed_everything
from hmaq_vlm.search import SearchEnvironment, run_search
from hmaq_vlm.training import build_optimizer, calibrate_quantizers, caption_training_step, set_trainable_stage


class _SmokeVision(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.proj = nn.Linear(3, 6)

    def forward_features(self, pixels: torch.Tensor) -> torch.Tensor:
        token = self.proj(pixels.mean(dim=(-1, -2)))
        return token[:, None].expand(-1, 5, -1)


class _SmokeLanguage(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.embed = nn.Embedding(17, 8)
        self.transformer = nn.Sequential(nn.Linear(8, 8), nn.Tanh())
        self.lm_head = nn.Linear(8, 17, bias=False)

    def get_input_embeddings(self) -> nn.Module:
        return self.embed

    def forward(self, inputs_embeds, attention_mask, position_ids, past_key_values=None, use_cache=False):
        logits = self.lm_head(self.transformer(inputs_embeds))
        return SimpleNamespace(logits=logits, past_key_values=(torch.tensor(1),) if use_cache else None)


def run_acceptance_smoke(output_dir: str | Path, *, seed: int = 11, search_budget: int = 4, timing_budget: int = 2) -> dict[str, object]:
    """Execute the complete CPU acceptance protocol without network or pretrained assets."""
    seed_everything(seed)
    output = Path(output_dir)
    model = HMAQVLM(_SmokeVision(), _SmokeLanguage(), vision_dim=6, language_dim=8)
    batch = {
        "pixel_values": torch.randn(2, 3, 4, 4),
        "input_ids": torch.tensor([[1, 4, 2], [1, 5, 2]]),
        "attention_mask": torch.ones(2, 3, dtype=torch.long),
        "labels": torch.tensor([[1, 4, 2], [1, 5, 2]]),
    }
    set_trainable_stage(model, 1)
    optimizer = build_optimizer(model, projector_lr=1e-3, vision_lr=1e-4, language_lr=1e-4)
    fp16_step = caption_training_step(model, batch, optimizer, gradient_clip=1.0)
    generated = model.generate(batch["pixel_values"], batch["input_ids"][:, :1], eos_token_id=2, max_new_tokens=3)

    search_model = copy.deepcopy(model)
    search_registry = build_quant_group_registry(search_model)
    search_model.eval()
    with torch.no_grad():
        search_teacher = search_model(**batch)
    registry = build_quant_group_registry(model)
    frozen_policy = MixedPrecisionPolicy({group.name: PrecisionAction(4, 8) for group in registry})
    inject_quantizers(model, registry, frozen_policy)
    calibrate_quantizers(model, [batch], lambda module, item: module(**item))
    set_trainable_stage(model, 4)
    optimizer = torch.optim.AdamW((parameter for parameter in model.parameters() if parameter.requires_grad), lr=1e-4)
    qat_step = caption_training_step(model, batch, optimizer, gradient_clip=1.0)

    group_names = [group.name for group in search_registry]

    def evaluate(policy: MixedPrecisionPolicy) -> dict[str, float]:
        with temporary_policy(search_model, search_registry, policy):
            calibrate_quantizers(search_model, [batch], lambda module, item: module(**item))
            with torch.no_grad():
                student = search_model(**batch)
                prefix, kl = consistency_losses(student.visual_prefix, search_teacher.visual_prefix, student.logits[:, :-1], search_teacher.logits[:, :-1], 2.0, student.labels[:, 1:].ne(-100))
            total = sum(group.parameters for group in search_registry)
            bitops = sum(group.parameters * policy.actions[group.name].weight_bits * policy.actions[group.name].activation_bits for group in search_registry) / (total * 256)
            size = sum(group.parameters * policy.actions[group.name].weight_bits for group in search_registry) / (total * 16)
            return {"cider": float(torch.exp(-student.loss.detach())), "bitops_ratio": bitops, "model_size_ratio": size, "prefix_distortion": float(prefix), "logit_kl": float(kl)}

    backend = ServerTimingBackend()

    def timing(policy: MixedPrecisionPolicy) -> dict[str, object]:
        with temporary_policy(search_model, search_registry, policy):
            calibrate_quantizers(search_model, [batch], lambda module, item: module(**item))
            with torch.no_grad():
                measurement = backend.measure(lambda: search_model(**batch).loss, MeasurementProtocol(warmups=0, repeats=1, synchronize_cuda=False))
            return measurement.to_dict()

    environment = SearchEnvironment(group_names, evaluate, timing_evaluator=timing, cache_path=output / "cache" / "candidates.json", model_hash="synthetic-vlm-v1", dataset_hash="synthetic-caption-v1", runtime_hash=f"torch-{torch.__version__}", protocol_hash=f"budget-{search_budget}-{timing_budget}")
    searches = {}
    report_rows = []
    for method in ("random", "greedy", "ppo", "mappo", "hierarchical_mappo"):
        run = run_search(method, environment, budget=search_budget, timing_budget=timing_budget, seed=seed)
        valid = [candidate for candidate in run.candidates if candidate.valid]
        best = max(valid, key=lambda candidate: candidate.reward)
        searches[method] = {"candidate_count": len(run.candidates), "best_reward": best.reward, "cache_hits": sum(candidate.cached for candidate in run.candidates)}
        report_rows.append({"method": method, **best.metrics})
        atomic_write_json(output / "audits" / f"{method}.json", list(run.audit))
    export_static_checkpoint(output / "checkpoints" / "static_qat.pt", model, frozen_policy, {"seed": seed, "synthetic_acceptance": True}, registry)
    build_result_artifacts(report_rows, output / "reports")
    summary: dict[str, object] = {"schema_version": "1.0", "seed": seed, "fp16_step": fp16_step, "qat_step": qat_step, "generated_tokens": generated.shape[1], "searches": searches}
    atomic_write_json(output / "acceptance_summary.json", summary)
    return summary
