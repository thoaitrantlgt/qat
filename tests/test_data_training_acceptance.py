from __future__ import annotations

import json
from pathlib import Path

import torch
import pytest

from hmaq_vlm.data import build_karpathy_manifests, load_manifest_set
from hmaq_vlm.data.sources import SourceCaptionImage, load_flickr30k_records
from hmaq_vlm.models import HMAQVLM
from hmaq_vlm.quantization import MixedPrecisionPolicy, PrecisionAction, build_quant_group_registry, inject_quantizers
from hmaq_vlm.training import build_optimizer, calibrate_quantizers, caption_training_step, set_trainable_stage
from hmaq_vlm.smoke import run_acceptance_smoke

from test_model_quant_training import TinyLM, TinyVision


def test_flickr_loader_uses_supplied_paths_and_image_level_records(tmp_path: Path) -> None:
    images = tmp_path / "images"
    images.mkdir()
    (images / "a.jpg").write_bytes(b"a")
    annotation = tmp_path / "dataset.json"
    annotation.write_text(json.dumps({"images": [{"filename": "a.jpg", "split": "train", "sentences": [{"raw": "A caption."}, {"tokens": ["Second", "caption"]}]}]}), encoding="utf-8")
    records = load_flickr30k_records(images, annotation)
    assert len(records) == 1
    assert records[0].captions == ("A caption.", "Second caption")
    assert records[0].source_split == "train"


def test_karpathy_manifest_preserves_validation_and_test_isolation(tmp_path: Path) -> None:
    records = []
    for index, split in enumerate(["train", "train", "restval", "val", "test"]):
        image = tmp_path / f"source-{index}.jpg"
        image.write_bytes(str(index).encode())
        records.append(SourceCaptionImage(str(index), str(image), (f"caption {index}",), split))
    paths = build_karpathy_manifests(records, tmp_path / "karpathy", seed=11, policy_fraction=0.34)
    loaded = load_manifest_set(paths)
    assert {item.image_id for item in loaded.splits["validation"]} == {"3"}
    assert {item.image_id for item in loaded.splits["test"]} == {"4"}
    assert not ({item.image_id for item in loaded.splits["policy_search"]} & {"3", "4"})


def test_one_fp16_step_and_one_calibrated_qat_step_are_finite() -> None:
    torch.manual_seed(1)
    model = HMAQVLM(TinyVision(), TinyLM(), vision_dim=6, language_dim=8)
    batch = {
        "pixel_values": torch.randn(2, 3, 4, 4),
        "input_ids": torch.tensor([[1, 4, 2], [1, 5, 2]]),
        "attention_mask": torch.ones(2, 3, dtype=torch.long),
        "labels": torch.tensor([[1, 4, 2], [1, 5, 2]]),
    }
    set_trainable_stage(model, 1)
    optimizer = build_optimizer(model, projector_lr=1e-3, vision_lr=1e-4, language_lr=1e-4)
    before = model.projector[0].weight.detach().clone()
    result = caption_training_step(model, batch, optimizer, gradient_clip=1.0)
    assert torch.isfinite(torch.tensor(result["loss"]))
    assert not torch.equal(before, model.projector[0].weight)
    registry = build_quant_group_registry(model)
    policy = MixedPrecisionPolicy({group.name: PrecisionAction(4, 8) for group in registry})
    inject_quantizers(model, registry, policy)
    calibrate_quantizers(model, [batch], lambda module, item: module(**item))
    assert all(bool(module.weight_quantizer.calibrated) and bool(module.activation_quantizer.calibrated) for module in model.modules() if hasattr(module, "weight_quantizer"))
    optimizer = torch.optim.AdamW((parameter for parameter in model.parameters() if parameter.requires_grad), lr=1e-4)
    qat_result = caption_training_step(model, batch, optimizer, gradient_clip=1.0)
    assert torch.isfinite(torch.tensor(qat_result["loss"]))


def test_activation_calibration_accumulates_all_batches_order_independently() -> None:
    from hmaq_vlm.quantization import LSQFakeQuantizer

    first, second = torch.ones(2, 3), torch.full((2, 3), 9.0)
    left = LSQFakeQuantizer(4)
    left.begin_calibration()
    left(first)
    left(second)
    left.end_calibration()
    right = LSQFakeQuantizer(4)
    right.begin_calibration()
    right(second)
    right(first)
    right.end_calibration()
    assert torch.allclose(left.scale, right.scale)
    empty = LSQFakeQuantizer(4)
    empty.begin_calibration()
    with pytest.raises(RuntimeError, match="no observations"):
        empty.end_calibration()


def test_executable_acceptance_pipeline_covers_all_methods_and_exports(tmp_path: Path) -> None:
    summary = run_acceptance_smoke(tmp_path, seed=11, search_budget=4, timing_budget=2)
    assert summary["fp16_step"]["loss"] > 0
    assert summary["qat_step"]["loss"] > 0
    assert summary["generated_tokens"] >= 2
    assert set(summary["searches"]) == {"random", "greedy", "ppo", "mappo", "hierarchical_mappo"}
    assert all(item["candidate_count"] == 4 for item in summary["searches"].values())
    assert (tmp_path / "checkpoints" / "static_qat.pt").is_file()
    assert (tmp_path / "reports" / "metrics.json").is_file()
