from __future__ import annotations

import json
from pathlib import Path

import pytest

from hmaq_vlm.config import ExperimentConfig, load_config
from hmaq_vlm.data import CaptionImage, ManifestSet, build_manifests, load_manifest_set
from hmaq_vlm.reproducibility import atomic_write_json, file_sha256, stable_hash
from hmaq_vlm.cli import build_parser


def test_typed_config_is_strict_and_resolved(tmp_path: Path) -> None:
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        "seed: 22\nmodel:\n  max_new_tokens: 17\ntrain:\n  micro_batch_size: 2\n",
        encoding="utf-8",
    )
    cfg = load_config(config_path)
    assert isinstance(cfg, ExperimentConfig)
    assert cfg.seed == 22
    assert cfg.model.max_new_tokens == 17
    assert cfg.train.effective_batch_size == 2 * cfg.train.gradient_accumulation
    bad = tmp_path / "bad.yaml"
    bad.write_text("unknown: true\n", encoding="utf-8")
    with pytest.raises(ValueError, match="unknown config key"):
        load_config(bad)
    wrong_type = tmp_path / "wrong_type.yaml"
    wrong_type.write_text("seed: eleven\ntrain:\n  epochs: 0\n", encoding="utf-8")
    with pytest.raises(ValueError, match="seed"):
        load_config(wrong_type)


def test_atomic_json_and_stable_hash_are_deterministic(tmp_path: Path) -> None:
    path = tmp_path / "nested" / "artifact.json"
    atomic_write_json(path, {"b": 2, "a": 1})
    assert json.loads(path.read_text(encoding="utf-8")) == {"a": 1, "b": 2}
    assert not list(path.parent.glob("*.tmp"))
    assert stable_hash({"b": 2, "a": 1}) == stable_hash({"a": 1, "b": 2})
    assert len(file_sha256(path)) == 64


def test_image_level_manifests_are_disjoint_and_detect_drift(tmp_path: Path) -> None:
    images = []
    for index in range(20):
        image = tmp_path / f"{index}.jpg"
        image.write_bytes(f"image-{index}".encode())
        images.append(CaptionImage(str(index), str(image), (f"caption {index}",)))
    paths = build_manifests(images, tmp_path / "manifests", seed=11)
    loaded = load_manifest_set(paths)
    assert isinstance(loaded, ManifestSet)
    assert sum(len(split) for split in loaded.splits.values()) == 20
    ids = [set(item.image_id for item in split) for split in loaded.splits.values()]
    assert not any(left & right for i, left in enumerate(ids) for right in ids[i + 1 :])
    Path(images[0].image_path).write_bytes(b"changed")
    with pytest.raises(ValueError, match="checksum drift"):
        load_manifest_set(paths)


def test_prepare_flickr_cli_supports_automatic_and_local_sources() -> None:
    parser = build_parser()
    automatic = parser.parse_args(
        [
            "prepare-flickr",
            "--cache",
            "data/flickr30k",
            "--output",
            "artifacts/manifests/flickr30k",
        ]
    )
    assert automatic.cache == Path("data/flickr30k")
    assert automatic.images is None
    local = parser.parse_args(
        [
            "prepare-flickr",
            "--images",
            "images",
            "--annotations",
            "dataset_flickr30k.json",
            "--output",
            "artifacts/manifests/flickr30k",
        ]
    )
    assert local.images == Path("images")
    assert local.annotations == Path("dataset_flickr30k.json")
