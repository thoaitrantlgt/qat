from __future__ import annotations

import json
import random
from dataclasses import asdict, dataclass
from pathlib import Path

from hmaq_vlm.reproducibility import atomic_write_bytes, file_sha256, stable_hash


SPLITS = ("train", "policy_search", "validation", "test")


@dataclass(frozen=True)
class CaptionImage:
    image_id: str
    image_path: str
    captions: tuple[str, ...]
    image_sha256: str | None = None


@dataclass(frozen=True)
class ManifestPaths:
    root: Path
    index: Path
    splits: dict[str, Path]


@dataclass(frozen=True)
class ManifestSet:
    splits: dict[str, tuple[CaptionImage, ...]]
    dataset_hash: str


def _immutable_write(path: Path, content: bytes) -> None:
    if path.exists():
        if path.read_bytes() != content:
            raise FileExistsError(f"immutable manifest already exists with different content: {path}")
        return
    atomic_write_bytes(path, content)


def build_manifests(images: list[CaptionImage], root: str | Path, seed: int, ratios: tuple[float, float, float] = (0.70, 0.10, 0.10)) -> ManifestPaths:
    if len({item.image_id for item in images}) != len(images):
        raise ValueError("duplicate image_id")
    if sum(ratios) >= 1.0 or any(value < 0 for value in ratios):
        raise ValueError("train/policy/validation ratios must be non-negative and sum below one")
    enriched = [CaptionImage(item.image_id, item.image_path, tuple(item.captions), file_sha256(item.image_path)) for item in images]
    shuffled = sorted(enriched, key=lambda item: item.image_id)
    random.Random(seed).shuffle(shuffled)
    n = len(shuffled)
    boundaries = [int(n * ratios[0]), int(n * sum(ratios[:2])), int(n * sum(ratios))]
    chunks = (shuffled[:boundaries[0]], shuffled[boundaries[0]:boundaries[1]], shuffled[boundaries[1]:boundaries[2]], shuffled[boundaries[2]:])
    output = Path(root)
    output.mkdir(parents=True, exist_ok=True)
    split_paths: dict[str, Path] = {}
    split_hashes: dict[str, str] = {}
    for name, records in zip(SPLITS, chunks, strict=True):
        path = output / f"{name}.jsonl"
        lines = [json.dumps(asdict(record), sort_keys=True, ensure_ascii=False) for record in records]
        content = (("\n".join(lines) + "\n") if lines else "").encode("utf-8")
        _immutable_write(path, content)
        split_paths[name] = path
        split_hashes[name] = file_sha256(path)
    index = output / "manifest.json"
    metadata = {"schema_version": "1.0", "seed": seed, "splits": split_hashes, "dataset_hash": stable_hash(split_hashes)}
    _immutable_write(index, (json.dumps(metadata, sort_keys=True, indent=2) + "\n").encode())
    return ManifestPaths(output, index, split_paths)


def load_manifest_set(paths: ManifestPaths, verify_images: bool = True) -> ManifestSet:
    metadata = json.loads(paths.index.read_text(encoding="utf-8"))
    loaded: dict[str, tuple[CaptionImage, ...]] = {}
    seen: set[str] = set()
    for name in SPLITS:
        path = paths.splits[name]
        if file_sha256(path) != metadata["splits"][name]:
            raise ValueError(f"manifest checksum drift: {name}")
        records = tuple(CaptionImage(**json.loads(line)) for line in path.read_text(encoding="utf-8").splitlines())
        ids = {record.image_id for record in records}
        if seen & ids:
            raise ValueError("split overlap detected")
        seen |= ids
        if verify_images:
            for record in records:
                if not record.image_sha256 or file_sha256(record.image_path) != record.image_sha256:
                    raise ValueError(f"image checksum drift: {record.image_id}")
        loaded[name] = records
    return ManifestSet(loaded, metadata["dataset_hash"])


def build_karpathy_manifests(records, root: str | Path, seed: int, policy_fraction: float = 0.10) -> ManifestPaths:
    if not 0 < policy_fraction < 1:
        raise ValueError("policy fraction must be between zero and one")
    pools = {name: [] for name in SPLITS}
    training = []
    for record in records:
        item = CaptionImage(record.image_id, record.image_path, tuple(record.captions), file_sha256(record.image_path))
        if record.source_split in ("train", "restval"):
            training.append(item)
        elif record.source_split == "val":
            pools["validation"].append(item)
        elif record.source_split == "test":
            pools["test"].append(item)
        else:
            raise ValueError(f"unknown Karpathy split: {record.source_split}")
    training.sort(key=lambda item: item.image_id)
    random.Random(seed).shuffle(training)
    policy_count = max(1, round(len(training) * policy_fraction)) if len(training) > 1 else 0
    pools["policy_search"] = training[:policy_count]
    pools["train"] = training[policy_count:]
    all_ids = [item.image_id for values in pools.values() for item in values]
    if len(all_ids) != len(set(all_ids)):
        raise ValueError("image overlap across Karpathy source splits")
    output = Path(root)
    output.mkdir(parents=True, exist_ok=True)
    split_paths, split_hashes = {}, {}
    for name in SPLITS:
        path = output / f"{name}.jsonl"
        lines = [json.dumps(asdict(item), sort_keys=True, ensure_ascii=False) for item in pools[name]]
        content = (("\n".join(lines) + "\n") if lines else "").encode("utf-8")
        _immutable_write(path, content)
        split_paths[name] = path
        split_hashes[name] = file_sha256(path)
    index = output / "manifest.json"
    metadata = {"schema_version": "1.0", "seed": seed, "source_split": "coco_karpathy", "splits": split_hashes, "dataset_hash": stable_hash(split_hashes)}
    _immutable_write(index, (json.dumps(metadata, sort_keys=True, indent=2) + "\n").encode())
    return ManifestPaths(output, index, split_paths)
