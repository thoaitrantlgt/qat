from __future__ import annotations

import json
import shutil
import tarfile
from urllib.request import urlopen
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from hmaq_vlm.reproducibility import atomic_write_bytes


COCO_DATASET = "yerevann/coco-karpathy"
COCO_REVISION = "448fdb1bc7b2d09e46881c4541a14d796a3d41e8"
FLICKR30K_DATASET = "cjc/flickr30k"
FLICKR30K_REVISION = "5cd04e71affaa3b289d9f545b386d06338aa337c"
FLICKR30K_ANNOTATIONS = "dataset_flickr30k.json"
FLICKR30K_IMAGES_ARCHIVE = "flickr30k-images.tar"


@dataclass(frozen=True)
class SourceCaptionImage:
    image_id: str
    image_path: str
    captions: tuple[str, ...]
    source_split: str


def _extract_flickr30k_archive(archive_path: Path, output_root: Path) -> Path:
    revision_root = output_root / FLICKR30K_REVISION
    if revision_root.is_dir():
        return revision_root
    staging_root = output_root / f".{FLICKR30K_REVISION}.tmp"
    if staging_root.exists():
        shutil.rmtree(staging_root)
    staging_root.mkdir(parents=True)
    resolved_staging = staging_root.resolve()
    try:
        with tarfile.open(archive_path, "r") as archive:
            for member in archive.getmembers():
                target = (staging_root / member.name).resolve()
                if target != resolved_staging and resolved_staging not in target.parents:
                    raise ValueError(f"unsafe Flickr30k archive member: {member.name}")
                if member.issym() or member.islnk() or not (member.isfile() or member.isdir()):
                    raise ValueError(f"unsafe Flickr30k archive member: {member.name}")
                if member.isdir():
                    target.mkdir(parents=True, exist_ok=True)
                    continue
                target.parent.mkdir(parents=True, exist_ok=True)
                source = archive.extractfile(member)
                if source is None:
                    raise ValueError(f"unable to read Flickr30k archive member: {member.name}")
                with source, target.open("wb") as destination:
                    shutil.copyfileobj(source, destination)
        output_root.mkdir(parents=True, exist_ok=True)
        staging_root.replace(revision_root)
    except BaseException:
        if staging_root.exists():
            shutil.rmtree(staging_root)
        raise
    return revision_root


def _locate_flickr30k_images(extracted_root: Path, annotations_path: Path) -> Path:
    payload = json.loads(annotations_path.read_text(encoding="utf-8"))
    entries = payload.get("images") if isinstance(payload, dict) else payload
    if not isinstance(entries, list) or not entries:
        raise ValueError("Flickr30k annotations must contain a non-empty images list")
    filename = entries[0].get("filename") or entries[0].get("file_name")
    if not filename:
        raise ValueError("Flickr30k annotation is missing filename")
    candidates = (extracted_root, extracted_root / "flickr30k-images", extracted_root / "images")
    for candidate in candidates:
        if (candidate / filename).is_file():
            return candidate.resolve()
    matches = list(extracted_root.rglob(filename))
    if len(matches) == 1:
        return matches[0].parent.resolve()
    raise FileNotFoundError(f"Flickr30k image not found after extraction: {filename}")


def download_flickr30k_source(
    cache_dir: str | Path,
    *,
    downloader: Callable[..., str] | None = None,
) -> tuple[Path, Path]:
    """Download the pinned Flickr30k images and canonical Karpathy annotations."""
    if downloader is None:
        try:
            from huggingface_hub import hf_hub_download
        except ImportError as error:
            raise RuntimeError("automatic Flickr30k download requires huggingface-hub") from error
        downloader = hf_hub_download
    cache_root = Path(cache_dir).resolve()
    hub_cache = cache_root / "huggingface"
    annotations_path = Path(
        downloader(
            repo_id=FLICKR30K_DATASET,
            repo_type="dataset",
            filename=FLICKR30K_ANNOTATIONS,
            revision=FLICKR30K_REVISION,
            cache_dir=str(hub_cache),
        )
    ).resolve()
    archive_path = Path(
        downloader(
            repo_id=FLICKR30K_DATASET,
            repo_type="dataset",
            filename=FLICKR30K_IMAGES_ARCHIVE,
            revision=FLICKR30K_REVISION,
            cache_dir=str(hub_cache),
        )
    ).resolve()
    extracted_root = _extract_flickr30k_archive(archive_path, cache_root / "extracted")
    return _locate_flickr30k_images(extracted_root, annotations_path), annotations_path


def load_flickr30k_records(images_path: str | Path, annotations_path: str | Path) -> list[SourceCaptionImage]:
    images_root = Path(images_path).resolve()
    if not images_root.is_dir():
        raise FileNotFoundError(f"Flickr30k image directory not found: {images_root}")
    payload = json.loads(Path(annotations_path).read_text(encoding="utf-8"))
    entries = payload.get("images") if isinstance(payload, dict) else payload
    if not isinstance(entries, list):
        raise ValueError("Flickr30k annotations must contain an images list")
    records = []
    for entry in entries:
        filename = entry.get("filename") or entry.get("file_name")
        image = (images_root / filename).resolve()
        if images_root not in image.parents or not image.is_file():
            raise FileNotFoundError(f"Flickr30k image missing or outside image directory: {filename}")
        captions = []
        for sentence in entry.get("sentences", entry.get("captions", [])):
            if isinstance(sentence, str):
                captions.append(sentence)
            elif sentence.get("raw"):
                captions.append(sentence["raw"])
            elif sentence.get("tokens"):
                captions.append(" ".join(sentence["tokens"]))
        if not captions:
            raise ValueError(f"image has no captions: {filename}")
        records.append(SourceCaptionImage(str(entry.get("imgid", entry.get("image_id", filename))), str(image), tuple(captions), str(entry.get("split", "unspecified"))))
    return records


def load_coco_karpathy_records(cache_dir: str | Path, *, max_records: int | None = None) -> list[SourceCaptionImage]:
    """Download and normalize the immutable, revision-pinned COCO Karpathy source."""
    try:
        from datasets import load_dataset
    except ImportError as error:
        raise RuntimeError("COCO loading requires the optional 'datasets' dependency") from error
    cache_root = Path(cache_dir).resolve()
    dataset = load_dataset(COCO_DATASET, revision=COCO_REVISION, cache_dir=str(cache_root / "huggingface"))
    records = []
    for split_name, split in dataset.items():
        for index, row in enumerate(split):
            filename = row.get("filename")
            folder = row.get("filepath", "images")
            image_path = cache_root / "images" / folder / filename
            if not image_path.exists():
                if not row.get("url"):
                    raise ValueError(f"COCO row has no downloadable image URL: {filename}")
                with urlopen(row["url"], timeout=60) as response:
                    atomic_write_bytes(image_path, response.read())
            captions: Any = row.get("sentences") or row.get("captions") or row.get("caption")
            if isinstance(captions, str):
                captions = [captions]
            normalized = []
            for caption in captions or []:
                normalized.append(caption.get("raw", " ".join(caption.get("tokens", []))) if isinstance(caption, dict) else str(caption))
            records.append(SourceCaptionImage(str(row.get("cocoid", row.get("image_id", f"{split_name}-{index}"))), str(image_path), tuple(normalized), str(row.get("split", split_name))))
            if max_records is not None and len(records) >= max_records:
                return records
    return records
