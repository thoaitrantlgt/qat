from __future__ import annotations

import json
from urllib.request import urlopen
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from hmaq_vlm.reproducibility import atomic_write_bytes


COCO_DATASET = "yerevann/coco-karpathy"
COCO_REVISION = "448fdb1bc7b2d09e46881c4541a14d796a3d41e8"


@dataclass(frozen=True)
class SourceCaptionImage:
    image_id: str
    image_path: str
    captions: tuple[str, ...]
    source_split: str


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
