from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import Any

import torch


class CaptionCollator:
    def __init__(self, image_processor: Callable[..., Any], tokenizer: Callable[..., Any], max_length: int = 64) -> None:
        self.image_processor = image_processor
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __call__(self, examples: list[Any]) -> dict[str, torch.Tensor]:
        try:
            from PIL import Image
        except ImportError as error:
            raise RuntimeError("caption collation requires Pillow") from error
        images = [Image.open(Path(example.image_path)).convert("RGB") for example in examples]
        captions = [example.captions[0] for example in examples]
        pixels = self.image_processor(images=images, return_tensors="pt")["pixel_values"]
        tokens = self.tokenizer(captions, padding=True, truncation=True, max_length=self.max_length, return_tensors="pt")
        labels = tokens["input_ids"].clone().masked_fill(tokens["attention_mask"] == 0, -100)
        return {"pixel_values": pixels, "input_ids": tokens["input_ids"], "attention_mask": tokens["attention_mask"], "labels": labels}
