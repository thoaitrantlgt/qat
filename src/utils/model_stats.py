from __future__ import annotations

from src.quantization.bitops import collect_resource_stats


def collect_model_stats(model, input_size: tuple[int, int, int] = (3, 32, 32)) -> dict:
    return collect_resource_stats(model, input_size=input_size)
