from __future__ import annotations

import statistics
import time
from abc import ABC, abstractmethod
from dataclasses import asdict, dataclass
from typing import Callable

import torch


@dataclass(frozen=True)
class MeasurementProtocol:
    warmups: int = 10
    repeats: int = 50
    synchronize_cuda: bool = True
    schema_version: str = "1.0"

    def __post_init__(self) -> None:
        if self.warmups < 0 or self.repeats < 1:
            raise ValueError("warmups must be non-negative and repeats must be positive")


@dataclass(frozen=True)
class HardwareMeasurement:
    backend: str
    server_fake_quant_ms: dict[str, float]
    sample_count: int
    diagnostic_only: bool = True
    claim_scope: str = "diagnostic_only_no_jetson_claims"
    schema_version: str = "1.0"

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


class HardwareBackend(ABC):
    version = "1.0"

    @abstractmethod
    def measure(self, bundle: Callable[[], object], protocol: MeasurementProtocol) -> HardwareMeasurement:
        raise NotImplementedError


def _sync(enabled: bool) -> None:
    if enabled and torch.cuda.is_available():
        torch.cuda.synchronize()


def _percentile(samples: list[float], fraction: float) -> float:
    ordered = sorted(samples)
    position = (len(ordered) - 1) * fraction
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    return ordered[lower] + (ordered[upper] - ordered[lower]) * (position - lower)


class ServerTimingBackend(HardwareBackend):
    def measure(self, bundle: Callable[[], object], protocol: MeasurementProtocol) -> HardwareMeasurement:
        for _ in range(protocol.warmups):
            bundle()
        _sync(protocol.synchronize_cuda)
        samples = []
        for _ in range(protocol.repeats):
            _sync(protocol.synchronize_cuda)
            start = time.perf_counter_ns()
            bundle()
            _sync(protocol.synchronize_cuda)
            samples.append((time.perf_counter_ns() - start) / 1_000_000)
        return HardwareMeasurement("server_fake_quant", {"p50": statistics.median(samples), "p95": _percentile(samples, 0.95)}, len(samples))
