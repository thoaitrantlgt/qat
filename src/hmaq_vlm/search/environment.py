from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Callable

from hmaq_vlm.quantization import MixedPrecisionPolicy
from hmaq_vlm.reproducibility import atomic_write_json, stable_hash


@dataclass(frozen=True)
class CandidateResult:
    policy_hash: str
    valid: bool
    reward: float
    metrics: dict[str, float]
    failure: str | None = None
    cached: bool = False
    schema_version: str = "1.0"


class SearchEnvironment:
    def __init__(self, groups: list[str], evaluator: Callable[[MixedPrecisionPolicy], dict[str, float]], *, timing_evaluator: Callable[[MixedPrecisionPolicy], dict[str, object]] | None = None, cache_path: str | Path, model_hash: str, dataset_hash: str, runtime_hash: str, protocol_hash: str) -> None:
        self.groups = tuple(groups)
        self.evaluator = evaluator
        self.timing_evaluator = timing_evaluator
        self.cache_path = Path(cache_path)
        self.context = {"model": model_hash, "dataset": dataset_hash, "runtime": runtime_hash, "protocol": protocol_hash}
        self._cache = json.loads(self.cache_path.read_text(encoding="utf-8")) if self.cache_path.exists() else {}

    def _key(self, policy: MixedPrecisionPolicy) -> str:
        return stable_hash({"context": self.context, "policy": policy.to_dict()})

    def evaluate(self, policy: MixedPrecisionPolicy) -> CandidateResult:
        if set(policy.actions) != set(self.groups):
            raise ValueError("policy groups do not match search environment")
        key = self._key(policy)
        if key in self._cache:
            return replace(CandidateResult(**self._cache[key]), cached=True)
        try:
            metrics = {name: float(value) for name, value in self.evaluator(policy).items()}
            required = {"cider", "bitops_ratio", "model_size_ratio", "prefix_distortion", "logit_kl"}
            if required - set(metrics):
                raise ValueError(f"missing candidate metrics: {sorted(required-set(metrics))}")
            if not all(math.isfinite(value) for value in metrics.values()):
                raise ValueError("non-finite candidate")
            if metrics["bitops_ratio"] > 1.05 or metrics["model_size_ratio"] > 1.05:
                raise ValueError("clearly over-budget candidate")
            if metrics["cider"] < 0:
                raise ValueError("catastrophically degraded candidate")
            penalty = max(0.0, metrics["bitops_ratio"] - 1.0) * 10 + max(0.0, metrics["model_size_ratio"] - 1.0) * 10
            reward = metrics["cider"] - 0.25 * metrics["bitops_ratio"] - 0.15 * metrics["model_size_ratio"] - metrics["prefix_distortion"] - metrics["logit_kl"] - penalty
            result = CandidateResult(key, True, reward, metrics)
        except (ValueError, RuntimeError, FloatingPointError) as error:
            result = CandidateResult(key, False, -1.0e9, {}, str(error))
        self._cache[key] = asdict(result)
        atomic_write_json(self.cache_path, self._cache)
        return result

    def measure_diagnostic(self, policy: MixedPrecisionPolicy) -> dict[str, object] | None:
        return self.timing_evaluator(policy) if self.timing_evaluator is not None else None
