from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path

from hmaq_vlm.reproducibility import atomic_write_json, file_sha256


@dataclass(frozen=True)
class JetsonExportBundle:
    model_path: str
    policy_path: str
    input_shape: tuple[int, ...] = (1, 3, 224, 224)
    batch_size: int = 1
    schema_version: str = "1.0"

    def write_contract(self, path: str | Path) -> None:
        payload = asdict(self)
        payload["checksums"] = {"model": file_sha256(self.model_path), "policy": file_sha256(self.policy_path)}
        payload["result_schema"] = {"backend": "jetson_backend_identifier", "native_precision_verified": "boolean", "latency_ms": {"p50": "number", "p95": "number"}, "energy_j": "number_or_null"}
        atomic_write_json(path, payload)
