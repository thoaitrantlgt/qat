from __future__ import annotations

from dataclasses import dataclass
from types import MappingProxyType
from typing import Mapping


VALID_BITS = (2, 4, 8, 16)


@dataclass(frozen=True, order=True)
class PrecisionAction:
    weight_bits: int
    activation_bits: int

    def __post_init__(self) -> None:
        if self.weight_bits not in VALID_BITS or self.activation_bits not in VALID_BITS:
            raise ValueError(f"bits must be in {VALID_BITS}")

    @property
    def label(self) -> str:
        return f"W{self.weight_bits}A{self.activation_bits}"


ACTION_SPACE = tuple(PrecisionAction(w, a) for w in VALID_BITS for a in VALID_BITS)


@dataclass(frozen=True)
class MixedPrecisionPolicy:
    actions: Mapping[str, PrecisionAction]

    def __post_init__(self) -> None:
        normalized = {str(name): action if isinstance(action, PrecisionAction) else PrecisionAction(*action) for name, action in self.actions.items()}
        object.__setattr__(self, "actions", MappingProxyType(dict(sorted(normalized.items()))))

    def to_dict(self) -> dict[str, dict[str, int]]:
        return {name: {"weight_bits": action.weight_bits, "activation_bits": action.activation_bits} for name, action in self.actions.items()}
