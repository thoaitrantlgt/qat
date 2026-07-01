from __future__ import annotations

from collections.abc import Sequence


DEFAULT_BIT_CHOICES = (2, 3, 4, 6, 8)
DEFAULT_ACTION_BITS = tuple(
    (weight_bits, activation_bits)
    for weight_bits in DEFAULT_BIT_CHOICES
    for activation_bits in DEFAULT_BIT_CHOICES
)


def _normalize_choices(name: str, choices: Sequence[int] | None) -> tuple[int, ...]:
    normalized = tuple(int(bit) for bit in (choices or DEFAULT_BIT_CHOICES))
    if not normalized:
        raise ValueError(f"{name} must contain at least one bit-width.")
    if any(bit < 2 for bit in normalized):
        raise ValueError(f"{name} must be >= 2, got {normalized}.")
    return normalized


def build_action_bits(
    bit_choices: Sequence[int] | None = None,
    weight_bit_choices: Sequence[int] | None = None,
    activation_bit_choices: Sequence[int] | None = None,
) -> tuple[tuple[int, int], ...]:
    weight_choices = _normalize_choices("weight_bit_choices", weight_bit_choices or bit_choices)
    activation_choices = _normalize_choices("activation_bit_choices", activation_bit_choices or bit_choices)
    return tuple(
        (weight_bits, activation_bits)
        for weight_bits in weight_choices
        for activation_bits in activation_choices
    )
