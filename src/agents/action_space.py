from __future__ import annotations

from collections.abc import Sequence


DEFAULT_BIT_CHOICES = (2, 3, 4, 6, 8)
DEFAULT_ACTION_BITS = tuple(
    (weight_bits, activation_bits)
    for weight_bits in DEFAULT_BIT_CHOICES
    for activation_bits in DEFAULT_BIT_CHOICES
)


def build_action_bits(bit_choices: Sequence[int] | None = None) -> tuple[tuple[int, int], ...]:
    choices = tuple(int(bit) for bit in (bit_choices or DEFAULT_BIT_CHOICES))
    if not choices:
        raise ValueError("bit_choices must contain at least one bit-width.")
    if any(bit < 2 for bit in choices):
        raise ValueError(f"bit_choices must be >= 2, got {choices}.")
    return tuple((weight_bits, activation_bits) for weight_bits in choices for activation_bits in choices)
