from __future__ import annotations

from dataclasses import dataclass

from torch import nn


@dataclass(frozen=True)
class QuantGroup:
    name: str
    module_path: str
    modality: str
    module_type: str
    parameters: int


def build_quant_group_registry(model: nn.Module) -> list[QuantGroup]:
    groups: list[QuantGroup] = []
    for name, module in model.named_modules():
        is_linear = isinstance(module, nn.Linear)
        is_conv1d = module.__class__.__name__ == "Conv1D" and hasattr(module, "nf") and hasattr(module, "nx")
        if not name or not (is_linear or is_conv1d) or "lm_head" in name or "output_head" in name:
            continue
        if name.startswith("vision_encoder"):
            modality = "vision"
        elif name.startswith("projector"):
            modality = "projector"
        elif name.startswith("language_model"):
            modality = "language"
        else:
            continue
        groups.append(QuantGroup(name, name, modality, "linear" if is_linear else "conv1d", sum(parameter.numel() for parameter in module.parameters())))
    return sorted(groups, key=lambda group: group.name)
