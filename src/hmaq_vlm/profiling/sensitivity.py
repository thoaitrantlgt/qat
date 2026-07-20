from __future__ import annotations

from collections.abc import Callable, Iterable
import random
import numpy as np
from typing import Any

import torch
from torch import nn

from hmaq_vlm.quantization import MixedPrecisionPolicy, PrecisionAction, QuantGroup, temporary_policy


REQUIRED_METRICS = {"caption_loss_delta", "prefix_distortion", "logit_kl", "activation_range", "gradient_norm", "parameters", "bitops", "model_size_bytes"}


class SensitivityProfiler:
    def __init__(self, model: nn.Module, registry: list[QuantGroup], probe: Callable[[nn.Module, QuantGroup, PrecisionAction], dict[str, Any]]) -> None:
        self.model = model
        self.registry = registry
        self.probe = probe

    def profile(self, actions: Iterable[PrecisionAction]) -> list[dict[str, Any]]:
        state = {name: value.detach().clone() for name, value in self.model.state_dict().items()}
        module_modes = {name: module.training for name, module in self.model.named_modules()}
        python_rng = random.getstate()
        numpy_rng = np.random.get_state()
        torch_rng = torch.get_rng_state()
        cuda_rng = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
        baseline = {group.name: PrecisionAction(16, 16) for group in self.registry}
        rows = []
        try:
            for group in self.registry:
                for action in actions:
                    selected = dict(baseline)
                    selected[group.name] = action
                    with temporary_policy(self.model, self.registry, MixedPrecisionPolicy(selected)):
                        metrics = self.probe(self.model, group, action)
                    missing = REQUIRED_METRICS - set(metrics)
                    if missing:
                        raise ValueError(f"probe omitted metrics: {sorted(missing)}")
                    values = [float(metrics[key]) for key in REQUIRED_METRICS]
                    if not all(torch.isfinite(torch.tensor(value)) for value in values):
                        raise ValueError("non-finite sensitivity metric")
                    rows.append({"schema_version": "1.0", "group": group.name, "modality": group.modality, "weight_bits": action.weight_bits, "activation_bits": action.activation_bits, **metrics})
        finally:
            self.model.load_state_dict(state)
            for name, module in self.model.named_modules():
                module.train(module_modes[name])
            random.setstate(python_rng)
            np.random.set_state(numpy_rng)
            torch.set_rng_state(torch_rng)
            if cuda_rng is not None:
                torch.cuda.set_rng_state_all(cuda_rng)
        return rows
