from __future__ import annotations

from typing import Any

from src.models.resnet_cifar import build_cifar_resnet
from src.models.resnet_cifar_qat import build_cifar_resnet_qat


def build_model_from_config(config: dict[str, Any], num_classes: int):
    model_name = config["model"]["name"]
    if "qat" in config:
        qat_config = config.get("qat", {})
        return build_cifar_resnet_qat(
            model_name,
            num_classes=num_classes,
            w_bits=int(qat_config.get("weight_bits", 8)),
            a_bits=int(qat_config.get("activation_bits", 8)),
        )
    return build_cifar_resnet(model_name, num_classes=num_classes)

