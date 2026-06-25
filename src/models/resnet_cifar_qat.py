from __future__ import annotations

import torch
from torch import nn

from src.quantization.policy_applier import set_uniform_bit_widths
from src.quantization.quant_layers import QuantConv2d, QuantLinear


class QATBasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes: int, planes: int, stride: int = 1, w_bits: int = 8, a_bits: int = 8) -> None:
        super().__init__()
        self.conv1 = QuantConv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False, w_bits=w_bits, a_bits=a_bits)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = QuantConv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False, w_bits=w_bits, a_bits=a_bits)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Identity()
        if stride != 1 or in_planes != planes:
            self.shortcut = nn.Sequential(
                QuantConv2d(in_planes, planes, kernel_size=1, stride=stride, bias=False, w_bits=w_bits, a_bits=a_bits),
                nn.BatchNorm2d(planes),
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out = out + self.shortcut(x)
        return self.relu(out)


class QATCifarResNet(nn.Module):
    def __init__(self, depth: int, num_classes: int, w_bits: int = 8, a_bits: int = 8) -> None:
        super().__init__()
        if (depth - 2) % 6 != 0:
            raise ValueError("CIFAR ResNet depth must satisfy depth = 6n + 2.")
        blocks_per_stage = (depth - 2) // 6
        self.in_planes = 16

        self.conv1 = QuantConv2d(3, 16, kernel_size=3, stride=1, padding=1, bias=False, w_bits=w_bits, a_bits=a_bits)
        self.bn1 = nn.BatchNorm2d(16)
        self.relu = nn.ReLU(inplace=True)
        self.layer1 = self._make_layer(16, blocks_per_stage, stride=1, w_bits=w_bits, a_bits=a_bits)
        self.layer2 = self._make_layer(32, blocks_per_stage, stride=2, w_bits=w_bits, a_bits=a_bits)
        self.layer3 = self._make_layer(64, blocks_per_stage, stride=2, w_bits=w_bits, a_bits=a_bits)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = QuantLinear(64, num_classes, w_bits=w_bits, a_bits=a_bits)

        self._initialize_weights()

    def _make_layer(self, planes: int, num_blocks: int, stride: int, w_bits: int, a_bits: int) -> nn.Sequential:
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for block_stride in strides:
            layers.append(QATBasicBlock(self.in_planes, planes, block_stride, w_bits=w_bits, a_bits=a_bits))
            self.in_planes = planes
        return nn.Sequential(*layers)

    def _initialize_weights(self) -> None:
        for module in self.modules():
            if isinstance(module, (nn.Conv2d, QuantConv2d)):
                if hasattr(module, "weight") and isinstance(module.weight, torch.Tensor):
                    nn.init.kaiming_normal_(module.weight, mode="fan_out", nonlinearity="relu")
                if hasattr(module, "bias") and getattr(module, "bias", None) is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.BatchNorm2d):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)
            elif isinstance(module, (nn.Linear, QuantLinear)):
                if hasattr(module, "weight") and isinstance(module.weight, torch.Tensor):
                    nn.init.normal_(module.weight, 0, 0.01)
                if hasattr(module, "bias") and getattr(module, "bias", None) is not None:
                    nn.init.zeros_(module.bias)

    def set_uniform_bits(self, w_bits: int, a_bits: int, first_last_bits: int | None = None) -> None:
        set_uniform_bit_widths(self, w_bits, a_bits)
        if first_last_bits is not None:
            self.conv1.set_bits(first_last_bits, first_last_bits)
            self.fc.set_bits(first_last_bits, first_last_bits)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.avgpool(out)
        out = torch.flatten(out, 1)
        return self.fc(out)


def qat_resnet20(num_classes: int, w_bits: int = 8, a_bits: int = 8) -> QATCifarResNet:
    return QATCifarResNet(depth=20, num_classes=num_classes, w_bits=w_bits, a_bits=a_bits)


def qat_resnet32(num_classes: int, w_bits: int = 8, a_bits: int = 8) -> QATCifarResNet:
    return QATCifarResNet(depth=32, num_classes=num_classes, w_bits=w_bits, a_bits=a_bits)


def qat_resnet56(num_classes: int, w_bits: int = 8, a_bits: int = 8) -> QATCifarResNet:
    return QATCifarResNet(depth=56, num_classes=num_classes, w_bits=w_bits, a_bits=a_bits)


def build_cifar_resnet_qat(name: str, num_classes: int, w_bits: int = 8, a_bits: int = 8) -> QATCifarResNet:
    builders = {
        "resnet20": qat_resnet20,
        "resnet32": qat_resnet32,
        "resnet56": qat_resnet56,
    }
    key = name.lower().replace("-", "")
    if key not in builders:
        raise ValueError(f"Unsupported CIFAR model '{name}'. Use one of {sorted(builders)}.")
    return builders[key](num_classes=num_classes, w_bits=w_bits, a_bits=a_bits)

