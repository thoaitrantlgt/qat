from __future__ import annotations

import torch
from torch import nn


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes: int, planes: int, stride: int = 1) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Identity()
        if stride != 1 or in_planes != planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes),
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out = out + self.shortcut(x)
        return self.relu(out)


class CifarResNet(nn.Module):
    def __init__(self, depth: int, num_classes: int) -> None:
        super().__init__()
        if (depth - 2) % 6 != 0:
            raise ValueError("CIFAR ResNet depth must satisfy depth = 6n + 2.")
        blocks_per_stage = (depth - 2) // 6
        self.in_planes = 16

        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(16)
        self.relu = nn.ReLU(inplace=True)
        self.layer1 = self._make_layer(16, blocks_per_stage, stride=1)
        self.layer2 = self._make_layer(32, blocks_per_stage, stride=2)
        self.layer3 = self._make_layer(64, blocks_per_stage, stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(64, num_classes)

        self._initialize_weights()

    def _make_layer(self, planes: int, num_blocks: int, stride: int) -> nn.Sequential:
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for block_stride in strides:
            layers.append(BasicBlock(self.in_planes, planes, block_stride))
            self.in_planes = planes
        return nn.Sequential(*layers)

    def _initialize_weights(self) -> None:
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.kaiming_normal_(module.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(module, nn.BatchNorm2d):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, 0, 0.01)
                nn.init.zeros_(module.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.avgpool(out)
        out = torch.flatten(out, 1)
        return self.fc(out)


def resnet20(num_classes: int) -> CifarResNet:
    return CifarResNet(depth=20, num_classes=num_classes)


def resnet32(num_classes: int) -> CifarResNet:
    return CifarResNet(depth=32, num_classes=num_classes)


def resnet56(num_classes: int) -> CifarResNet:
    return CifarResNet(depth=56, num_classes=num_classes)


def build_cifar_resnet(name: str, num_classes: int) -> CifarResNet:
    builders = {
        "resnet20": resnet20,
        "resnet32": resnet32,
        "resnet56": resnet56,
    }
    key = name.lower().replace("-", "")
    if key not in builders:
        raise ValueError(f"Unsupported CIFAR model '{name}'. Use one of {sorted(builders)}.")
    return builders[key](num_classes=num_classes)

