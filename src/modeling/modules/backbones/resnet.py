from typing import Sequence

import torch
import torch.nn as nn

from src.modeling.layers import DeformConv2d


def conv3x3(in_channels: int, out_channels: int, stride: int = 1) -> nn.Conv2d:
    return nn.Conv2d(
        in_channels,
        out_channels,
        kernel_size=3,
        stride=stride,
        padding=1,
        bias=False,
    )


def deform_conv3x3(
    in_channels: int, out_channels: int, stride: int = 1
) -> DeformConv2d:
    return DeformConv2d(
        in_channels,
        out_channels,
        kernel_size=3,
        stride=stride,
        padding=1,
        bias=False,
    )


def conv1x1(in_planes: int, out_planes: int, stride: int = 1) -> nn.Conv2d:
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(
        self,
        inplanes: int,
        planes: int,
        stride: int = 1,
        downsample: nn.Module | None = None,
        deformable: bool = False,
    ) -> None:
        super().__init__()
        self.deformable = deformable
        self.downsample = downsample or nn.Identity()

        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        if not deformable:
            self.conv2 = conv3x3(planes, planes)
        else:
            self.conv2 = deform_conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.act = nn.ReLU(inplace=True)

        self._init_weights()

    def _init_weights(self) -> None:
        nn.init.kaiming_normal_(
            self.conv1.weight,
            mode="fan_out",
            nonlinearity="relu",
        )

        if not isinstance(self.conv2, DeformConv2d):
            nn.init.kaiming_normal_(
                self.conv2.weight,
                mode="fan_out",
                nonlinearity="relu",
            )

        nn.init.ones_(self.bn1.weight)
        nn.init.zeros_(self.bn1.bias)

        nn.init.zeros_(self.bn2.weight)
        nn.init.zeros_(self.bn2.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = self.downsample(x)

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.act(out)

        out = self.conv2(out)
        out = self.bn2(out)

        out += residual
        out = self.act(out)

        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(
        self,
        inplanes: int,
        planes: int,
        stride: int = 1,
        downsample: nn.Module | None = None,
        deformable: bool = False,
    ) -> None:
        super().__init__()

        self.deformable = deformable
        self.downsample = downsample or nn.Identity()

        self.conv1 = conv1x1(inplanes, planes)
        self.bn1 = nn.BatchNorm2d(planes)
        if not deformable:
            self.conv2 = conv3x3(planes, planes, stride=stride)
        else:
            self.conv2 = deform_conv3x3(planes, planes, stride=stride)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = conv1x1(planes, planes * self.expansion)
        self.bn3 = nn.BatchNorm2d(planes * self.expansion)
        self.act = nn.ReLU(inplace=True)

        self._init_weights()

    def _init_weights(self) -> None:
        nn.init.kaiming_normal_(
            self.conv1.weight,
            mode="fan_out",
            nonlinearity="relu",
        )

        if not isinstance(self.conv2, DeformConv2d):
            nn.init.kaiming_normal_(
                self.conv2.weight,
                mode="fan_out",
                nonlinearity="relu",
            )

        nn.init.kaiming_normal_(
            self.conv3.weight,
            mode="fan_out",
            nonlinearity="relu",
        )

        nn.init.ones_(self.bn1.weight)
        nn.init.zeros_(self.bn1.bias)

        nn.init.ones_(self.bn2.weight)
        nn.init.zeros_(self.bn2.bias)

        nn.init.zeros_(self.bn3.weight)
        nn.init.zeros_(self.bn3.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = self.downsample(x)

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.act(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.act(out)

        out = self.conv3(out)
        out = self.bn3(out)

        out += residual
        out = self.act(out)

        return out


class ResNet(nn.Module):
    def __init__(
        self,
        block_type: type[BasicBlock | Bottleneck],
        blocks_per_stage: Sequence[int],
        stem_width: int = 64,
        deformable: bool = False,
    ) -> None:
        super().__init__()

        self.block_type = block_type
        self.deformable = deformable

        self.in_channels = stem_width

        self.conv1 = nn.Conv2d(
            3,
            self.in_channels,
            kernel_size=7,
            stride=2,
            padding=3,
            bias=False,
        )

        self.bn1 = nn.BatchNorm2d(self.in_channels)
        self.act = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.stages = nn.ModuleList(
            [
                self._create_stage(
                    planes=stem_width * 2**i,
                    blocks=stage_blocks,
                    stride=1 if i == 0 else 2,
                    deformable=self.deformable if i > 0 else False,
                )
                for i, stage_blocks in enumerate(blocks_per_stage)
            ]
        )

        self.out_channels = [
            stem_width * 2**i * self.block_type.expansion
            for i in range(len(blocks_per_stage))
        ]

    def _init_weights(self) -> None:
        nn.init.kaiming_normal_(
            self.conv1.weight,
            mode="fan_out",
            nonlinearity="relu",
        )
        nn.init.ones_(self.bn1.weight)
        nn.init.zeros_(self.bn1.bias)

    def forward(self, x) -> list[torch.Tensor]:
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.act(x)
        x = self.maxpool(x)

        outputs = []
        for stage in self.stages:
            x = stage(x)
            outputs.append(x)

        return outputs

    def _create_stage(
        self, planes: int, blocks: int, stride: int = 1, deformable: bool = False
    ) -> nn.Sequential:
        downsample = None
        if stride != 1 or self.in_channels != planes * self.block_type.expansion:
            downsample = nn.Sequential(
                conv1x1(
                    self.in_channels, planes * self.block_type.expansion, stride=stride
                ),
                nn.BatchNorm2d(planes * self.block_type.expansion),
            )

        layers = []
        layers.append(
            self.block_type(
                self.in_channels,
                planes,
                stride,
                downsample,
                deformable,
            )
        )

        self.in_channels = planes * self.block_type.expansion
        for _ in range(blocks - 1):
            layers.append(
                self.block_type(
                    self.in_channels,
                    planes,
                    deformable=deformable,
                )
            )

        return nn.Sequential(*layers)


def resnet18(deformable: bool = False):
    return ResNet(BasicBlock, [2, 2, 2, 2], deformable=deformable)


def resnet34(deformable: bool = False):
    return ResNet(BasicBlock, [3, 4, 6, 3], deformable=deformable)


def resnet50(deformable: bool = False):
    return ResNet(Bottleneck, [3, 4, 6, 3], deformable=deformable)


def resnet101(deformable: bool = False):
    return ResNet(Bottleneck, [3, 4, 23, 3], deformable=deformable)


def resnet152(deformable: bool = False):
    return ResNet(Bottleneck, [3, 8, 36, 3], deformable=deformable)
