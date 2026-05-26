from typing import Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.ops import Conv2dNormActivation


class FpnFusion(nn.Module):
    def __init__(self, in_channels: Sequence[int], out_channels: int = 256) -> None:
        super().__init__()
        self.num_levels = len(in_channels)
        assert out_channels % self.num_levels == 0
        self.out_channels = out_channels
        reduced_channels = self.out_channels // self.num_levels

        self.lateral_convs = nn.ModuleList(
            [
                nn.Conv2d(in_ch, reduced_channels, kernel_size=1, bias=False)
                for in_ch in in_channels
            ]
        )

        self.smooth_convs = nn.ModuleList(
            [
                Conv2dNormActivation(
                    reduced_channels, reduced_channels, kernel_size=3, padding=1
                )
                for _ in range(self.num_levels)
            ]
        )

        self.fusion = Conv2dNormActivation(
            self.out_channels, self.out_channels, kernel_size=3, padding=1
        )

    def forward(self, x: Sequence[torch.Tensor]) -> torch.Tensor:
        assert len(x) == self.num_levels

        lateral = [conv(f) for conv, f in zip(self.lateral_convs, x)]

        # Smooth the top level first (was missing before)
        lateral[-1] = self.smooth_convs[-1](lateral[-1])

        # Top-down fusion
        for i in range(self.num_levels - 2, -1, -1):
            upsampled = F.interpolate(
                lateral[i + 1],
                size=lateral[i].shape[2:],
                mode="bilinear",
                align_corners=False,
            )
            fused = upsampled + lateral[i]
            lateral[i] = self.smooth_convs[i](fused)

        target_size = lateral[0].shape[2:]
        fused = torch.cat(
            [
                F.interpolate(l, size=target_size, mode="bilinear", align_corners=False)
                for l in lateral
            ],
            dim=1,
        )

        return self.fusion(fused)
