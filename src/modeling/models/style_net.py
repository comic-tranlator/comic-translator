from typing import Callable

import timm
import torch
import torch.nn as nn

from src.modeling.modules import StyleNetHead, StyleNetPrediction


class StyleNet(nn.Module):
    def __init__(
        self,
        backbone: str,
        head: Callable[[torch.Tensor | nn.Module], StyleNetHead],
        pretrained: bool = True,
    ) -> None:
        super().__init__()
        self.backbone = timm.create_model(
            backbone,
            pretrained=pretrained,
            num_classes=0,
            global_pool="avg",
        )
        in_features = self.backbone.num_features
        self.head = head(in_features)

    def forward(self, x: torch.Tensor) -> StyleNetPrediction:
        feats = self.backbone(x)
        return self.head(feats)
