from collections import namedtuple

import torch.nn as nn

FontNetPrediction = namedtuple("FontNetPrediction", ["typeface", "weight", "posture"])


import timm

from src.util.font import Posture, TypeFace


class FontNet(nn.Module):
    def __init__(self, dropout: float = 0.3):
        super().__init__()
        # DINOv2 is especially good — pretrained on visual structure, not ImageNet labels
        self.backbone = timm.create_model(
            "vit_small_patch16_224.dino",
            pretrained=True,
            num_classes=0,
        )
        embed_dim = self.backbone.embed_dim

        self.drop = nn.Dropout(dropout)

        self.typeface_head = nn.Linear(embed_dim, len(TypeFace))
        self.posture_head = nn.Linear(embed_dim, len(Posture))
        self.weight_head = nn.Sequential(nn.Linear(embed_dim, 1), nn.Sigmoid())

    def forward(self, x):
        feat = self.drop(self.backbone(x))
        return FontNetPrediction(
            self.typeface_head(feat),
            self.weight_head(feat),
            self.posture_head(feat),
        )
