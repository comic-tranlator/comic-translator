from collections import namedtuple

import torch
import torch.nn as nn

StyleNetPrediction = namedtuple("StyleNetPrediction", ["font_color", "stroke_color"])


class StyleNetHead(nn.Module):
    def __init__(self, in_features: int, dropout: float = 0.3) -> None:
        super().__init__()

        def color_head():
            return nn.Sequential(
                nn.Dropout(dropout),
                nn.Linear(in_features, 64),
                nn.GELU(),
                nn.Linear(64, 3),
            )

        self.text_color_head = color_head()
        self.stroke_color_head = color_head()

    def _activate(self, x: torch.Tensor) -> torch.Tensor:
        L = torch.sigmoid(x[..., :1])  # L ∈ [0, 1]
        ab = torch.tanh(x[..., 1:])  # a, b ∈ [-1, 1]
        return torch.cat([L, ab], dim=-1)

    def forward(self, x: torch.Tensor) -> StyleNetPrediction:
        return StyleNetPrediction(
            self._activate(self.text_color_head(x)),
            self._activate(self.stroke_color_head(x)),
        )
