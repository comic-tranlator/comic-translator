import torch
import torch.nn as nn

from src.modeling.modules import StyleNetPrediction
from src.util.color import hex_to_rgb, rgb_to_lab


class StyleNetLoss(nn.Module):
    def __init__(
        self,
        max_stroke_thickness: int = 8,
        font_color_weight: float = 1.0,
        stroke_color_weight: float = 1.0,
        stroke_thickness_weight: float = 0.25,
    ) -> None:
        super().__init__()

        self.max_stroke_thickness = max_stroke_thickness

        self.font_color_weight = font_color_weight
        self.stroke_color_weight = stroke_color_weight

        self.font_color_loss_fn = nn.SmoothL1Loss()
        self.stroke_color_loss_fn = nn.SmoothL1Loss()

    def forward(self, pred: StyleNetPrediction, gt: dict) -> torch.Tensor:
        gt_font_color_lab = self._color_target_to_tensor(
            gt["font_color"], pred.font_color
        )
        font_color_loss = self.font_color_loss_fn(pred.font_color, gt_font_color_lab)

        gt_stroke_color_lab = self._color_target_to_tensor(
            gt["stroke_color"], pred.stroke_color
        )
        stroke_color_loss = self.stroke_color_loss_fn(
            pred.stroke_color, gt_stroke_color_lab
        )

        total = (
            self.font_color_weight * font_color_loss
            + self.stroke_color_weight * stroke_color_loss
        )

        return total

    def _color_target_to_tensor(
        self, value: object, pred_ref: torch.Tensor
    ) -> torch.Tensor:
        if isinstance(value, str):
            color = torch.tensor(
                rgb_to_lab(hex_to_rgb(value)),
                dtype=pred_ref.dtype,
                device=pred_ref.device,
            )
            return color.unsqueeze(0).expand(pred_ref.shape[0], -1)

        if isinstance(value, torch.Tensor):
            return value.to(device=pred_ref.device, dtype=pred_ref.dtype)

        if isinstance(value, (list, tuple)):
            if value and all(isinstance(channel, torch.Tensor) for channel in value):
                return torch.stack(list(value), dim=-1).to(
                    device=pred_ref.device,
                    dtype=pred_ref.dtype,
                )

            color = torch.tensor(
                value,
                dtype=pred_ref.dtype,
                device=pred_ref.device,
            )
            if color.ndim == 1:
                color = color.unsqueeze(0)
            return color

        raise TypeError(f"Unsupported color target type: {type(value)!r}")
