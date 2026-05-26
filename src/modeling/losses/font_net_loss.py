import torch
import torch.nn as nn

from src.modeling.models import FontNetPrediction

class FontNetLoss(nn.Module):
    def __init__(self) -> None:
        super().__init__()

        self.typeface_loss_fn = nn.CrossEntropyLoss()
        self.weight_loss_fn = nn.SmoothL1Loss()
        self.posture_loss_fn = nn.CrossEntropyLoss()

    def forward(self, pred: FontNetPrediction, gt: dict) -> torch.Tensor:
        typeface_loss = self.typeface_loss_fn(pred.typeface, gt["typeface"])

        gt_weight_normalized = ((gt["weight"].float() - 100.0) / 800).unsqueeze(1)
        gt_weight_normalized = gt_weight_normalized.to(pred.weight.dtype)
        weight_loss = self.weight_loss_fn(pred.weight, gt_weight_normalized)

        posture_loss = self.posture_loss_fn(pred.posture, gt["posture"])
        total = typeface_loss + weight_loss * 0.5 + posture_loss * 0.3

        return total
