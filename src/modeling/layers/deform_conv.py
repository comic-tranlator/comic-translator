import torch
import torch.nn as nn
from torchvision.ops import DeformConv2d as _DeformConv2d


class DeformConv2d(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        stride: int = 1,
        padding: int = 0,
        dilation: int = 1,
        bias: bool = True,
    ):
        super().__init__()

        self.kernel_size = kernel_size

        # Predicts x, y offsets + mask weight for each kernel position
        self.offset_mask_conv = nn.Conv2d(
            in_channels,
            3 * kernel_size * kernel_size,  # 2k² for x,y offsets + k² for masks
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            bias=True,
        )

        self.deform_conv = _DeformConv2d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            bias=bias,
        )

        self._init_weights()

    def _init_weights(self) -> None:
        # Zero-init makes the layer behave exactly like
        # a standard conv at the start of training
        nn.init.zeros_(self.offset_mask_conv.weight)
        if self.offset_mask_conv.bias is not None:
            nn.init.zeros_(self.offset_mask_conv.bias)

        nn.init.kaiming_normal_(
            self.deform_conv.weight,
            mode="fan_out",
            nonlinearity="relu",
        )
        if self.deform_conv.bias is not None:
            nn.init.zeros_(self.deform_conv.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        k2 = self.kernel_size * self.kernel_size

        out = self.offset_mask_conv(x)  # (B, 3 * k * k, H, W)

        offset = out[:, : 2 * k2]  # (B, 2 * k * k, H, W)
        mask = torch.sigmoid(
            out[:, 2 * k2 :]
        )  # (B, k * k, H, W) - how much each position contributes

        return self.deform_conv(x, offset, mask)
