import math
from typing import Any, Iterable

import albumentations as A
import cv2
import numpy as np
import torch
import torchvision.transforms.functional as TF
from torchvision.transforms import v2 as T

DTYPE_MAP = {
    "float": torch.float32,
    "double": torch.float64,
    "half": torch.float16,
    "int": torch.int32,
    "long": torch.int64,
    "short": torch.int16,
    "byte": torch.uint8,
    "bool": torch.bool,
}


class ToTensor:
    def __init__(
        self,
        dtype: str = "float",
        scale: bool = False,
        squeeze_channel: bool = False,
    ) -> None:
        self.squeeze_channel = squeeze_channel

        target_dtype = DTYPE_MAP[dtype]
        self.transform = T.Compose([T.ToImage(), T.ToDtype(target_dtype, scale=scale)])

    def __call__(self, x: Any) -> torch.Tensor:
        x = self.transform(x)

        if self.squeeze_channel:
            return x.squeeze()

        return x


class ToGrayscale:
    def __call__(self, x: np.ndarray) -> np.ndarray:
        if len(x.shape) == 3 and x.shape[2] == 3:
            return cv2.cvtColor(x, cv2.COLOR_RGB2GRAY)
        return x


class Normalize:
    def __init__(
        self,
        mean: tuple[float, float, float],
        std: tuple[float, float, float],
        max_pixel_value: float = 255.0,
    ) -> None:
        self.transform = A.Normalize(
            mean=mean,
            std=std,
            max_pixel_value=max_pixel_value,
        )

    def __call__(self, x: np.ndarray) -> np.ndarray:
        return self.transform(image=x)["image"]


class Resize:
    def __init__(self, w: int, h: int) -> None:
        self.size = (w, h)
        self.transform = A.Resize(w, h)

    def __call__(self, x: Any) -> Any:
        return self.transform(image=x)["image"]


class ScaleToHeight:
    target_height: int

    def __init__(self, target_height: int) -> None:
        self.target_height = target_height

    def __call__(self, x: np.ndarray) -> np.ndarray:
        height, width = x.shape[:2]
        target_width = round(self.target_height * width / height)
        return cv2.resize(
            x,
            (target_width, self.target_height),
            interpolation=cv2.INTER_LANCZOS4,
        )


class RandomPatchCrop:
    def __init__(
        self,
        patch_size: tuple[int, int],
        use_grid_sampling: bool = False,
        pad_with: int = 255,
    ) -> None:
        self.patch_size = patch_size
        self.use_grid_sampling = use_grid_sampling
        self.pad_with = pad_with

    def __call__(self, x: np.ndarray) -> np.ndarray:
        img_h, img_w = x.shape[:2]
        p_h, p_w = self.patch_size

        # Pad if smaller than patch
        if img_h < p_h or img_w < p_w:
            x = cv2.copyMakeBorder(
                x,
                0,
                max(0, p_h - img_h),
                0,
                max(0, p_w - img_w),
                cv2.BORDER_CONSTANT,
                value=self.pad_with,
            )
            img_h, img_w = x.shape[:2]

        if self.use_grid_sampling:
            num_rows = max(1, (img_h - p_h) // p_h + 1)
            num_cols = max(1, (img_w - p_w) // p_w + 1)
            start_y = min(np.random.randint(0, num_rows) * p_h, img_h - p_h)
            start_x = min(np.random.randint(0, num_cols) * p_w, img_w - p_w)
        else:
            start_y = np.random.randint(0, img_h - p_h + 1)
            start_x = np.random.randint(0, img_w - p_w + 1)

        return x[start_y : start_y + p_h, start_x : start_x + p_w]


class ForceAspectRatio:
    ratio: float

    def __init__(self, ratio: float) -> None:
        self.ratio = ratio

    def __call__(self, x: np.ndarray) -> np.ndarray:
        height, _ = x.shape[:2]
        new_width = round(height * self.ratio)
        squeezed = cv2.resize(x, (new_width, height), interpolation=cv2.INTER_LANCZOS4)
        return squeezed.astype(dtype="uint8")


class VariableAspectRatio:
    ratio_range: tuple[float, float]

    def __init__(self, ratio_range: tuple[float, float]) -> None:
        self.ratio_range = ratio_range

    def __call__(self, x: np.ndarray) -> np.ndarray:
        height, _ = x.shape[:2]
        ratio = np.random.uniform(low=self.ratio_range[0], high=self.ratio_range[1])
        new_width = round(height * ratio)
        squeezed = cv2.resize(x, (new_width, height), interpolation=cv2.INTER_LANCZOS4)
        return squeezed.astype(dtype="uint8")


class PadToModulo:
    def __init__(self, mod):
        self.mod = mod

    def __call__(self, img: torch.Tensor) -> torch.Tensor:
        _, h, w = img.shape
        pad_h = self.ceil_modulo(h, self.mod) - h
        pad_w = self.ceil_modulo(w, self.mod) - w
        if pad_h == 0 and pad_w == 0:
            return img

        return TF.pad(img, padding=[0, 0, pad_w, pad_h], padding_mode="reflect")

    def ceil_modulo(self, x: int, mod: int):
        return math.ceil(x / mod) * mod
