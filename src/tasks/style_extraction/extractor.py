from dataclasses import dataclass
from typing import Any, Callable, Iterator

import albumentations as A
import numpy as np
import torch

from src.modeling import StyleNet
from src.modeling.modules import StyleNetPrediction
from src.util.color import lab_to_rgb


@dataclass
class TextStyle:
    font_color: tuple[float, float, float]
    stroke_color: tuple[float, float, float]


class StyleExtractor:
    def __init__(
        self,
        model: StyleNet,
        checkpoint: str,
        transforms: list[Callable] | None = None,
        max_stroke_thickness: int = 8,
        device: torch.device = torch.device("cuda"),
    ) -> None:
        self.device = device
        self.model = model
        self.model.load_state_dict(torch.load(checkpoint)["model_state_dict"])
        self.model.eval()
        self.model.to(self.device)

        self.transforms = transforms or []
        self.max_stroke_thickness = max_stroke_thickness

    def __call__(self, images: list[np.ndarray]) -> Iterator[TextStyle]:
        transformed = [self.apply_transform(image) for image in images]
        if not transformed:
            return

        x = torch.stack(transformed).to(self.device)

        with torch.inference_mode():
            prediction = self.model(x)
        for postprocessed in self.postprocess(prediction):
            yield postprocessed

    def apply_transform(self, image: np.ndarray) -> Any:
        sample = {"image": image}
        for transform in self.transforms:
            sample = transform(sample)
        return sample["image"]

    def postprocess(self, prediction: StyleNetPrediction) -> Iterator[TextStyle]:
        font_colors = prediction.font_color.detach().cpu().numpy()
        stroke_colors = prediction.stroke_color.detach().cpu().numpy()

        for font_lab_norm, stroke_lab_norm in zip(font_colors, stroke_colors):
            font_rgb = np.clip(lab_to_rgb(self.denormalize_lab(font_lab_norm)), 0.0, 1.0)
            stroke_rgb = np.clip(lab_to_rgb(self.denormalize_lab(stroke_lab_norm)), 0.0, 1.0)
            yield TextStyle(font_color=tuple(font_rgb), stroke_color=tuple(stroke_rgb))

    def denormalize_lab(self, lab: np.ndarray) -> np.ndarray:
        L = lab[0] * 100.0        # [0, 1]   → [0, 100]
        a = lab[1] * 128.0        # [-1, 1]  → [-128, 128]
        b = lab[2] * 128.0        # [-1, 1]  → [-128, 128]
        return np.array([L, a, b])
