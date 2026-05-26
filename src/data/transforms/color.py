from typing import Literal

from src.util.color import hex_to_rgb, rgb_to_lab


class ConvertToLab:
    def __init__(self, normalize: bool = True, source: Literal["rgb", "hex"] = "hex") -> None:
        self.normalize = normalize
        self.source = source

    def __call__(self, x: str | tuple[int, int, int]) -> tuple[float, ...]:
        rgb = x if self.source == "rgb" else hex_to_rgb(x)
        L, a, b = rgb_to_lab(rgb)  # type: ignore[arg-type]
        if self.normalize:
            return L / 100, a / 128, b / 128
        return L, a, b
