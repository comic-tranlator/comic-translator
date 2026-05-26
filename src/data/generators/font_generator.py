import colorsys
from collections import namedtuple
from pathlib import Path
from typing import Iterator

import numpy as np
from pictex import Canvas
from PIL import Image

from src.util.color import get_lab_contrast, rgb_to_hex
from src.util.font import load_font_registry
from src.util.path import ASSETS_DIR

FontDatasetAnnotation = namedtuple(
    "FontDatasetAnnotation",
    [
        "font",
        "typeface",
        "weight",
        "posture",
        "lang",
        "font_color",
        "stroke_color",
        "stroke_thickness",
    ],
)


class FontDatasetGenerator:
    DICTS_DIR = ASSETS_DIR / "dicts"

    def __init__(
        self,
        min_font_size: int,
        max_font_size: int,
        backgrounds_dir: str,
        *,
        min_margin: int = 20,
        max_margin: int = 40,
        rotation_offset: int = 30,
        max_chars_per_line: int = 20,
        max_phrase_length: int = 8,
        min_color_contrast: float = 75,
        stroke_thickness: int = 8,
        expansion: int = 64,
        seed: int = 42,
    ) -> None:
        self.font_registry = load_font_registry()
        self.dicts = self._load_dicts(FontDatasetGenerator.DICTS_DIR)

        self.min_font_size = min_font_size
        self.max_font_size = max_font_size
        self.backgrounds = sorted(Path(backgrounds_dir).iterdir())

        self.rotation_offset = rotation_offset
        self.max_chars_per_line = max_chars_per_line
        self.max_phrase_length = max_phrase_length
        self.min_color_contrast = min_color_contrast
        self.stroke_thickness = stroke_thickness
        self.expansion = expansion

        self.rnd = np.random.default_rng(seed)

    def __iter__(self) -> Iterator[tuple[np.ndarray, FontDatasetAnnotation]]:
        for font_name, font_entry in self.font_registry.items():
            for lang in font_entry.langs:
                for _ in range(self.expansion):
                    phrase = self._generate_random_phrase(
                        lang, 1, self.max_phrase_length
                    )

                    # Generate attributes
                    (
                        font_size,
                        font_color,
                        stroke_color,
                        stroke_thickness,
                        rotation,
                        background,
                    ) = self._generate_attributes()

                    text = self._render_text_layer(
                        phrase=phrase,
                        font_path=font_entry.path,
                        font_size=font_size,
                        font_color=font_color,
                        stroke_color=stroke_color,
                        stroke_thickness=stroke_thickness,
                        rotation=rotation,
                    )

                    image = self._apply_background_cover(background, text)

                    annotation = FontDatasetAnnotation(
                        font_name,
                        font_entry.typeface,
                        font_entry.weight,
                        font_entry.posture,
                        lang,
                        rgb_to_hex(font_color),
                        rgb_to_hex(stroke_color),
                        stroke_thickness,
                    )

                    yield image, annotation

    def _render_text_layer(
        self,
        phrase: str,
        font_path: Path,
        font_size: int,
        font_color: tuple,
        stroke_color: tuple,
        stroke_thickness: int,
        rotation: int,
    ) -> Image.Image:
        canvas = (
            Canvas()
            .font_family(font_path)
            .font_size(font_size)
            .color(rgb_to_hex(font_color))
            .text_stroke(stroke_thickness, rgb_to_hex(stroke_color), "outline")
            .text_align("center")
            .overflow("hidden")
        )
        text = canvas.render(phrase).to_pillow().convert("RGBA")
        return text.rotate(rotation, expand=True, resample=Image.Resampling.BICUBIC)

    def _apply_background_cover(
        self, background: Image.Image, text_layer: Image.Image
    ) -> np.ndarray:
        target_w, target_h = text_layer.size
        bg_w, bg_h = background.size

        target_aspect = target_w / target_h
        bg_aspect = bg_w / bg_h

        # Determine center-cropping box parameters based on aspect ratio differentials
        if bg_aspect > target_aspect:
            new_w = int(bg_h * target_aspect)
            left = (bg_w - new_w) // 2
            top = 0
            right = left + new_w
            bottom = bg_h
        else:
            new_h = int(bg_w / target_aspect)
            left = 0
            top = (bg_h - new_h) // 2
            right = bg_w
            bottom = top + new_h

        # Crop background, downsample smoothly, and blend the transparent text matrix over top
        cropped_bg = background.crop((left, top, right, bottom))
        resized_bg = cropped_bg.resize((target_w, target_h), Image.Resampling.LANCZOS)
        resized_bg.alpha_composite(text_layer)

        return np.array(resized_bg.convert("RGB"))

    def _generate_attributes(
        self,
    ) -> tuple[int, tuple, tuple, int, int, Image.Image]:
        while True:
            font_size = self.rnd.integers(self.min_font_size, self.max_font_size).item()
            font_color = self._generate_rgb()

            stroke_thickness = self.rnd.integers(0, self.stroke_thickness).item()
            stroke_color = self._generate_rgb()

            background = self._pick_random_background()
            background_mean = np.array(background).mean(axis=(0, 1))

            rotation = self.rnd.integers(
                -self.rotation_offset, self.rotation_offset
            ).item()

            if self.is_valid_colors(
                font_color, background_mean, stroke_color, stroke_thickness
            ):
                return (
                    font_size,
                    font_color,
                    stroke_color,
                    stroke_thickness,
                    rotation,
                    background,
                )

    def is_valid_colors(
        self,
        font_color: tuple[float, float, float],
        background_color: tuple[float, float, float],
        stroke_color: tuple[float, float, float],
        stroke_thickness: int,
    ) -> bool:
        if stroke_thickness > 0:
            contrasts_bg = (
                get_lab_contrast(stroke_color, background_color)
                > self.min_color_contrast
            )
            contrasts_stroke = (
                get_lab_contrast(font_color, stroke_color) > self.min_color_contrast
            )
            return contrasts_bg and contrasts_stroke
        else:
            contrasts_bg = (
                get_lab_contrast(font_color, background_color) > self.min_color_contrast
            )
            return contrasts_bg

    def _load_dicts(self, dicts_dir: Path) -> dict[str, list[str]]:
        dicts = {}
        for filepath in Path(dicts_dir).glob("*.txt"):
            lang = filepath.stem
            with open(filepath, "r", encoding="utf-8") as f:
                lines = [line.strip() for line in f if line.strip()]
                if lines:
                    dicts[lang] = lines
        return dicts

    def _generate_rgb(self) -> tuple[float, float, float]:
        hue, sat, val = self.rnd.uniform(0, 1, 3)
        return colorsys.hsv_to_rgb(hue, sat, val)

    def _generate_random_phrase(
        self, lang: str, min_length: int, max_length: int
    ) -> str:
        length = self.rnd.integers(min_length, max_length)

        words = self.dicts[lang]
        sentence = []
        for _ in range(length):
            word_idx = self.rnd.integers(0, len(words) - 1)
            sentence.append(words[word_idx])

        full_text = " ".join(sentence)

        wrapped_chars = []
        current_line_len = 0

        for char in full_text:
            if current_line_len >= self.max_chars_per_line:
                if char == " ":
                    wrapped_chars.append("\n")
                    current_line_len = 0
                    continue
                else:
                    wrapped_chars.append("\n")
                    current_line_len = 0

            wrapped_chars.append(char)
            current_line_len += 1

        return "".join(wrapped_chars)

    def _pick_random_background(self) -> Image.Image:
        bg_idx = self.rnd.integers(0, len(self.backgrounds) - 1)

        path = self.backgrounds[bg_idx]
        with Image.open(path) as background:
            return background.convert("RGBA")
