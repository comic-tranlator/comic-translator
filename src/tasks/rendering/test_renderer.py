from typing import Iterable

from pictex import Canvas
from PIL import Image

from src.tasks.style_extraction import TextStyle
from src.util.color import rgb_to_hex


class TextRenderer:
    def __init__(self, font_path: str = "assets/fonts/animeace.otf") -> None:
        self.font_path = font_path

    def render_page(
        self,
        inpainted: Image.Image,
        bboxes: Iterable[tuple[int, int, int, int]],
        styles: Iterable[TextStyle],
        texts: Iterable[str],
    ) -> Image.Image:
        result = inpainted.convert("RGBA")
        for style, bbox, text in zip(styles, bboxes, texts):
            if not text:
                continue
            self._render_region(result, bbox, text, style)
        return result

    def _render_region(
        self,
        image: Image.Image,
        bbox: tuple[int, int, int, int],
        text: str,
        style,
    ) -> None:
        x1, y1, x2, y2 = bbox
        w, h = x2 - x1, y2 - y1
        canvas = (
            Canvas()
            .font_family(self.font_path)
            .color(rgb_to_hex(style.font_color))
            .text_align("center")
            .size(w, h)
        )
        text_img, _ = self._fit_text(canvas, text, w, h)
        tw, th = text_img.size
        paste_x = max(0, x1 + (w - tw) // 2)
        paste_y = max(0, y1 + (h - th) // 2)
        image.paste(text_img, (paste_x, paste_y), mask=text_img.split()[3])

    def _fit_text(self, canvas: Canvas, text: str, w: int, h: int) -> tuple:
        low, high = 10, 128  # min_size, max_size

        while low <= high:
            mid = (low + high) // 2
            img = canvas.font_size(mid).render(text).to_pillow().convert("RGBA")
            if img.size[0] <= w and img.size[1] <= h:
                low = mid + 1
            else:
                high = mid - 1

        img = canvas.font_size(high).render(text).to_pillow().convert("RGBA")
        return img, high

    @staticmethod
    def _estimate_font_size(
        w: int,
        h: int,
        text: str,
        min_size: int = 10,
        max_size: int = 128,
    ) -> int:
        import numpy as np

        if not text.strip():
            return min_size
        size_from_height = int(h * 0.8)
        char_width = max(w / max(len(text), 1), 1)
        size_from_width = int(char_width / 0.55)
        return int(np.clip(min(size_from_height, size_from_width), min_size, max_size))
