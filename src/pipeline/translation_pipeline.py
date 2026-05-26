from typing import Iterator

import cv2
import numpy as np
import torch
from PIL import Image

from src.factory import factory
from src.tasks.detection import PaddleDetector
from src.tasks.inpainting import LamaInpainter
from src.tasks.ocr import FalconOcr
from src.tasks.rendering import TextRenderer
from src.tasks.translation import TranslationAgent
from src.util import build_mask

from .config import PipelineConfig


class TranslationPipeline:
    def __init__(
        self, source_language: str, target_language: str, config: PipelineConfig
    ) -> None:
        self.device = torch.device(
            config.device if torch.cuda.is_available() else "cpu"
        )
        self.detector = PaddleDetector()
        self.ocr = FalconOcr(self.device)
        self.inpainter = LamaInpainter(self.device)
        self.style_extractor = factory.build(config.style_extractor, device=self.device)
        self.renderer = TextRenderer()
        self.translator = TranslationAgent(
            target_language=target_language,
            source_language=source_language,
            model="mistral-small-latest",
            mistral_api_key="bBrOACSoBiDJuL0iY8KhnluY84DXcBgp",
        )

    def __call__(self, pages: list[np.ndarray]) -> Iterator[Image.Image]:
        masks, pages_polys = self._build_masks(pages)
        for idx, inpainted in enumerate(self.inpainter(pages, masks)):
            page = pages[idx]
            polys = pages_polys[idx]
            bboxes = self._get_text_bboxes(page, polys)
            if not bboxes:
                yield Image.fromarray(inpainted).convert("RGBA")
                continue

            crops = [page[y1:y2, x1:x2] for x1, y1, x2, y2 in bboxes]
            styles = self.style_extractor(crops)

            # OCR → translate → render
            texts = list(self.ocr(crops))
            translated_texts = self.translator(texts)

            inpainted_pil = Image.fromarray(inpainted)
            yield self.renderer.render_page(
                inpainted_pil, bboxes, styles, translated_texts
            )

    def _build_masks(
        self, pages: list[np.ndarray]
    ) -> tuple[list[np.ndarray], list[list[np.ndarray]]]:
        masks, pages_polys = [], []
        for page_polys, page in zip(self.detector(pages), pages):
            polys = self._merge_overlapping_polys(self._normalize_polys(page_polys))
            h, w = page.shape[:2]
            masks.append(build_mask((h, w), polys))
            pages_polys.append(polys)
        return masks, pages_polys

    def _get_text_bboxes(
        self, page: np.ndarray, polys: list[np.ndarray]
    ) -> list[tuple[int, int, int, int]]:
        h_img, w_img = page.shape[:2]
        bboxes = []
        for poly in polys:
            x, y, w, h = cv2.boundingRect(poly.astype(np.int32))
            bboxes.append(
                (
                    max(0, x),
                    max(0, y),
                    min(w_img, x + w),
                    min(h_img, y + h),
                )
            )
        return bboxes

    def _normalize_polys(self, polys: list[np.ndarray]) -> list[np.ndarray]:
        normalized = []
        for poly in polys:
            arr = np.asarray(poly, dtype=np.float32).reshape(-1, 2)
            if arr.size > 0 and arr.shape[0] >= 3:
                normalized.append(arr)
        return normalized

    def _merge_overlapping_polys(self, polys: list[np.ndarray]) -> list[np.ndarray]:
        if not polys:
            return polys
        all_pts = np.vstack(polys)
        max_x = int(np.ceil(all_pts[:, 0].max())) + 2
        max_y = int(np.ceil(all_pts[:, 1].max())) + 2
        canvas = np.zeros((max_y, max_x), dtype=np.uint8)
        for poly in polys:
            cv2.fillPoly(canvas, [poly.reshape(-1, 1, 2).astype(np.int32)], color=255)
        _, labels = cv2.connectedComponents(canvas)
        merged = []
        for label in range(1, labels.max() + 1):
            mask = (labels == label).astype(np.uint8) * 255
            contours, _ = cv2.findContours(
                mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
            )
            for contour in contours:
                pts = contour.reshape(-1, 2).astype(np.float32)
                if pts.shape[0] >= 3:
                    merged.append(pts)
        return merged
