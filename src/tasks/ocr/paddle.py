from typing import Iterator

import numpy as np
from paddleocr import TextRecognition


class PaddleOcr:
    def __init__(self) -> None:
        self.ocr = TextRecognition()

    def __call__(
        self, images: list[np.ndarray]
    ) -> Iterator[str]:
        for pred in self.ocr.predict(images, batch_size=len(images)):
            print(pred["page_index"], pred["vis_font"], pred["rec_text"])
            yield pred["rec_text"]
