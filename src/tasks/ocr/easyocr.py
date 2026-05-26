from typing import Iterator

import easyocr
import numpy as np


class EasyOcr:
    def __init__(self) -> None:
        self.reader = easyocr.Reader(["ch_sim", "en", "ja"])

    def __call__(self, images: list[np.ndarray]) -> Iterator[str]:
        for pred in self.reader.readtext_batched(images, batch_size=len(images)):
            yield pred
