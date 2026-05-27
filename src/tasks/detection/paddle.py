import os
from typing import Iterator

os.environ["PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK"] = "True"

import numpy as np
from paddleocr import TextDetection


class PaddleDetector:
    limit_side_len: int

    def __init__(
        self,
        limit_side_len: int = 1440,
        model_name: str = "PP-OCRv5_mobile_det",
    ) -> None:
        self.limit_side_len = limit_side_len

        self.model = TextDetection(
            model_name=model_name,
            limit_side_len=limit_side_len,
            limit_type="max",
            enable_mkldnn=False,
        )

    def __call__(
        self, images: list[np.ndarray]
    ) -> Iterator[list[np.ndarray]]:
        preds = self.model.predict(images)
        for pred in preds:
            yield pred["dt_polys"]
