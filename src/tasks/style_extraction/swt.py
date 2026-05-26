from typing import Iterator, Literal

import cv2
import numpy as np

from src.util.swt import SWTLocalizer


class SwtSegmenter:
    def __call__(
        self, images: list[np.ndarray], text_modes: list[Literal["lb_df", "db_lf"]]
    ) -> Iterator[np.ndarray]:
        assert len(images) == len(text_modes)

        swtl = SWTLocalizer(images=images)

        for swt_image, text_mode in zip(swtl.swtimages, text_modes):
            mask = swt_image.transformImage(
                text_mode=text_mode,
                maximum_angle_deviation=np.pi / 2,
                gaussian_blurr_kernel=(9, 9),
                minimum_stroke_width=3,
                maximum_stroke_width=20,
                include_edges_in_swt=False,
                display=False,
            )

            mask = (mask > 0).astype(np.uint8)

            clean_mask = cv2.morphologyEx(
                mask, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8)
            )

            yield clean_mask
