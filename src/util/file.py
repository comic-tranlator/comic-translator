from pathlib import Path

import numpy as np
from pdf2image import convert_from_bytes
from PIL import Image


def load_file(filepath: Path) -> list[np.ndarray]:
    if not filepath.exists():
        raise FileNotFoundError(f"File {filepath} isn't found")

    suffix = filepath.suffix.lower()

    if suffix in (".jpg", ".jpeg", ".png"):
        with Image.open(filepath) as image:
            return [np.array(image.convert("RGB"))]

    if suffix == ".pdf":
        with filepath.open("rb") as file:
            pdf_bytes = file.read()

        images = [
            np.array(image.convert("RGB"))
            for image in convert_from_bytes(
                pdf_bytes,
                dpi=200,
                size=1920,
                thread_count=4,
            )
        ]

        return images

    raise RuntimeError(f"File {filepath} is not supported")
