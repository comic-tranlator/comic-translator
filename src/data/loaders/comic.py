from pathlib import Path
from typing import Any, TypedDict

import numpy as np
from PIL import Image

from src.data.loaders.base import DatasetLoader


class ComicStyleSample(TypedDict):
    image: Any
    font: str


class ComicStyleLoader(DatasetLoader):
    folder: Path
    image_paths: list[Path]

    def __init__(self, root: str | Path, train: bool = False) -> None:
        super().__init__()

        self.folder = Path(root) / ("train" if train else "test")

        self.image_paths = sorted((self.folder / "images").iterdir())

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx: int) -> ComicStyleSample:
        image_path = self.image_paths[idx]
        gt_path = self.folder / "gt" / f"{image_path.stem}.txt"

        with Image.open(self.image_paths[idx]) as img:
            image = np.array(img)

        (font,) = gt_path.read_text().split(",")

        return {"image": image, "font": font}
