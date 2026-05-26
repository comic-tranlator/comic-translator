from __future__ import annotations

from typing import Any, Callable

import torch

from .loaders import DatasetLoader
from .utils import DataShape, shape_data


class BasicDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        loader: DatasetLoader,
        input_shape: DataShape,
        target_shape: DataShape,
        transforms: list[Callable] | None = None,
    ) -> None:
        self._loader = loader

        self.input_shape = input_shape
        self.target_shape = target_shape

        self.transforms = transforms or []

    def __len__(self) -> int:
        return len(self._loader)

    def __getitem__(self, index: int) -> tuple[Any, Any]:
        sample = self._loader[index]

        for transform in self.transforms:
            sample = transform(sample)

        inputs = shape_data(sample, self.input_shape)
        targets = shape_data(sample, self.target_shape)

        return inputs, targets
