from typing import Callable, Iterable

from albucore import Any


class KeyedTransform:
    keys: Iterable[str]
    transform: Callable

    def __init__(self, keys: Iterable[str], transform: Callable) -> None:
        self.keys = keys
        self.transform = transform

    def __call__(self, sample: dict[str, Any]) -> dict[str, Any]:
        for key in self.keys:
            sample[key] = self.transform(sample[key])
        return sample
