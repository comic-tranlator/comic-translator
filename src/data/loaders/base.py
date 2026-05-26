import abc
from typing import Any


class DatasetLoader(abc.ABC):
    @abc.abstractmethod
    def __len__(self) -> int: ...

    @abc.abstractmethod
    def __getitem__(self, index: int) -> dict[str, Any]: ...
