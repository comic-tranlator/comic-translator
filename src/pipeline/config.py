from dataclasses import dataclass
from typing import Any


@dataclass
class PipelineConfig:
    style_extractor: dict[str, Any]

    device: str
