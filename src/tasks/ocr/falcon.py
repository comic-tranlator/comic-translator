from typing import Iterator

import numpy as np
import torch
from transformers import AutoModelForCausalLM


class FalconOcr:
    def __init__(self, device: torch.device, batch_size: int = 6):
        self.batch_size = batch_size
        self.model = AutoModelForCausalLM.from_pretrained(
            "tiiuae/Falcon-OCR",
            trust_remote_code=True,
            dtype=torch.bfloat16,
            device_map=device,
        )

    def __call__(self, images: list[np.ndarray]) -> Iterator[str]:
        for i in range(0, len(images), self.batch_size):
            batch = images[i : i + self.batch_size]
            for pred in self.model.generate(batch, category="text"):  # type: ignore
                yield pred
