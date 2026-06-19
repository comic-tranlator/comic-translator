import os
from typing import Iterator

os.environ["TORCHDYNAMO_RECOMPILE_LIMIT"] = "32"

import numpy as np
import torch
from PIL import Image
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
        if not images:
            return

        with torch.inference_mode():
            for start in range(0, len(images), self.batch_size):
                batch = images[start : start + self.batch_size]
                batch_pil = [Image.fromarray(image) for image in batch]
                for pred in self.model.generate(batch_pil, category="text"):  # type: ignore
                    yield pred

                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
