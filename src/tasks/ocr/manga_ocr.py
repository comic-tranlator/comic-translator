import numpy as np
import torch
from PIL import Image
from transformers import AutoImageProcessor, AutoTokenizer, VisionEncoderDecoderModel


class MangaOcr:
    def __init__(
        self,
        device: torch.device,
        model_name: str = "kha-white/manga-ocr-base",
    ):
        self.model_name = model_name
        self.device = device

        # Load components
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)

        self.image_processor = AutoImageProcessor.from_pretrained(model_name)

        self.model = VisionEncoderDecoderModel.from_pretrained(
            model_name,
            dtype="auto",
        ).to(self.device)

        self.model.eval()

    def __call__(self, images: list[np.ndarray]) -> list[str]:
        images_pil = [Image.fromarray(image) for image in images]
        results = []
        with torch.inference_mode():
            for image in images_pil:
                pixel_values = self.image_processor(
                    image,
                    return_tensors="pt",
                ).pixel_values.to(self.device)

                generated_ids = self.model.generate(  # type: ignore
                    pixel_values,
                    max_new_tokens=128,
                )

                text = self.tokenizer.batch_decode(
                    generated_ids,
                    skip_special_tokens=True,
                )[0]

                results.append(text)
        return results
