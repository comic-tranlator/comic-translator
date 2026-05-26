from typing import Iterator

import cv2
import numpy as np
import torch
import torchvision.transforms.functional as TF
import torchvision.transforms.v2 as T

from src.data.transforms import PadToModulo
from src.util.hub import download_model


class LamaInpainter:
    WEIGHTS_URL = "https://github.com/enesmsahin/simple-lama-inpainting/releases/download/v0.1.0/big-lama.pt"
    TRANSFORMS = T.Compose(
        [T.ToImage(), T.ToDtype(torch.float32, scale=True), PadToModulo(8)]
    )
    MASK_TRANSFORMS = T.Compose(
        [
            T.ToImage(),
            T.ToDtype(torch.float32, scale=True),
            PadToModulo(8),
            T.Lambda(lambda m: (m > 0).float()),
        ]
    )

    def __init__(self, device: torch.device, batch_size: int = 4) -> None:
        self.device = device
        self.batch_size = batch_size

        weights_path = download_model(LamaInpainter.WEIGHTS_URL)

        self.model = torch.jit.load(weights_path)
        self.model.eval()
        self.model.to(self.device)

    def __call__(
        self, images: list[np.ndarray], masks: list[np.ndarray]
    ) -> Iterator[np.ndarray]:
        assert len(images) == len(masks), "Amount of images and masks must be equal"

        yield from self.process_batches(images, masks)

    def process_batches(
        self, images: list[np.ndarray], masks: list[np.ndarray]
    ) -> Iterator[np.ndarray]:
        num_images = len(images)

        with torch.inference_mode():
            for i in range(0, num_images, self.batch_size):
                raw_batch_imgs = images[i : i + self.batch_size]
                raw_batch_msks = masks[i : i + self.batch_size]
                batch_imgs, batch_msks, batch_sizes = self.preprocess_batch(
                    raw_batch_imgs, raw_batch_msks
                )

                image_tensor = torch.stack(
                    [
                        LamaInpainter.TRANSFORMS(img).to(self.device)
                        for img in batch_imgs
                    ]
                )
                mask_tensor = torch.stack(
                    [
                        LamaInpainter.MASK_TRANSFORMS(msk).to(self.device)
                        for msk in batch_msks
                    ]
                )

                predictions = self.model(image_tensor, mask_tensor)

                for pred, original_size in zip(predictions, batch_sizes):
                    yield self.postprocess(pred, original_size)

                if self.device.type == "cuda":
                    torch.cuda.empty_cache()

    def preprocess_batch(
        self, images: list[np.ndarray], masks: list[np.ndarray]
    ) -> tuple[list[np.ndarray], list[np.ndarray], list[tuple[int, int]]]:
        original_h = []
        original_w = []

        for image, mask in zip(images, masks):
            img_h, img_w = image.shape[:2]
            mask_h, mask_w = mask.shape[:2]

            assert img_h == mask_h and img_w == mask_w, (
                "Mask and image must have identical sizes"
            )

            original_h.append(img_h)
            original_w.append(img_w)

        target_h = int(np.median(original_h))
        target_w = int(np.median(original_w))

        collated_images = [
            cv2.resize(image, (target_w, target_h), interpolation=cv2.INTER_CUBIC)
            for image in images
        ]
        collated_masks = [
            cv2.resize(mask, (target_w, target_h), interpolation=cv2.INTER_NEAREST)
            for mask in masks
        ]

        return collated_images, collated_masks, list(zip(original_h, original_w))

    def postprocess(
        self, prediction: torch.Tensor, original_size: tuple[int, int]
    ) -> np.ndarray:
        h, w = original_size
        restored = TF.resize(
            prediction,
            [h, w],
            interpolation=T.InterpolationMode.BILINEAR,
            antialias=True,
        )
        image_np = restored.permute(1, 2, 0).detach().cpu().numpy()
        return np.clip(image_np * 255, 0, 255).astype(np.uint8)
