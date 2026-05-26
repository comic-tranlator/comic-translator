from typing import Any

import albumentations as A
A.SmallestMaxSize

class Albumentations:
    def __init__(
        self,
        image_keys: list[str],
        transforms: list[A.BasicTransform | A.BaseCompose],
        mask_keys: list[str] | None = None,
    ) -> None:
        self.main_image_key = image_keys[0]

        self.additional_image_keys = set(image_keys[1:])
        self.mask_keys = set(mask_keys or [])

        additional_targets = {}

        for key in self.additional_image_keys:
            additional_targets[f"image_{key}"] = "image"

        for key in self.mask_keys:
            additional_targets[f"mask_{key}"] = "mask"

        self.transform = A.Compose(transforms, additional_targets=additional_targets)

    def __call__(self, sample: dict[str, Any]) -> dict[str, Any]:
        transform_kwargs = {}
        transform_kwargs["image"] = sample[self.main_image_key]

        for key in self.additional_image_keys:
            img = sample[key]
            transform_kwargs[f"image_{key}"] = img

        for key in self.mask_keys:
            img = sample[key]
            transform_kwargs[f"mask_{key}"] = img

        result = self.transform(**transform_kwargs)

        sample[self.main_image_key] = result["image"]

        for key in self.additional_image_keys:
            img_key = f"image_{key}"
            sample[key] = result[img_key]

        for key in self.mask_keys:
            mask_key = f"mask_{key}"
            sample[key] = result[mask_key]

        return sample
