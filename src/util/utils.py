from typing import Any

import cv2
import numpy as np
import torch


def to_device(data: Any, device: torch.device) -> Any:
    if isinstance(data, torch.Tensor):
        return data.to(device)
    if isinstance(data, dict):
        return {k: to_device(v, device) for k, v in data.items()}
    if isinstance(data, list):
        return [to_device(item, device) for item in data]
    if isinstance(data, tuple):
        return tuple(to_device(item, device) for item in data)
    return data


def deep_merge(base: dict, overrides: dict) -> dict:
    result = base.copy()

    for key, value in overrides.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = deep_merge(result[key], value)
        else:
            result[key] = value

    return result


def build_mask(size: tuple[int, int], polygons: list[np.ndarray]) -> np.ndarray:
    mask = np.zeros(size)

    for poly in polygons:
        pts = np.array(poly, dtype=np.int32).reshape((-1, 1, 2))
        cv2.fillPoly(mask, [pts], 1.00)

    return mask
