import cv2
import numpy as np
from skimage.color import deltaE_ciede2000, lab2rgb


def get_lab_contrast(
    rgb1: tuple[float, float, float],
    rgb2: tuple[float, float, float],
) -> float:
    lab1 = rgb_to_lab(rgb1)
    lab2 = rgb_to_lab(rgb2)

    delta_e = deltaE_ciede2000(lab1, lab2)

    return float(delta_e)


def rgb_to_hex(rgb: tuple[float, float, float]) -> str:
    r, g, b = [round(val * 255) for val in rgb]
    return f"#{r:02x}{g:02x}{b:02x}"


def hex_to_rgb(hex_color: str) -> tuple[float, float, float]:
    hex_color = hex_color.lstrip("#")

    return (
        int(hex_color[0:2], 16) / 255,
        int(hex_color[2:4], 16) / 255,
        int(hex_color[4:6], 16) / 255,
    )


def rgb_to_lab(rgb: tuple[float, float, float]) -> tuple[float, float, float]:
    arr = np.array([[rgb]], dtype=np.float32)

    lab = cv2.cvtColor(arr, cv2.COLOR_RGB2Lab)

    return tuple(lab[0][0])


def lab_to_rgb(lab):
    lab = np.asarray(lab, dtype=np.float32)
    lab = lab.reshape(1, 1, 3)

    rgb = lab2rgb(lab)

    return rgb.reshape(3)
