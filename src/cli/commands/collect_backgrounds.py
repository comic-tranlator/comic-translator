from pathlib import Path

import click
import cv2
import numpy as np
import torch
from pdf2image import convert_from_path

from src.tasks.detection import PaddleDetector
from src.tasks.inpainting import LamaInpainter
from src.util import build_mask


def merge_boxes(
    polys: list, distance_threshold: int = 20
) -> list[tuple[int, int, int, int]]:
    rects = [cv2.boundingRect(p.astype(np.int32)) for p in polys]

    if not rects:
        return []

    max_x = max(r[0] + r[2] for r in rects) + distance_threshold
    max_y = max(r[1] + r[3] for r in rects) + distance_threshold

    mask = np.zeros((max_y, max_x), dtype=np.uint8)
    for x, y, w, h in rects:
        cv2.rectangle(mask, (x, y), (x + w, y + h), 255, -1)

    kernel = cv2.getStructuringElement(
        cv2.MORPH_RECT, (distance_threshold, distance_threshold)
    )
    dilated_mask = cv2.dilate(mask, kernel, iterations=1)

    contours, _ = cv2.findContours(
        dilated_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )

    merged_rects = []
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)

        pad = distance_threshold // 2
        x_true = max(0, x + pad)
        y_true = max(0, y + pad)
        w_true = max(1, w - distance_threshold)
        h_true = max(1, h - distance_threshold)

        merged_rects.append((x_true, y_true, w_true, h_true))

    return merged_rects


@click.command(name="collect_backgrounds")
@click.option("-f", "--pdf_file", required=True, type=click.Path(path_type=Path))
@click.option("-o", "--output_dir", required=True, type=click.Path(path_type=Path))
def collect_backgrounds_cmd(pdf_file: Path, output_dir: Path):
    output_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    click.echo("Loading PDF File...")
    images = [
        np.array(image)
        for image in convert_from_path(
            pdf_file,
            dpi=200,
            size=1920,
            thread_count=4,
        )
    ]
    click.echo(f"Loaded {len(images)} pages")

    detector = PaddleDetector()
    inpainter = LamaInpainter(device)

    masks = []
    images_polys = []
    for polys in detector(images):
        idx = len(masks)
        h, w = images[idx].shape[:2]
        mask = build_mask((h, w), polys)
        masks.append(mask)
        images_polys.append(polys)

    box_idx = 0
    distance_threshold = 20
    min_size = 150

    for inpainted, polys in zip(inpainter(images, masks), images_polys):
        grouped_rects = merge_boxes(polys, distance_threshold=distance_threshold)

        for x, y, w, h in grouped_rects:
            img_h, img_w = inpainted.shape[:2]
            x1, y1 = max(0, x), max(0, y)
            x2, y2 = min(img_w, x + w), min(img_h, y + h)

            cropped_box = inpainted[y1:y2, x1:x2]

            final_w = x2 - x1
            final_h = y2 - y1
            if final_w < min_size or final_h < min_size:
                continue

            if cropped_box.size == 0:
                continue

            cropped_bgr = cv2.cvtColor(cropped_box, cv2.COLOR_RGB2BGR)

            output_path = output_dir.absolute() / f"box_{box_idx:05d}.png"
            cv2.imwrite(str(output_path), cropped_bgr)

            box_idx += 1

    click.echo(
        f"Successfully saved {box_idx} cropped images to {output_dir.absolute()}"
    )


if __name__ == "__main__":
    collect_backgrounds_cmd()
