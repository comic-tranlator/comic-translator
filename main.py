import time
from typing import Iterator, Literal

import cv2
import numpy as np
from sklearn.cluster import KMeans

from src.util.swt import SWTLocalizer


def determine_text_mode(img: np.ndarray) -> Literal["lb_df", "db_lf"]:
    """
    Automatically detects if an image is Light Background/Dark Foreground ('lb_df')
    or Dark Background/Light Foreground ('db_lf').
    Assumes input 'img' is a 2D grayscale array (e.g., after CLAHE).
    """
    h, w = img.shape[:2]

    # 1. Extract border pixels (Top, Bottom, Left, Right edges)
    top_edge = img[0, :]
    bottom_edge = img[-1, :]
    left_edge = img[:, 0]
    right_edge = img[:, -1]

    # Combine all edge pixels into a 1D array to calculate true background mean
    border_pixels = np.concatenate([top_edge, bottom_edge, left_edge, right_edge])
    avg_background_brightness = np.mean(border_pixels)

    # 2. Extract a central patch (where the text actually lives)
    # We grab the middle 50% of the crop
    cy, cx = h // 2, w // 2
    dy, dx = h // 4, w // 4
    center_patch = img[cy - dy : cy + dy, cx - dx : cx + dx]
    avg_center_brightness = np.mean(center_patch)

    # 3. Decision Logic
    # If the background edge is brighter than the center text region,
    # it's a Light Background with Dark Foreground.
    if avg_background_brightness > avg_center_brightness:
        return "lb_df"
    else:
        return "db_lf"


class SwtSegmenter:
    def __call__(
        self, images: list[np.ndarray], text_modes: list[Literal["lb_df", "db_lf"]]
    ) -> Iterator[np.ndarray]:
        assert len(images) == len(text_modes)
        swtl = SWTLocalizer(images=images)
        for image, swt_image, text_mode in zip(images, swtl.swtimages, text_modes):
            mask = swt_image.transformImage(
                text_mode=text_mode,
                maximum_angle_deviation=np.pi / 2,
                gaussian_blurr_kernel=(9, 9),
                minimum_stroke_width=3,
                maximum_stroke_width=20,
                include_edges_in_swt=False,
                display=False,
            )
            mask = (mask > 0).astype(np.uint8)
            clean_mask = cv2.morphologyEx(
                mask, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8)
            )

            result_mask = segment_text_by_color(image, clean_mask)
            yield result_mask


def segment_text_by_color(
    image: np.ndarray,
    swt_mask: np.ndarray,
    n_color_clusters: int = 8,
    second_color_ratio: float = 0.35,
    flood_fill_thresh: int = 8,
) -> np.ndarray:
    """
    Keeps at most 2 colors from the SWT mask region (fill + optional stroke),
    then flood-fills outward with a tight tolerance to recover pixels the SWT
    segmenter missed.

    Args:
        image: Original BGR image.
        swt_mask: Binary mask from SWT (uint8, 0/1).
        n_color_clusters: KMeans clusters used to quantise mask colors.
        second_color_ratio: Keep the 2nd-largest color cluster only if its
            pixel count is at least this fraction of the 1st (0-1).
            e.g. 0.35 means stroke must cover >= 35% of the fill area.
            Set higher (e.g. 0.5) to be stricter, lower to be more permissive.
        flood_fill_thresh: Per-channel BGR tolerance for flood fill expansion.

    Returns:
        Final binary text mask (uint8, 0/255).
    """
    # ------------------------------------------------------------------ #
    # 1. Collect masked pixels and cluster their colors in Lab space       #
    # ------------------------------------------------------------------ #
    mask_coords = np.argwhere(swt_mask > 0)  # (N, 2) – row, col
    if len(mask_coords) == 0:
        return np.zeros(swt_mask.shape, dtype=np.uint8)

    masked_pixels_bgr = image[mask_coords[:, 0], mask_coords[:, 1]]  # (N, 3)
    masked_pixels_lab = (
        cv2.cvtColor(
            masked_pixels_bgr.reshape(1, -1, 3).astype(np.uint8), cv2.COLOR_BGR2Lab
        )
        .reshape(-1, 3)
        .astype(np.float32)
    )

    k = min(n_color_clusters, len(mask_coords))
    kmeans = KMeans(n_clusters=k, n_init=5, random_state=0)
    labels = kmeans.fit_predict(masked_pixels_lab)

    # ------------------------------------------------------------------ #
    # 2. Rank clusters by pixel count                                      #
    # ------------------------------------------------------------------ #
    cluster_sizes = np.bincount(labels, minlength=k)
    rank_by_area = np.argsort(-cluster_sizes)  # largest first

    first_idx = rank_by_area[0]
    second_idx = rank_by_area[1] if k > 1 else None

    # ------------------------------------------------------------------ #
    # 3. Decide which clusters to keep                                     #
    #   - Always keep #1 (text fill).                                      #
    #   - Keep #2 only if its area is >= second_color_ratio of #1         #
    #     (stroke or two-tone text). Drop everything else.                 #
    # ------------------------------------------------------------------ #
    kept_labels = {first_idx}
    if second_idx is not None:
        ratio = cluster_sizes[second_idx] / max(cluster_sizes[first_idx], 1)
        if ratio >= second_color_ratio:
            kept_labels.add(second_idx)

    # ------------------------------------------------------------------ #
    # 4. Build a seed mask from kept clusters                              #
    # ------------------------------------------------------------------ #
    pixel_kept = np.isin(labels, list(kept_labels))

    seed_mask = np.zeros(image.shape[:2], dtype=np.uint8)
    seed_mask[mask_coords[pixel_kept, 0], mask_coords[pixel_kept, 1]] = 255

    # ------------------------------------------------------------------ #
    # 5. Flood-fill from every seed pixel with a tight colour tolerance   #
    #    to recover text pixels the SWT segmenter missed.                 #
    #                                                                      #
    #    cv2.floodFill requires the mask to be exactly (h+2, w+2)         #
    #    relative to the input image — no pre-padding needed.             #
    # ------------------------------------------------------------------ #
    h, w = image.shape[:2]
    img_for_fill = image.copy()  # floodFill may write newVal in-place
    flood_result = np.zeros((h + 2, w + 2), dtype=np.uint8)

    ff_flags = (
        cv2.FLOODFILL_MASK_ONLY  # write into mask, not image
        | cv2.FLOODFILL_FIXED_RANGE  # tolerance relative to seed pixel
        | (255 << 8)  # fill value written into the mask
    )
    lo = (flood_fill_thresh,) * 3
    hi = (flood_fill_thresh,) * 3

    seed_points = np.argwhere(seed_mask > 0)
    # Sub-sample seeds for speed: keep ~2000 representative seeds
    if len(seed_points) > 2000:
        step = max(1, int(np.sqrt(len(seed_points) / 2000)))
        seed_points = seed_points[::step]

    for row, col in seed_points:
        ff_mask = np.zeros((h + 2, w + 2), dtype=np.uint8)
        cv2.floodFill(
            img_for_fill,
            ff_mask,
            seedPoint=(int(col), int(row)),
            newVal=(0, 0, 0),  # unused with FLOODFILL_MASK_ONLY
            loDiff=lo,
            upDiff=hi,
            flags=ff_flags,
        )
        flood_result = cv2.bitwise_or(flood_result, ff_mask)

    # Strip the 1-pixel border OpenCV adds around the mask
    filled_mask = flood_result[1 : h + 1, 1 : w + 1]

    # ------------------------------------------------------------------ #
    # 6. Combine seed mask and flood-filled result                        #
    # ------------------------------------------------------------------ #
    final_mask = cv2.bitwise_or(seed_mask, filled_mask)

    # ------------------------------------------------------------------ #
    # 7. Fill holes — flood-fill from all image borders into the          #
    #    background, then invert: anything unreached is a hole inside     #
    #    the mask and gets filled in.                                      #
    # ------------------------------------------------------------------ #
    # Pad by 1 so the border flood-fill starts cleanly outside mask area
    padded_mask = cv2.copyMakeBorder(
        final_mask, 1, 1, 1, 1, cv2.BORDER_CONSTANT, value=0
    )
    bg_flood = padded_mask.copy()
    cv2.floodFill(bg_flood, None, (0, 0), 255)
    # Unpad and invert: pixels the background fill didn't reach are holes
    bg_flood = bg_flood[1 : h + 1, 1 : w + 1]
    holes = cv2.bitwise_not(bg_flood)
    final_mask = cv2.bitwise_or(final_mask, holes)

    # Light closing to merge small gaps along stroke edges
    final_mask = cv2.morphologyEx(
        final_mask, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8)
    )

    return final_mask


# Image Path
img = cv2.imread("sample2.png")
if img is None:
    raise
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
text_mode = determine_text_mode(img)
segmenter = SwtSegmenter()

start = time.perf_counter()
mask = list(segmenter([img], [text_mode]))[0]
print("Time:", time.perf_counter() - start)
mask = segment_text_by_color(img, mask)
print("Time:", time.perf_counter() - start)
cv2.imwrite("mask.png", mask)
