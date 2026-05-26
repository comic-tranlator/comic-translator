from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import click
import numpy as np
import yaml
from PIL import Image

from src.factory import factory


@dataclass
class GenerationConfig:
    generator: dict[str, Any]
    output: str
    split: float


@click.command(name="generate_dataset")
@click.option("-c", "--config-path", required=True, type=click.Path(path_type=Path))
def generate_dataset_cmd(config_path: Path):
    config = GenerationConfig(**yaml.safe_load(config_path.read_text()))

    generator = factory.build(config.generator)

    click.echo("Generating dataset...")
    with click.progressbar(
        run_dataset_generation(generator, config.split, Path(config.output)),
        label="Progress",
    ) as bar:
        for filename in bar:
            bar.label = f"Saving {filename}"

    click.echo(f"Dataset successfully saved to: {config.output}")


def run_dataset_generation(generator: Iterable, split: float, output: Path):
    train_dir = output / "train"
    test_dir = output / "test"

    train_images_dir = train_dir / "images"
    test_images_dir = test_dir / "images"

    train_gt_dir = train_dir / "gt"
    test_gt_dir = test_dir / "gt"

    train_images_dir.mkdir(parents=True, exist_ok=True)
    test_images_dir.mkdir(parents=True, exist_ok=True)

    train_gt_dir.mkdir(parents=True, exist_ok=True)
    test_gt_dir.mkdir(parents=True, exist_ok=True)

    train_samples = 0
    test_samples = 0

    futures = []
    with ThreadPoolExecutor(max_workers=8) as executor:
        for image, gt in generator:
            total_samples = train_samples + test_samples
            is_train = total_samples == 0 or (train_samples / total_samples) < split

            if is_train:
                image_path = train_images_dir / f"{train_samples:05d}.png"
                gt_path = train_gt_dir / f"{train_samples:05d}.txt"
                train_samples += 1
            else:
                image_path = test_images_dir / f"{test_samples:05d}.png"
                gt_path = test_gt_dir / f"{test_samples:05d}.txt"
                test_samples += 1

            future = executor.submit(save_sample, image, image_path, gt, gt_path)
            futures.append(future)

            yield str(image_path)

        for future in as_completed(futures):
            future.result()


def save_sample(
    image_array: np.ndarray, image_path: Path, gt: tuple, gt_path: Path
) -> None:
    image = Image.fromarray(image_array.astype(np.uint8))
    image.save(image_path)
    gt_path.write_text(",".join(map(str, gt)))
