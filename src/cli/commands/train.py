from dataclasses import dataclass
from pathlib import Path
from typing import Any

import click
from torch.utils.data import DataLoader
import yaml

from src.engine import Trainer
from src.factory import factory


@dataclass
class TrainConfig:
    model: dict[str, Any]
    optimizer: dict[str, Any]
    loss: dict[str, Any]
    use_amp: bool

    callbacks: list[dict[str, Any] | str]

    train_loader: dict[str, Any]
    val_loader: dict[str, Any]

    epochs: int


@click.command(name="train")
@click.option("-c", "--config_file", required=True, type=click.Path(path_type=Path))
def train_cmd(config_file: Path):
    config = TrainConfig(**yaml.safe_load(config_file.read_text()))

    model = factory.build(config.model)

    # Optimizers always must be built with partial
    # to provide runtime model.parameters() value
    optimizer_fn = factory.build(config.optimizer, partial=True)
    loss_fn = factory.build(config.loss)

    train_loader = factory.construct(DataLoader, config.train_loader)
    val_loader = factory.construct(DataLoader, config.val_loader)

    trainer = Trainer(
        model,
        optimizer_fn,
        loss_fn,
        use_amp=config.use_amp,
        callbacks=factory.build(config.callbacks),
    )

    trainer.fit(
        train_loader,
        val_loader=val_loader,
        epochs=config.epochs,
    )
