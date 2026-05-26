from dataclasses import dataclass, field
from typing import Any

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader


@dataclass
class TrainConfig:
    model: dict[str, Any]
    optimizer: dict[str, Any]
    loss: dict[str, Any]

    train_metrics: list[dict[str, Any] | str]
    val_metrics: list[dict[str, Any] | str]

    callbacks: list[dict[str, Any] | str]

    train_loader: dict[str, Any]
    val_loader: dict[str, Any] | None

    epochs: int


@dataclass(kw_only=True, frozen=True)
class BatchResult:
    inputs: Any
    targets: Any
    outputs: Any
    loss: Any


@dataclass(kw_only=True, frozen=True)
class TrainContext:
    model: nn.Module
    optimizer: optim.Optimizer

    epochs: int
    train_loader: DataLoader
    val_loader: DataLoader | None = None

    device: torch.device
    metrics: dict[str, float] = field(default_factory=dict)
