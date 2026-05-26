from __future__ import annotations

from typing import Callable, Iterable

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from src.util import to_device

from .callbacks import Callback, CallbackList
from .schema import BatchResult, TrainContext


class Trainer:
    def __init__(
        self,
        model: nn.Module,
        optimizer_fn: Callable[[Iterable[torch.Tensor]], optim.Optimizer],
        loss_fn: nn.Module,
        *,
        device: torch.device | None = None,
        use_amp: bool = False,
        callbacks: list[Callback] | None = None,
    ) -> None:
        self.device = device or torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        self.use_amp = use_amp

        self.loss_fn = loss_fn
        self.model = model.to(self.device)
        self.optimizer = optimizer_fn(model.parameters())

        self.callbacks = CallbackList(callbacks or [])

        self.scaler = torch.amp.GradScaler(
            "cuda", enabled=self.use_amp and self.device.type == "cuda"
        )

    def fit(
        self,
        train_loader: DataLoader,
        *,
        val_loader: DataLoader | None = None,
        epochs: int = 1,
    ) -> None:
        ctx = TrainContext(
            model=self.model,
            optimizer=self.optimizer,
            epochs=epochs,
            train_loader=train_loader,
            val_loader=val_loader,
            device=self.device,
        )

        self.callbacks.on_train_start(ctx)

        for epoch in range(1, epochs + 1):
            self._run_epoch(epoch, ctx)

        self.callbacks.on_train_end(ctx)

    def _run_epoch(
        self,
        epoch,
        ctx: TrainContext,
    ) -> None:
        self.callbacks.on_epoch_start(epoch, ctx)

        self._train_epoch(ctx)
        self._eval_epoch(ctx)

        self.callbacks.on_epoch_end(epoch, ctx)

    def _train_epoch(self, ctx: TrainContext) -> None:
        ctx.model.train()

        for batch_idx, (inputs, targets) in enumerate(ctx.train_loader):
            batch_size = len(inputs)
            self.callbacks.on_train_batch_start(batch_idx, batch_size, ctx)
            inputs = to_device(inputs, self.device)
            targets = to_device(targets, self.device)

            ctx.optimizer.zero_grad()

            with torch.autocast(device_type=self.device.type, enabled=self.use_amp):
                outputs = ctx.model(inputs)
                loss = self.loss_fn(outputs, targets)

            self.scaler.scale(loss).backward()
            self.scaler.step(ctx.optimizer)
            self.scaler.update()

            batch_result = BatchResult(
                inputs=inputs, targets=targets, outputs=outputs, loss=loss
            )

            self.callbacks.on_train_batch_end(batch_idx, batch_size, batch_result, ctx)

    @torch.no_grad()
    def _eval_epoch(self, ctx: TrainContext) -> None:
        if not ctx.val_loader:
            return

        ctx.model.eval()
        self.callbacks.on_validation_start(ctx)

        for batch_idx, (inputs, targets) in enumerate(ctx.val_loader):
            batch_size = len(inputs)
            self.callbacks.on_val_batch_start(batch_idx, batch_size, ctx)

            inputs = to_device(inputs, self.device)
            targets = to_device(targets, self.device)

            with torch.autocast(device_type=self.device.type, enabled=self.use_amp):
                outputs = ctx.model(inputs)
                loss = self.loss_fn(outputs, targets)

            batch_result = BatchResult(
                inputs=inputs, targets=targets, outputs=outputs, loss=loss
            )

            self.callbacks.on_val_batch_end(batch_idx, batch_size, batch_result, ctx)

        self.callbacks.on_validation_end(ctx)
