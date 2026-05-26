from __future__ import annotations

from pathlib import Path
from typing import Literal

import torch

from ..schema import TrainContext
from .callback import Callback


class CheckpointCallback(Callback):
    """
    Saves model (or full training) checkpoints during training.

    Metric keys follow the snapshot() convention from MetricsTracker:
    train metrics are plain ("loss", "loss.ce"), validation metrics are
    prefixed with "val_" ("val_loss", "val_loss.ce").

    Args:
        dirpath: Directory to save checkpoints into.
        monitor: Metric key to track for improvement (e.g. "val_loss").
                 When None, saves unconditionally on every scheduled epoch.
        mode: "min" or "max" — whether a lower or higher monitored value
              is considered better.
        top_k: Keep only the k best metric-triggered checkpoints on disk.
               Unlimited when None. Does not affect last.pt.
        every_n_epochs: Save every n epochs. Set to None to disable
                        periodic saving (useful with monitor only).
        save_last: Always write last.pt at the end of each epoch.
        save_mode: "model_only" saves state_dict only.
                   "full" saves epoch, model, optimizer, and best value —
                   enough to resume training exactly.
        filename_pattern: Template string for checkpoint filenames.
                          Supports {epoch}, {monitor}, {value}, and any
                          metric key from tracker.snapshot(), all with
                          optional Python format specs, e.g. {value:.4f}.
                          Defaults to "epoch={epoch}-{monitor}={value:.4f}"
                          when monitor is set, otherwise "epoch={epoch}".
    """

    def __init__(
        self,
        dirpath: str | Path = "checkpoints",
        *,
        monitor: str | None = None,
        mode: Literal["min", "max"] = "min",
        top_k: int | None = None,
        every_n_epochs: int | None = 1,
        save_last: bool = True,
        save_mode: Literal["model_only", "full"] = "full",
        filename_pattern: str | None = None,
    ) -> None:
        self.dirpath = Path(dirpath)
        self.monitor = monitor
        self.mode = mode
        self.save_last = save_last
        self.save_mode = save_mode
        self.filename_pattern = filename_pattern

        # Validate top_k
        if top_k is not None:
            if top_k <= 0:
                raise ValueError(f"`top_k` must be strictly positive, got {top_k}.")
            if monitor is None:
                raise ValueError(
                    "You must specify a `monitor` metric to use `top_k` checkpointing."
                )
        self.top_k = top_k

        # Validate every_n_epochs
        if every_n_epochs is not None and every_n_epochs <= 0:
            raise ValueError(
                f"`every_n_epochs` must be strictly positive, got {every_n_epochs}."
            )
        self.every_n_epochs = every_n_epochs

        self._best_value: float = float("inf") if mode == "min" else float("-inf")
        self._top_k_checkpoints: list[tuple[float, Path]] = []

    def on_epoch_end(self, epoch: int, ctx: TrainContext) -> None:
        if self.monitor is not None:
            self._process_monitored_metric(epoch, ctx)
        elif self._is_scheduled_epoch(epoch):
            self._save_checkpoint(epoch, ctx=ctx)

        if self.save_last:
            self._save_last(epoch, ctx)

    def _process_monitored_metric(self, epoch: int, ctx: TrainContext) -> None:
        if self.monitor is None:
            return

        monitor_value = ctx.metrics.get(self.monitor)
        if monitor_value is None:
            raise KeyError(
                f"Monitored metric '{self.monitor}' not found in tracker snapshot. "
                f"Available metrics: {list(ctx.metrics.keys())}"
            )

        if self._is_improvement(monitor_value):
            self._best_value = monitor_value
            self._save_checkpoint(epoch, ctx)

    def _is_scheduled_epoch(self, epoch: int) -> bool:
        if self.every_n_epochs is None:
            return False
        return epoch % self.every_n_epochs == 0

    def _is_improvement(self, value: float) -> bool:
        return (
            value < self._best_value if self.mode == "min" else value > self._best_value
        )

    def _format_filename(
        self,
        pattern: str,
        epoch: int,
        metrics: dict[str, float],
    ) -> str:
        lookup: dict[str, object] = {
            "epoch": epoch,
            "monitor": self.monitor or "",
            "value": metrics.get(self.monitor or "", float("nan")),
            **metrics,
        }

        try:
            filepath = pattern.format(**lookup)
        except KeyError as e:
            missing_key = e.args[0]
            raise KeyError(
                f"Unknown placeholder '{{{missing_key}}}' in filename_pattern. "
                f"Available keys: {list(lookup.keys())}"
            ) from None

        if not filepath.endswith((".pt", ".ckpt")):
            filepath += ".ckpt"

        return filepath

    def _save_checkpoint(
        self,
        epoch: int,
        ctx: TrainContext,
    ) -> None:
        if self.filename_pattern:
            pattern = self.filename_pattern
        else:
            pattern = (
                "epoch={epoch}"
                if self.monitor is None
                else "epoch={epoch}_{monitor}={value:.4f}"
            )

        filename = self._format_filename(
            pattern,
            epoch,
            ctx.metrics,
        )
        path = self.dirpath / filename
        self.save_state_to_disk(path, epoch, ctx)

        self._manage_top_k(ctx.metrics.get(self.monitor or "", 0), path)

    def _save_last(self, epoch: int, ctx: TrainContext) -> None:
        self.save_state_to_disk(self.dirpath / "last.ckpt", epoch, ctx)

    def save_state_to_disk(self, path: Path, epoch: int, ctx: TrainContext) -> None:
        self.dirpath.mkdir(parents=True, exist_ok=True)
        if self.save_mode == "model_only":
            torch.save(ctx.model.state_dict(), path)
        else:
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": ctx.model.state_dict(),
                    "optimizer_state_dict": ctx.optimizer.state_dict(),
                    "best_value": self._best_value,
                },
                path,
            )

    def _manage_top_k(self, value: float, path: Path) -> None:
        if self.top_k is None:
            return

        self._top_k_checkpoints.append((value, path))
        # sort so index 0 is always the best
        self._top_k_checkpoints.sort(key=lambda t: t[0], reverse=(self.mode == "max"))
        while len(self._top_k_checkpoints) > self.top_k:
            _, worst_path = self._top_k_checkpoints.pop()
            if worst_path.exists():
                worst_path.unlink()
