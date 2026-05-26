from tqdm.auto import tqdm

from ..schema import BatchResult, TrainContext
from .callback import Callback


class ProgressCallback(Callback):
    def __init__(self, leave: bool = True):
        self.leave = leave
        self.pbar = None

    def on_epoch_start(self, epoch: int, ctx: TrainContext) -> None:
        total_steps = len(ctx.train_loader)
        if ctx.val_loader:
            total_steps += len(ctx.val_loader)

        self.pbar = tqdm(
            total=total_steps, desc=f"Epoch {epoch}/{ctx.epochs}", leave=self.leave, dynamic_ncols=True
        )

    def on_train_batch_end(
        self,
        batch_id: int,
        batch_size: int,
        batch_result: BatchResult,
        ctx: TrainContext,
    ) -> None:
        formatted_metrics = {k: f"{v:.4f}" for k, v in ctx.metrics.items()}
        self.pbar.set_postfix(formatted_metrics)
        self.pbar.update(1)

    def on_val_batch_end(
        self,
        batch_id: int,
        batch_size: int,
        batch_result: BatchResult,
        ctx: TrainContext,
    ) -> None:
        formatted_metrics = {k: f"{v:.4f}" for k, v in ctx.metrics.items()}
        self.pbar.set_postfix(formatted_metrics)
        self.pbar.update(1)

    def on_epoch_end(self, epoch: int, ctx: TrainContext) -> None:
        if self.pbar is not None:
            self.pbar.close()
