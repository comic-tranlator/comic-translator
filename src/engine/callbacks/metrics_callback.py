from ..schema import BatchResult, TrainContext
from .callback import Callback


class MetricsCallback(Callback):
    def on_epoch_start(self, epoch: int, ctx: TrainContext) -> None:
        self.train_loss = 0.0
        self.train_steps = 0
        self.val_loss = 0.0
        self.val_steps = 0

        # Reset metrics for the new epoch
        ctx.metrics.clear()

    def on_train_batch_end(
        self,
        batch_id: int,
        batch_size: int,
        batch_result: BatchResult,
        ctx: TrainContext,
    ) -> None:
        self.train_loss += batch_result.loss.item()
        self.train_steps += 1

        ctx.metrics["train_loss"] = self.train_loss / self.train_steps

    def on_val_batch_end(
        self,
        batch_id: int,
        batch_size: int,
        batch_result: BatchResult,
        ctx: TrainContext,
    ) -> None:
        self.val_loss += batch_result.loss.item()
        self.val_steps += 1

        ctx.metrics["val_loss"] = self.val_loss / self.val_steps
