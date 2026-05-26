from ..schema import BatchResult, TrainContext
from .callback import Callback


class CallbackList:
    def __init__(self, callbacks: list[Callback]) -> None:
        self.callbacks = callbacks

    def on_train_start(self, ctx: TrainContext) -> None:
        for callback in self.callbacks:
            callback.on_train_start(ctx)

    def on_train_end(self, ctx: TrainContext) -> None:
        for callback in self.callbacks:
            callback.on_train_end(ctx)

    def on_epoch_start(self, epoch: int, ctx: TrainContext) -> None:
        for callback in self.callbacks:
            callback.on_epoch_start(epoch, ctx)

    def on_epoch_end(self, epoch: int, ctx: TrainContext) -> None:
        for callback in self.callbacks:
            callback.on_epoch_end(epoch, ctx)

    def on_train_batch_start(
        self, batch_id: int, batch_size: int, ctx: TrainContext
    ) -> None:
        for callback in self.callbacks:
            callback.on_train_batch_start(batch_id, batch_size, ctx)

    def on_train_batch_end(
        self,
        batch_id: int,
        batch_size: int,
        batch_result: BatchResult,
        ctx: TrainContext,
    ) -> None:
        for callback in self.callbacks:
            callback.on_train_batch_end(batch_id, batch_size, batch_result, ctx)

    def on_val_batch_start(
        self, batch_id: int, batch_size: int, ctx: TrainContext
    ) -> None:
        for callback in self.callbacks:
            callback.on_val_batch_start(batch_id, batch_size, ctx)

    def on_val_batch_end(
        self,
        batch_id: int,
        batch_size: int,
        batch_result: BatchResult,
        ctx: TrainContext,
    ) -> None:
        for callback in self.callbacks:
            callback.on_val_batch_end(batch_id, batch_size, batch_result, ctx)

    def on_validation_start(self, ctx: TrainContext) -> None:
        for callback in self.callbacks:
            callback.on_validation_start(ctx)

    def on_validation_end(self, ctx: TrainContext) -> None:
        for callback in self.callbacks:
            callback.on_validation_end(ctx)
