from ..schema import BatchResult, TrainContext


class Callback:
    def on_train_start(self, ctx: TrainContext) -> None: ...
    def on_train_end(self, ctx: TrainContext) -> None: ...

    def on_epoch_start(self, epoch: int, ctx: TrainContext) -> None: ...
    def on_epoch_end(self, epoch: int, ctx: TrainContext) -> None: ...

    def on_train_batch_start(
        self, batch_id: int, batch_size: int, ctx: TrainContext
    ) -> None: ...
    def on_train_batch_end(
        self,
        batch_id: int,
        batch_size: int,
        batch_result: BatchResult,
        ctx: TrainContext,
    ) -> None: ...

    def on_val_batch_start(
        self, batch_id: int, batch_size: int, ctx: TrainContext
    ) -> None: ...
    def on_val_batch_end(
        self,
        batch_id: int,
        batch_size: int,
        batch_result: BatchResult,
        ctx: TrainContext,
    ) -> None: ...

    def on_validation_start(self, ctx: TrainContext) -> None: ...
    def on_validation_end(self, ctx: TrainContext) -> None: ...
