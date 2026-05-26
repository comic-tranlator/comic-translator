from .callbacks import CheckpointCallback, MetricsCallback, ProgressCallback
from .utils import load_from_checkpoint

REGISTRY = {
    cls.__name__: cls
    for cls in (
        # Callbacks
        ProgressCallback,
        CheckpointCallback,
        MetricsCallback,
        # Utils
        load_from_checkpoint,
    )
}
