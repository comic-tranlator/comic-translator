from .collect_backgrounds import collect_backgrounds_cmd
from .generate_dataset import generate_dataset_cmd
from .train import train_cmd
from .translate import translate_cmd

__all__ = [
    "collect_backgrounds_cmd",
    "generate_dataset_cmd",
    "train_cmd",
    "translate_cmd",
]
