from pathlib import Path

import torch
import torch.nn as nn


def load_from_checkpoint[T: nn.Module](model: T, path: str | Path) -> T:
    checkpoint = torch.load(path)
    model.load_state_dict(checkpoint["model_state_dict"])

    model.eval()

    return model
