from .dataset import BasicDataset
from .generators import FontDatasetGenerator
from .loaders import FontDatasetLoader
from .transforms import (
    Albumentations,
    ConvertToLab,
    ForceAspectRatio,
    KeyedTransform,
    Normalize,
    PadToModulo,
    PrepareFontAttributes,
    RandomPatchCrop,
    Resize,
    ScaleToHeight,
    ToGrayscale,
    ToTensor,
    VariableAspectRatio,
)


def wrap_keyed(cls):
    def builder(keys: list[str], **kwargs):
        return KeyedTransform(keys, cls(**kwargs))

    builder.__name__ = cls.__name__
    return builder


keyed_transforms = [
    wrap_keyed(cls)
    for cls in (
        Resize,
        ToGrayscale,
        ToTensor,
        Normalize,
        ForceAspectRatio,
        RandomPatchCrop,
        ScaleToHeight,
        PadToModulo,
        VariableAspectRatio,
        ConvertToLab,
    )
]


REGISTRY = {
    cls.__name__: cls
    for cls in (
        # Datasets
        BasicDataset,
        # Loaders
        FontDatasetLoader,
        # Transforms
        Albumentations,
        PrepareFontAttributes,
        *keyed_transforms,
        # Generators
        FontDatasetGenerator,
    )
}
