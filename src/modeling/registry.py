from .losses import FontNetLoss, StyleNetLoss
from .models import DeepFont, DeepFontAutoencoder, FontNet, StyleNet
from .modules import (
    FpnFusion,
    StyleNetHead,
    resnet18,
    resnet34,
    resnet50,
    resnet101,
    resnet152,
)

REGISTRY = {
    cls.__name__: cls
    for cls in (
        # Models
        FontNet,
        DeepFont,
        DeepFontAutoencoder,
        StyleNet,
        # Modules
        resnet18,
        resnet34,
        resnet50,
        resnet101,
        resnet152,
        FpnFusion,
        StyleNetHead,
        # Losses
        FontNetLoss,
        StyleNetLoss,
    )
}
