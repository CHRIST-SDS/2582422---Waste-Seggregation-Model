"""Model factory: transfer-learning image classifiers (MobileNet/EfficientNet/ResNet)."""
import torch.nn as nn
from torchvision import models

ARCHS = {
    "mobilenet_v3_small": models.mobilenet_v3_small,
    "efficientnet_b0": models.efficientnet_b0,
    "resnet18": models.resnet18,
    "resnet50": models.resnet50,
}


def build_model(arch: str = "resnet18", num_classes: int = 6, pretrained: bool = True) -> nn.Module:
    """Build a classifier with the final head replaced for `num_classes` classes."""
    if arch not in ARCHS:
        raise ValueError(f"Unknown arch {arch!r}. Choose from {sorted(ARCHS)}.")

    weights = "DEFAULT" if pretrained else None
    model = ARCHS[arch](weights=weights)

    if "resnet" in arch:
        in_feats = model.fc.in_features
        model.fc = nn.Linear(in_feats, num_classes)
    elif "efficientnet" in arch:
        in_feats = model.classifier[1].in_features
        model.classifier[1] = nn.Linear(in_feats, num_classes)
    elif "mobilenet" in arch:
        in_feats = model.classifier[-1].in_features
        model.classifier[-1] = nn.Linear(in_feats, num_classes)
    else:
        raise ValueError(f"No head-replacement logic implemented for arch {arch!r}")
    return model
