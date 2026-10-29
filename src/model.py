"""Model definitions.

  * CustomCNN    - trained from scratch (random init).
  * build_mobilenet - MobileNetV2 with ImageNet weights and a new 4-class head.

Both expose `.features` (conv feature extractor ending in a spatial map) so Grad-CAM
can hook the same attribute on either architecture.
"""
import sys
from pathlib import Path

import torch
import torch.nn as nn
from torchvision import models

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src import config as C  # noqa: E402


def _block(cin: int, cout: int, pool: bool) -> nn.Sequential:
    layers = [
        nn.Conv2d(cin, cout, 3, padding=1, bias=False), nn.BatchNorm2d(cout), nn.ReLU(inplace=True),
        nn.Conv2d(cout, cout, 3, padding=1, bias=False), nn.BatchNorm2d(cout), nn.ReLU(inplace=True),
    ]
    if pool:
        layers.append(nn.MaxPool2d(2))
    return nn.Sequential(*layers)


class CustomCNN(nn.Module):
    """5 conv blocks (2x [conv-BN-ReLU] each), global average pooling, dropout, linear.

    224 -> 112 -> 56 -> 28 -> 14 spatial; the last block keeps the 14x14 map so
    Grad-CAM heatmaps stay reasonably sharp.
    """

    def __init__(self, num_classes: int = len(C.CLASSES), dropout: float = 0.4):
        super().__init__()
        self.features = nn.Sequential(
            _block(3, 32, True), _block(32, 64, True), _block(64, 128, True),
            _block(128, 256, True), _block(256, 256, False),
        )
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Sequential(nn.Flatten(), nn.Dropout(dropout), nn.Linear(256, num_classes))
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")

    def forward(self, x):
        return self.classifier(self.pool(self.features(x)))


class MobileNetV2Classifier(nn.Module):
    def __init__(self, num_classes: int = len(C.CLASSES), pretrained: bool = True, dropout: float = 0.3):
        super().__init__()
        weights = models.MobileNet_V2_Weights.IMAGENET1K_V1 if pretrained else None
        base = models.mobilenet_v2(weights=weights)
        self.features = base.features                       # ends in a 1280 x 7 x 7 map
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Sequential(nn.Flatten(), nn.Dropout(dropout), nn.Linear(1280, num_classes))

    def forward(self, x):
        return self.classifier(self.pool(self.features(x)))

    # --- transfer-learning stages -------------------------------------------
    def freeze_backbone(self) -> None:
        for p in self.features.parameters():
            p.requires_grad = False

    def unfreeze_last(self, n_blocks: int) -> None:
        """Make the last `n_blocks` of `features` trainable (features has 19 blocks)."""
        for blk in list(self.features)[-n_blocks:]:
            for p in blk.parameters():
                p.requires_grad = True


def build_model(name: str, pretrained: bool = True) -> nn.Module:
    if name == "cnn":
        return CustomCNN()
    if name == "mobilenet":
        return MobileNetV2Classifier(pretrained=pretrained)
    raise ValueError(f"unknown model {name!r}")


def n_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())
