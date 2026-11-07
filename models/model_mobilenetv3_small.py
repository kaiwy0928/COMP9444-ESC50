"""MobileNetV3-Small model for ESC-50 sound classification.

Input: (B, 2, 128, 216), log-mel + delta.
Output: (B, 50) logits.
Weights: ImageNet-1K pretrained.
"""

import torch
from torch import nn
from torchvision.models import mobilenet_v3_small, MobileNet_V3_Small_Weights


def build_model(in_channels: int, n_classes: int = 50) -> nn.Module:
    # Load MobileNetV3-Small with ImageNet pretrained weights.
    model = mobilenet_v3_small(
        weights=MobileNet_V3_Small_Weights.IMAGENET1K_V1
    )

    # Adapt the first convolution while preserving pretrained weights.
    first_conv = model.features[0][0]
    new_conv = nn.Conv2d(
        in_channels,
        first_conv.out_channels,
        kernel_size=first_conv.kernel_size,
        stride=first_conv.stride,
        padding=first_conv.padding,
        bias=False,
    )

    with torch.no_grad():
        pretrained_weight = first_conv.weight.mean(dim=1, keepdim=True)
        new_conv.weight.copy_(pretrained_weight.repeat(1, in_channels, 1, 1))

    model.features[0][0] = new_conv

    # Adapt the classifier to the number of sound classes.
    model.classifier[-1] = nn.Linear(
        model.classifier[-1].in_features,
        n_classes,
    )

    return model


# Smoke test before a full run:
# SETTINGS = ["--folds", "1", "--epochs", "3"]
SETTINGS = ["--seeds", "42", "43", "44"]

if __name__ == "__main__":
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from run_model import main
    main([__file__, *SETTINGS])


