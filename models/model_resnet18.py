"""ImageNet-pretrained ResNet18 for ESC-50.

Fine-tuning: train all backbone layers and the classifier.
All BatchNorm layers use their standard training behaviour.
"""

import torch
from torch import nn
from torchvision.models import ResNet18_Weights, resnet18


class AudioResNet18(nn.Module):
    def __init__(self, in_channels=2, n_classes=50):
        super().__init__()

        self.backbone = resnet18(
            weights=ResNet18_Weights.IMAGENET1K_V1
        )

        # Adapt the pretrained RGB convolution to our input channels.
        old_conv = self.backbone.conv1
        new_conv = nn.Conv2d(
            in_channels,
            old_conv.out_channels,
            kernel_size=old_conv.kernel_size,
            stride=old_conv.stride,
            padding=old_conv.padding,
            bias=False,
        )

        with torch.no_grad():
            mean_weight = old_conv.weight.mean(dim=1, keepdim=True)
            new_conv.weight.copy_(
                mean_weight.repeat(1, in_channels, 1, 1)
                * (3.0 / in_channels)
            )

        self.backbone.conv1 = new_conv

        feature_dim = self.backbone.fc.in_features
        self.backbone.fc = nn.Identity()

        # All pretrained parameters participate in fine-tuning.
        for parameter in self.backbone.parameters():
            parameter.requires_grad = True

        self.classifier = nn.Linear(feature_dim, n_classes)
        self.train(True)

    def forward(self, x):
        features = self.backbone(x)
        return self.classifier(features)


def build_model(in_channels: int, n_classes: int = 50) -> nn.Module:
    return AudioResNet18(
        in_channels=in_channels,
        n_classes=n_classes,
    )


# Used only when running this file directly.
# Fixed 30-epoch comparison; run_model.py supplies the common recipe.
SETTINGS = ["--seeds", "42", "43", "44", "--epochs", "30", "--lr", "1e-4", "--weight-decay", "0.001"]

if __name__ == "__main__":
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from run_model import main

    main([__file__, *SETTINGS])