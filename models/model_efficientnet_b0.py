"""ImageNet EfficientNet-B0 adapted to ESC-50 log-mel + delta inputs.

The input and all training/evaluation rules remain those of run_model.py.
Source: torchvision EfficientNet-B0, IMAGENET1K_V1 weights.
"""
import torch
from torch import nn
from torchvision.models import EfficientNet_B0_Weights, efficientnet_b0


class AudioEfficientNetB0(nn.Module):
    def __init__(self, in_channels=2, n_classes=50, pretrained=True):
        super().__init__()
        if in_channels != 2:
            raise ValueError('Expected two channels: log-mel and delta.')
        weights = EfficientNet_B0_Weights.IMAGENET1K_V1 if pretrained else None
        self.network = efficientnet_b0(weights=weights)
        original = self.network.features[0][0]
        stem = nn.Conv2d(in_channels, original.out_channels, original.kernel_size,
                         original.stride, original.padding, bias=False)
        if pretrained:
            # Initially identical to repeating log-mel across RGB channels.
            # Delta starts with zero contribution but its weights remain trainable.
            with torch.no_grad():
                stem.weight[:, 0:1].copy_(original.weight.sum(dim=1, keepdim=True))
                stem.weight[:, 1:2].zero_()
        self.network.features[0][0] = stem
        self.network.classifier[1] = nn.Linear(self.network.classifier[1].in_features, n_classes)

    def forward(self, x):
        if x.ndim != 4 or x.shape[1] != 2:
            raise ValueError('Expected (batch, 2, frequency, time).')
        # Native spectrogram size: no RGB transforms, crops or ImageNet normalization.
        return self.network(x)


def build_model(in_channels: int, n_classes: int = 50) -> nn.Module:
    return AudioEfficientNetB0(in_channels, n_classes)


# Fixed final budget, recorded in local_tools/experiment_plan.json before development.
SETTINGS = ['--seeds', '42', '43', '44', '--epochs', '60', '--lr', '0.0003', '--weight-decay', '0.001', '--', '--save-models']

if __name__ == '__main__':
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from run_model import main
    main([__file__, *SETTINGS])
