"""model_multiscale_cnn: our final single model, built from basic PyTorch layers (experiments 65/66 + 70).

The baseline CRNN with three changes:
  1. the first convolution of every block is multi-scale: 3x3 | 7x1 (frequency) | 1x7 (time) in parallel;
  2. SE channel attention (Hu et al. 2018) at the end of every block, before the pooling;
  3. no BiLSTM: per-time-step projection 1024 -> 256, mean over time, classifier.

    input (B, 2, 128 mel, 216 frames)
    4 blocks: MultiScaleConv-BN-ReLU-Conv3x3-BN-ReLU-SE-MaxPool-Dropout   -> (B, 128, 8, 54)
    per time step: flatten channels x bands (1024) -> Linear 256 -> ReLU
    mean over time -> Dropout -> Linear 256 -> 50 classes
Parameters: 0.84M (baseline 1.25M). Common recipe (run_model.py), final protocol: 88.77 +- 0.40% (3 seeds).
"""
import torch
from torch import nn


class MultiScaleConv(nn.Module):
    """3x3 | 7x1 | 1x7 convolutions in parallel, concatenated to c_out channels."""

    def __init__(self, c_in: int, c_out: int):
        super().__init__()
        half, quarter = c_out // 2, c_out // 4
        self.square = nn.Conv2d(c_in, half, 3, padding=1, bias=False)
        self.freq = nn.Conv2d(c_in, quarter, (7, 1), padding=(3, 0), bias=False)
        self.time = nn.Conv2d(c_in, c_out - half - quarter, (1, 7), padding=(0, 3), bias=False)

    def forward(self, x):
        return torch.cat([self.square(x), self.freq(x), self.time(x)], dim=1)


class SE(nn.Module):
    """Squeeze-and-excitation (Hu et al. 2018): one weight per channel from the channel's mean over the map."""

    def __init__(self, channels: int, reduction: int = 8):
        super().__init__()
        self.mlp = nn.Sequential(nn.Linear(channels, channels // reduction), nn.ReLU(inplace=True),
                                 nn.Linear(channels // reduction, channels), nn.Sigmoid())

    def forward(self, x):                       # x: (B, C, F, T)
        w = self.mlp(x.mean(dim=(2, 3)))        # (B, C)
        return x * w[:, :, None, None]


class ConvBlock(nn.Module):
    """One building block of the CNN; CRNN stacks four of them."""

    def __init__(self, c_in: int, c_out: int, pool: tuple[int, int], dropout: float = 0.1):
        super().__init__()
        self.net = nn.Sequential(
            MultiScaleConv(c_in, c_out), nn.BatchNorm2d(c_out), nn.ReLU(inplace=True),        # CHANGED: multi-scale
            nn.Conv2d(c_out, c_out, 3, padding=1, bias=False), nn.BatchNorm2d(c_out), nn.ReLU(inplace=True),
            SE(c_out),                                                                  # CHANGED: SE attention
            nn.MaxPool2d(pool), nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)


class MultiScaleCNN(nn.Module):
    def __init__(self, n_classes: int = 50, n_mels: int = 128, in_channels: int = 1, channels=(32, 64, 128, 128),
                 pools=((2, 2), (2, 2), (2, 1), (2, 1)), hidden: int = 256, dropout: float = 0.5):
        super().__init__()
        blocks, c_in, f = [], in_channels, n_mels
        for c, p in zip(channels, pools):
            blocks.append(ConvBlock(c_in, c, p))
            c_in, f = c, f // p[0]
        self.cnn = nn.Sequential(*blocks)
        self.proj = nn.Sequential(nn.Linear(c_in * f, hidden), nn.ReLU(inplace=True))
        self.head = nn.Sequential(nn.Dropout(dropout), nn.Linear(hidden, n_classes))   # CHANGED: no BiLSTM before it

    def forward(self, x):                       # x: (B, C, n_mels, T)
        z = self.cnn(x)                         # (B, C', F', T')
        b, c, f, t = z.shape
        z = z.permute(0, 3, 1, 2).reshape(b, t, c * f)  # one feature vector per time step
        z = self.proj(z)                        # CHANGED: (B, T', 256) straight from the projection, no BiLSTM
        return self.head(z.mean(dim=1))         # average over time -> logits


def build_model(in_channels: int, n_classes: int = 50) -> nn.Module:
    return MultiScaleCNN(n_classes=n_classes, in_channels=in_channels)


# Click Run on this file to train and evaluate it with the common recipe (same as run_model.py).
# Smoke test first: SETTINGS = ["--folds", "1", "--epochs", "3"]
SETTINGS = ["--seeds", "42", "43", "44"]

if __name__ == "__main__":
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from run_model import main
    main([__file__, *SETTINGS])
