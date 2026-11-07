"""model_baseline: the baseline CRNN, built from basic PyTorch layers (no torchvision /
pretrained models). 69.05% without augmentation; with the common recipe (run_model.py) 86.85 +- 0.58% (final protocol).

    input  (B, C, 128 mel, 216 frames); C = 2 for the baseline (log-mel + delta)
    4 conv blocks, each Conv-BN-ReLU-Conv-BN-ReLU-MaxPool; pooling shrinks frequency 16x and time 4x
        -> (B, 128 channels, 8 mel, 54 frames)
    flatten channels x mel per frame -> (B, 54, 1024) -> Linear 1024 -> 256
    BiLSTM (hidden 128 per direction) -> (B, 54, 256)
    mean over time -> dropout -> Linear 256 -> 50 classes

Every model file in models/ defines build_model(in_channels, n_classes); src/train.py calls it once per fold.
baseline/3_train_baseline.py trains this file without augmentation (the 69.05% reference).
"""
from torch import nn


class ConvBlock(nn.Module):
    """One building block of the CNN; CRNN stacks four of them."""

    def __init__(self, c_in: int, c_out: int, pool: tuple[int, int], dropout: float = 0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(c_in, c_out, 3, padding=1, bias=False), nn.BatchNorm2d(c_out), nn.ReLU(inplace=True),
            nn.Conv2d(c_out, c_out, 3, padding=1, bias=False), nn.BatchNorm2d(c_out), nn.ReLU(inplace=True),
            nn.MaxPool2d(pool), nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)


class CRNN(nn.Module):
    def __init__(self, n_classes: int = 50, n_mels: int = 128, in_channels: int = 1, channels=(32, 64, 128, 128),
                 pools=((2, 2), (2, 2), (2, 1), (2, 1)), rnn_hidden: int = 128, rnn_layers: int = 1,
                 dropout: float = 0.5):
        super().__init__()
        blocks, c_in, f = [], in_channels, n_mels
        for c, p in zip(channels, pools):
            blocks.append(ConvBlock(c_in, c, p))
            c_in, f = c, f // p[0]
        self.cnn = nn.Sequential(*blocks)
        self.proj = nn.Sequential(nn.Linear(c_in * f, 2 * rnn_hidden), nn.ReLU(inplace=True))
        self.rnn = nn.LSTM(2 * rnn_hidden, rnn_hidden, num_layers=rnn_layers, batch_first=True, bidirectional=True)
        self.head = nn.Sequential(nn.Dropout(dropout), nn.Linear(2 * rnn_hidden, n_classes))

    def forward(self, x):                       # x: (B, C, n_mels, T)
        z = self.cnn(x)                         # (B, C', F', T')
        b, c, f, t = z.shape
        z = z.permute(0, 3, 1, 2).reshape(b, t, c * f)  # one feature vector per time step
        z, _ = self.rnn(self.proj(z))           # (B, T', 2H)
        return self.head(z.mean(dim=1))         # average over time -> logits


def build_model(in_channels: int, n_classes: int = 50) -> nn.Module:
    return CRNN(n_classes=n_classes, in_channels=in_channels)


# Click Run on this file to train and evaluate it with the common recipe (same as run_model.py).
# Smoke test first: SETTINGS = ["--folds", "1", "--epochs", "3"]
SETTINGS = ["--seeds", "42", "43", "44"]

if __name__ == "__main__":
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from run_model import main
    main([__file__, *SETTINGS])
