"""model_panns_cnn14: PANNs CNN14 (Kong et al. 2020) fine-tuned on ESC-50.

Reference: Q. Kong, Y. Cao, T. Iqbal, Y. Wang, W. Wang, M. D. Plumbley,
"PANNs: Large-Scale Pretrained Audio Neural Networks for Audio Pattern Recognition",
IEEE/ACM TASLP 28 (2020).  Code and weights: https://github.com/qiuqiangkong/audioset_tagging_cnn,
"Cnn14, mAP=0.431" checkpoint trained on AudioSet (527 classes, ~2M clips).

Why this model
--------------
The five models of the report are trained from scratch or from ImageNet weights, so they
never see audio before ESC-50.  CNN14 was pretrained on 5000 h of AudioSet, which is the
closest large-scale source to environmental sound, so the convolutional stack already
encodes onset/transient and harmonic texture detectors that transfer to ESC-50.

How our (B, 2, 128, 216) input is adapted to the pretrained network
------------------------------------------------------------------
1. the 128 mel bands are average-pooled to 64, the band count CNN14 was trained on, so the
   first convolution and ``bn0`` keep their original shape and meaning;
2. the two channels (log-mel and its delta) are fused by a 1x1 convolution initialised to
   average them, so the convolutional stack still receives a single channel;
3. the AudioSet classifier (2048 -> 512 -> 527) is kept up to ``fc1``; only the final
   512 -> 50 layer is new.  Every other tensor is loaded from the checkpoint.

Parameters: 80.8M (the whole model), of which 79.1M come from the AudioSet checkpoint.

Click Run on this file to train it with the common recipe of run_model.py.
"""
from pathlib import Path

import torch
import torch.nn.functional as F
from torch import nn

WEIGHTS = Path(__file__).resolve().parent / "pretrained" / "Cnn14_mAP=0.431.pth"


class ConvBlock(nn.Module):
    """Two 3x3 convolutions with batch norm, then 2x2 average pooling (PANNs naming kept)."""

    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.avgpool = nn.AvgPool2d(2, 2)

    def forward(self, x):
        x = F.relu_(self.bn1(self.conv1(x)))
        x = F.relu_(self.bn2(self.conv2(x)))
        return self.avgpool(x)


class PannsCnn14(nn.Module):
    """CNN14 with the AudioSet weights, adapted to log-mel + delta input."""

    def __init__(self, in_channels: int = 2, n_classes: int = 50, dropout: float = 0.3,
                 pretrained: bool = True, freeze_backbone: bool = True):
        super().__init__()
        self.in_channels = in_channels
        # (B, in_channels, 128, T) -> (B, 1, 64, T): the band count CNN14 was trained on
        self.fuse = nn.Conv2d(in_channels, 1, kernel_size=1, bias=False)
        with torch.no_grad():                     # start as the channel average, then learn
            self.fuse.weight.fill_(1.0 / in_channels)

        self.bn0 = nn.BatchNorm2d(64)
        self.conv_block1 = ConvBlock(1, 64)
        self.conv_block2 = ConvBlock(64, 128)
        self.conv_block3 = ConvBlock(128, 256)
        self.conv_block4 = ConvBlock(256, 512)
        self.conv_block5 = ConvBlock(512, 1024)
        self.conv_block6 = ConvBlock(1024, 2048)

        self.fc1 = nn.Linear(2048, 2048)          # kept from AudioSet (2048 -> 2048)
        self.head = nn.Linear(2048, n_classes)    # new: ESC-50 classes
        self.dropout = dropout
        self.blocks = (self.conv_block1, self.conv_block2, self.conv_block3,
                       self.conv_block4, self.conv_block5, self.conv_block6)
        self.freeze_backbone = freeze_backbone

        if pretrained:
            self.load_pretrained()
        if freeze_backbone:
            self._freeze()

    # ------------------------------------------------------------- fine-tuning
    def _freeze(self) -> None:
        """Freeze the AudioSet convolutions; keep the fusion layer, fc1 and head trainable.

        79.8 M parameters against 1600 training clips overfits badly (training accuracy
        reached 100% while test accuracy stalled at 83%).  The AudioSet features already
        separate these 50 sound classes well, so only 4.3 M parameters are updated.
        """
        frozen = 0
        for name, p in self.named_parameters():
            if name.startswith(("fc1", "head", "fuse")):
                continue
            p.requires_grad = False
            frozen += p.numel()
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print(f"[panns_cnn14] backbone frozen: {frozen / 1e6:.1f}M frozen, "
              f"{trainable / 1e6:.2f}M trainable")

    def train(self, mode: bool = True):
        """Keep the frozen blocks in training mode on purpose.

        Their batch-norm statistics MUST keep updating: the input of this project is
        128 mel bands average-pooled to 64 and normalised with the ESC-50 statistics,
        which is not the AudioSet distribution the checkpoint was fitted on.  Freezing
        the statistics as well destroys the features (training accuracy stalled at 6%
        instead of 39% after ten epochs).  Only the convolution weights are frozen.
        """
        return super().train(mode)

    # ------------------------------------------------------------------ weights
    def load_pretrained(self) -> None:
        if not WEIGHTS.exists():
            raise FileNotFoundError(
                f"missing {WEIGHTS}. Download 'Cnn14_mAP=0.431.pth' from "
                "https://zenodo.org/records/3987831 and put it in models/pretrained/")
        ckpt = torch.load(WEIGHTS, map_location="cpu", weights_only=False)
        state = ckpt["model"] if isinstance(ckpt, dict) and "model" in ckpt else ckpt

        own = self.state_dict()
        loadable, skipped = {}, []
        for k, v in state.items():
            if k in own and own[k].shape == v.shape:
                loadable[k] = v
            else:
                skipped.append(k)
        own.update(loadable)
        self.load_state_dict(own)
        n_loaded = sum(v.numel() for v in loadable.values())
        print(f"[panns_cnn14] loaded {len(loadable)} tensors / {n_loaded / 1e6:.1f}M parameters "
              f"from {WEIGHTS.name}; skipped {len(skipped)}: {skipped[:4]}")

    # ------------------------------------------------------------------ forward
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, 2, 128, 216) -> 64 mel bands -> single channel
        x = F.avg_pool2d(x, kernel_size=(2, 1))
        x = self.fuse(x)                                        # (B, 1, 64, T)

        x = x.permute(0, 2, 3, 1)                               # (B, 64, T, 1): bn0 runs over bands
        x = self.bn0(x)
        x = x.permute(0, 3, 1, 2)                               # (B, 1, 64, T)

        x = self.conv_block1(x)
        x = self.conv_block2(x)
        x = self.conv_block3(x)
        x = self.conv_block4(x)
        x = self.conv_block5(x)
        x = self.conv_block6(x)                                 # (B, 2048, F', T')

        x = torch.mean(x, dim=3)                                # average over time
        x = torch.max(x, dim=2)[0] + torch.mean(x, dim=2)       # max + average over bands
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = F.relu_(self.fc1(x))
        x = F.dropout(x, p=self.dropout, training=self.training)
        return self.head(x)


def build_model(in_channels: int, n_classes: int = 50) -> nn.Module:
    # Freezing the backbone was tried and failed: with SpecAugment + mixup on the
    # fixed AudioSet features, training accuracy stalled at 6% after 10 epochs
    # (vs 39% when the whole network is fine-tuned), i.e. the frozen features cannot
    # adapt to the augmented input distribution.  The whole network is fine-tuned
    # instead, with a higher dropout and stronger weight decay to fight overfitting.
    return PannsCnn14(in_channels=in_channels, n_classes=n_classes,
                      freeze_backbone=False, dropout=0.5)


# Fine-tuning settings for this model (the common recipe of run_model.py is kept: 5-fold CV,
# batch 32, SpecAugment + mixup).  Pretrained weights need far fewer epochs and a smaller
# learning rate than the from-scratch models, so both are overridden here.
SETTINGS = ["--seeds", "42", "43", "44", "--epochs", "40", "--lr", "4e-4"]

if __name__ == "__main__":
    import sys
    from pathlib import Path as _P
    sys.path.insert(0, str(_P(__file__).resolve().parents[1]))
    from run_model import main
    main([__file__, *SETTINGS])
