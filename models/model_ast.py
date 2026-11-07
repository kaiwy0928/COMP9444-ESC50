"""model_ast: Audio Spectrogram Transformer (Gong et al. 2021) fine-tuned on ESC-50.

Reference: Y. Gong, Y.-A. Chung, J. Glass, "AST: Audio Spectrogram Transformer",
Interspeech 2021.  Checkpoint: MIT/ast-finetuned-audioset-10-10-0.4593
(DeiT-style ViT-B, 12 layers, 768 hidden, 86.6 M parameters), pretrained on
AudioSet (527 classes) from ImageNet-initialised weights.

Why this model
--------------
The same argument as the other pretrained entries, taken further: AST is a pure
transformer over spectrogram patches, pretrained on 5000 h of AudioSet, so it
carries the strongest available onset/harmonic representation for environmental
sound.  It is the only model in the comparison whose pretraining corpus is both
large and audio-only.

How our (B, 2, 128, 216) input is adapted
-----------------------------------------
1. the two channels (log-mel and its delta) are fused by a 1x1 convolution
   initialised to average them, so the patch embedding still sees one channel
   and keeps its pretrained shape (768, 1, 16, 16);
2. AST patches with kernel 16 and stride 10, so 216 frames give
   12 (bands) x 21 (time) = 252 patches instead of the 1212 it was trained on.
   The patch embedding is stride-based and therefore input-length agnostic, but
   the position embeddings are not: they are bicubically interpolated from the
   pretrained 12 x 101 grid to the 12 x 21 grid on the first forward pass, and
   the two special tokens (cls, distillation) are carried over untouched;
3. only the final 768 -> 50 projection is new.  Every other tensor, including
   the cls/dist tokens, comes from the AudioSet checkpoint.
"""
import os
from pathlib import Path

os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")

import torch
import torch.nn.functional as F
from torch import nn

HF_ID = "MIT/ast-finetuned-audioset-10-10-0.4593"
PATCH, STRIDE = 16, 10


def _grid(frames: int) -> tuple[int, int]:
    """(bands, time) patch grid that ASTPatchEmbeddings produces for 128 x frames."""
    return (128 - PATCH) // STRIDE + 1, (frames - PATCH) // STRIDE + 1


class AudioAST(nn.Module):
    """AST with AudioSet weights, adapted to log-mel + delta input."""

    def __init__(self, in_channels: int = 2, n_classes: int = 50, dropout: float = 0.1,
                 pretrained: bool = True, freeze_backbone: bool = False):
        super().__init__()
        from transformers import ASTForAudioClassification

        self.fuse = nn.Conv2d(in_channels, 1, kernel_size=1, bias=False)
        with torch.no_grad():
            self.fuse.weight.fill_(1.0 / in_channels)

        if pretrained:
            self.ast = ASTForAudioClassification.from_pretrained(HF_ID)
            n = sum(p.numel() for p in self.ast.parameters())
            print(f"[ast] loaded {n / 1e6:.1f}M parameters from {HF_ID}")
        else:
            self.ast = ASTForAudioClassification(self.ast_config())

        # new task head; everything else stays as pretrained
        hidden = self.ast.config.hidden_size
        self.ast.classifier.dense = nn.Linear(hidden, n_classes)
        self.ast.config.num_labels = n_classes
        self.ast.num_labels = n_classes
        self.ast.config.problem_type = "single_label_classification"

        self.ast.config.classifier_dropout_prob = dropout
        self.ast.config.hidden_dropout_prob = dropout
        self.ast.config.attention_probs_dropout_prob = dropout

        self._src_grid: tuple[int, int] | None = None   # grid the position embeddings currently describe
        self._resized_for: int | None = None            # input length they were built for

        if freeze_backbone:
            self._freeze()

    @staticmethod
    def ast_config():
        from transformers import ASTConfig
        return ASTConfig(num_mel_bins=128, num_labels=50)

    def _freeze(self) -> None:
        for name, p in self.named_parameters():
            if name.startswith(("fuse", "ast.classifier")):
                continue
            p.requires_grad = False
        frozen = sum(p.numel() for p in self.parameters() if not p.requires_grad)
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print(f"[ast] backbone frozen: {frozen / 1e6:.1f}M frozen, {trainable / 1e6:.2f}M trainable")

    # ------------------------------------------------------- position embedding
    def _fit_positions(self, frames: int) -> None:
        """Bicubically resample the pretrained position grid to the one this input produces."""
        if self._resized_for == frames:
            return
        emb = self.ast.audio_spectrogram_transformer.embeddings
        pos = emb.position_embeddings                       # (1, 2 + nf*nt, hidden)
        n_special = pos.shape[1] - (lambda g: g[0] * g[1])(
            self._src_grid if self._src_grid else (12, 101))
        special, patch_pos = pos[:, :n_special], pos[:, n_special:]

        src = self._src_grid or (12, 101)                   # the grid the checkpoint was trained on
        dst = _grid(frames)
        if src == dst:
            self._resized_for = frames
            return
        hidden = patch_pos.shape[-1]
        grid = patch_pos.reshape(1, src[0], src[1], hidden).permute(0, 3, 1, 2)   # (1, C, nf, nt)
        grid = F.interpolate(grid, size=dst, mode="bicubic", align_corners=False)
        patch_pos = grid.permute(0, 2, 3, 1).reshape(1, dst[0] * dst[1], hidden)
        emb.position_embeddings = nn.Parameter(torch.cat([special, patch_pos], dim=1))
        self._src_grid, self._resized_for = dst, frames
        print(f"[ast] position embeddings {src[0]}x{src[1]} -> {dst[0]}x{dst[1]} for {frames} frames")

    # ------------------------------------------------------------------ forward
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, 2, 128, T) -> single channel -> (B, T, 128) as AST expects
        x = self.fuse(x)
        x = x.squeeze(1).transpose(1, 2).contiguous()
        self._fit_positions(x.shape[1])
        return self.ast(input_values=x).logits


def build_model(in_channels: int, n_classes: int = 50) -> nn.Module:
    # The whole transformer is fine-tuned: with only 1600 training clips a frozen
    # AST saturates around a linear probe, and the SpecAugment/mixup recipe of
    # run_model.py needs the encoder to adapt to the augmented distribution.
    return AudioAST(in_channels=in_channels, n_classes=n_classes, dropout=0.1)


SETTINGS = ["--seeds", "42", "43", "44", "--epochs", "20", "--lr", "1e-4", "--weight-decay", "1e-2"]

if __name__ == "__main__":
    import sys
    from pathlib import Path as _P
    sys.path.insert(0, str(_P(__file__).resolve().parents[1]))
    from run_model import main
    main([__file__, *SETTINGS])
