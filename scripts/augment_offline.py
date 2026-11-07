"""Build the offline waveform-augmented copies (see src/augment.py) and a preview figure.

Outputs in data/processed/augmented/:
    logmel_..._waveaug4.npz       X_aug (2000, 4, 128, 216) float16, same clip order as the original .npz
    logmel_..._waveaug4.json      parameters + which augmentations each copy got
    preview_augmentation.png/pdf  one clip: original vs pitch shift / time stretch / noise / SpecAugment

Usage: python scripts/augment_offline.py [--jobs 8]
"""
import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.augment import AugConfig, build_aug_cache, coloured_noise, fix_length  # noqa: E402
from src.data import AUG_DIR, ESC50, FeatureConfig, load_audio, log_mel, spec_augment  # noqa: E402


def preview(feat: FeatureConfig):
    import librosa
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import torch
    from src.evaluate import CM_CMAP, save, set_style
    set_style()
    fn = "1-100038-A-14.wav"  # chirping birds: clear harmonic structure, so the changes are easy to see
    y = load_audio(ESC50 / "audio" / fn, feat.sample_rate)
    rng = np.random.default_rng(0)
    noise = coloured_noise(len(y), "pink", rng)
    noise *= np.sqrt(np.mean(y ** 2) / 10 ** (15 / 10) / np.mean(noise ** 2))
    versions = {
        "Original": y,
        "Pitch +2 semitones": librosa.effects.pitch_shift(y, sr=feat.sample_rate, n_steps=2),
        "Stretch ×1.15": fix_length(librosa.effects.time_stretch(y, rate=1.15), len(y), rng),
        "Pink noise 15 dB": y + noise,
    }
    specs = {k: log_mel(v / np.abs(v).max(), feat) for k, v in versions.items()}
    torch.manual_seed(0)
    np.random.seed(3)
    base = specs["Original"]
    masked = spec_augment(torch.from_numpy((base - base.mean()) / base.std())[None], freq_width=20, time_width=30)[0]
    specs["SpecAugment"] = masked.numpy() * base.std() + base.mean()
    fig, axes = plt.subplots(1, 5, figsize=(7.2, 1.9), sharey=True)
    for ax, (k, s) in zip(axes, specs.items()):
        ax.imshow(s, origin="lower", aspect="auto", cmap=CM_CMAP.reversed(), vmin=-100, vmax=40, extent=(0, 5, 0, 128))
        ax.set(title=k, xlabel="Time (s)", xticks=[0, 2.5, 5])
    axes[0].set_ylabel("Log-mel band")
    save(fig, AUG_DIR, "preview_augmentation")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--jobs", type=int, default=8)
    a = ap.parse_args()
    feat, cfg = FeatureConfig(), AugConfig()
    path = build_aug_cache(feat, cfg, n_jobs=a.jobs)
    preview(feat)
    print(f"saved {path} ({path.stat().st_size / 1e6:.0f} MB) and preview_augmentation.png/pdf")


if __name__ == "__main__":
    main()
