"""Build the delta (first temporal derivative) channel of the log-mel features, with images to look at.
Used as a second input channel next to log-mel (--features logmel_delta); nothing else changes.

Outputs:
    data/processed/delta/logmel_sr22050_m128_fft1024_hop512_delta9.npz   D (2000, 128, 216) float32 dB/frame, filename;
                                                                    same clip order as the log-mel file and clips.csv
    data/processed/delta/logmel_..._delta9.json                        delta parameters
    data/processed/delta/images/<id>_<class>/<clip>.png             every clip's delta as an image (same colour scale)
    data/processed/delta/overview/<id>_<class>.png                  the 40 clips of each class on one sheet
    data/processed/delta/preview_waveform_logmel_delta.png / .pdf   raw waveform vs log-mel vs delta, 5 example clips

Usage: python scripts/preprocess_delta.py   (needs the log-mel file from scripts/preprocess.py)
"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from preprocess import export_images  # noqa: E402
from src.data import (DELTA_DIR, DELTA_WIDTH, ESC50, FeatureConfig, build_delta, class_names,  # noqa: E402
                      delta_name, load_cache)

OUT = DELTA_DIR


def diverging_cmap():
    """Blue (energy falling) - white (no change) - red (energy rising), from the NPG palette."""
    from matplotlib.colors import LinearSegmentedColormap
    return LinearSegmentedColormap.from_list("npg_div", ["#1B2A4A", "#3C5488", "#FFFFFF", "#E64B35", "#7E1E12"])


def preview(mel: dict, D: np.ndarray, lim: float):
    import librosa
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from src.evaluate import CM_CMAP, NPG, save, set_style
    set_style()
    names = class_names()
    picks = ["dog", "rain", "sneezing", "glass_breaking", "siren"]
    fig, axes = plt.subplots(3, 5, figsize=(7.2, 4.6), gridspec_kw={"height_ratios": [0.8, 1.3, 1.3]})
    for col, cls in enumerate(picks):
        i = int(np.flatnonzero(mel["y"] == names.index(cls))[0])
        raw, sr = librosa.load(ESC50 / "audio" / mel["filename"][i], sr=None, mono=True)
        axes[0, col].plot(np.arange(len(raw)) / sr, raw, color=NPG[col], lw=0.4)
        axes[0, col].set(title=cls.replace("_", " "), xlim=(0, 5), xticks=[], yticks=[])
        axes[1, col].imshow(mel["X"][i], origin="lower", aspect="auto", cmap=CM_CMAP.reversed(), vmin=-100, vmax=40,
                            extent=(0, 5, 0, 128))
        axes[1, col].set(xticks=[], yticks=[0, 64, 128] if col == 0 else [])
        im = axes[2, col].imshow(D[i], origin="lower", aspect="auto", cmap=diverging_cmap(), vmin=-lim, vmax=lim,
                                 extent=(0, 5, 0, 128))
        axes[2, col].set(xlabel="Time (s)", xticks=[0, 2.5, 5], yticks=[0, 64, 128] if col == 0 else [])
    axes[0, 0].set_ylabel("Raw\n44.1 kHz")
    axes[1, 0].set_ylabel("Log-mel band")
    axes[2, 0].set_ylabel("Delta band")
    fig.tight_layout()
    cb = fig.colorbar(im, ax=axes[2, :], fraction=0.02, pad=0.01)
    cb.set_label("dB / frame")
    save(fig, OUT, "preview_waveform_logmel_delta")


def main():
    path = build_delta(FeatureConfig())
    mel = load_cache(FeatureConfig())
    with np.load(path, allow_pickle=True) as d:
        D = d["D"]
    lim = float(np.ceil(np.percentile(np.abs(D), 99.5)))  # symmetric colour scale, so white = no change
    export_images({"X": D, "y": mel["y"], "fold": mel["fold"], "filename": mel["filename"]}, out=OUT,
                  label=f"log-mel delta (width {DELTA_WIDTH}), blue = falling, red = rising, 0–11 kHz",
                  vmin=-lim, vmax=lim, cmap=diverging_cmap(), unit="dB/frame")
    preview(mel, D, lim)
    print(f"saved {path} ({path.stat().st_size / 1e6:.0f} MB) and images in {OUT}")
    print(f"D {D.shape} {D.dtype}; range {D.min():.1f} .. {D.max():.1f} dB/frame; colour scale ±{lim:.0f}; "
          f"name {delta_name()}")


if __name__ == "__main__":
    main()
