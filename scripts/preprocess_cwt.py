"""Build the CWT (Morlet scalogram) version of the dataset, with images to look at. Only the input representation
changes; every model and training setting stays the same as for log-mel (see src/data.py, CWTConfig).

Outputs:
    data/processed/cwt/cwt_morlet6_sr22050_k128_f40-10500_hop512.npz   X (2000, 128, 216) float32 dB, y, fold, filename,
                                                                    silence; same clip order as clips.csv
    data/processed/cwt/cwt_morlet6_....json                          CWT parameters
    data/processed/cwt/images/<id>_<class>/<clip>.png               every clip as an image (same colour scale)
    data/processed/cwt/overview/<id>_<class>.png                    the 40 clips of each class on one sheet
    data/processed/cwt/preview_waveform_logmel_cwt.png / .pdf       raw waveform vs log-mel vs CWT, 5 example clips
    data/processed/cwt/frequencies.csv                              centre frequency of each of the 128 rows

Usage: python scripts/preprocess_cwt.py [--skip-build]
"""
import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from preprocess import export_images  # noqa: E402
from src.data import (CWT_DIR, ESC50, CWTConfig, FeatureConfig, build_cache, cache_path,  # noqa: E402
                      class_names, load_cache)

OUT = CWT_DIR


def preview(cwt: dict, mel: dict, cfg: CWTConfig, vmin: float, vmax: float):
    import librosa
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from src.evaluate import CM_CMAP, NPG, save, set_style
    set_style()
    names = class_names()
    picks = ["dog", "rain", "sneezing", "glass_breaking", "siren"]
    freqs = cfg.frequencies()
    ticks = [int(np.argmin(np.abs(freqs - f))) for f in (100, 1000, 10000)]
    fig, axes = plt.subplots(3, 5, figsize=(7.2, 4.6), gridspec_kw={"height_ratios": [0.8, 1.3, 1.3]})
    for col, cls in enumerate(picks):
        i = int(np.flatnonzero(cwt["y"] == names.index(cls))[0])
        raw, sr = librosa.load(ESC50 / "audio" / cwt["filename"][i], sr=None, mono=True)
        axes[0, col].plot(np.arange(len(raw)) / sr, raw, color=NPG[col], lw=0.4)
        axes[0, col].set(title=cls.replace("_", " "), xlim=(0, 5), xticks=[], yticks=[])
        axes[1, col].imshow(mel["X"][i], origin="lower", aspect="auto", cmap=CM_CMAP.reversed(), vmin=-100, vmax=40,
                            extent=(0, 5, 0, 128))
        axes[1, col].set(xticks=[], yticks=[0, 64, 128] if col == 0 else [])
        axes[2, col].imshow(cwt["X"][i], origin="lower", aspect="auto", cmap=CM_CMAP.reversed(), vmin=vmin, vmax=vmax,
                            extent=(0, 5, 0, 128))
        axes[2, col].set(xlabel="Time (s)", xticks=[0, 2.5, 5])
        axes[2, col].set_yticks([t + 0.5 for t in ticks] if col == 0 else [], ["100 Hz", "1 kHz", "10 kHz"] if col == 0 else [])
    axes[0, 0].set_ylabel("Raw\n44.1 kHz")
    axes[1, 0].set_ylabel("Log-mel band")
    axes[2, 0].set_ylabel("CWT (Morlet)")
    fig.tight_layout()
    save(fig, OUT, "preview_waveform_logmel_cwt")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--skip-build", action="store_true")
    a = ap.parse_args()
    cfg = CWTConfig()
    path = cache_path(cfg) if a.skip_build else build_cache(cfg)
    d = load_cache(cfg)
    X = d["X"]
    # one colour scale for every image: from the 1st to the 99.9th percentile of all values, rounded to 10 dB
    vmin = float(np.floor(np.percentile(X, 1) / 10) * 10)
    vmax = float(np.ceil(np.percentile(X, 99.9) / 10) * 10) + 0.0  # + 0.0 turns -0.0 into 0.0 for the labels
    OUT.mkdir(parents=True, exist_ok=True)
    np.savetxt(OUT / "frequencies.csv", np.c_[np.arange(cfg.n_scales), cfg.frequencies()], fmt=["%d", "%.2f"],
               delimiter=",", header="row,centre_frequency_hz", comments="")
    export_images(d, out=OUT, label=f"CWT Morlet, {cfg.fmin:g} Hz–{cfg.fmax / 1000:g} kHz log-spaced", vmin=vmin,
                  vmax=vmax)
    preview(d, load_cache(FeatureConfig()), cfg, vmin, vmax)
    print(f"saved {path} ({path.stat().st_size / 1e6:.0f} MB) and images in {OUT}")
    print(f"X {X.shape} {X.dtype}; range {X.min():.1f} .. {X.max():.1f} dB; image colour scale {vmin:.0f} .. {vmax:.0f} dB")


if __name__ == "__main__":
    main()
