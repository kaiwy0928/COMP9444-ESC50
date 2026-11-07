"""Build the processed log-mel dataset from the raw ESC-50 copy in data/raw/.

Outputs (ready to share):
    data/processed/clips.csv                                      one row per clip: index, filename, fold, class, silence
    data/processed/logmel/logmel_sr22050_m128_fft1024_hop512.npz  X (2000, 128, 216) float32 log-mel in dB, y, fold,
                                                                  filename, silence
    data/processed/logmel/logmel_sr22050_m128_fft1024_hop512.json the feature parameters used
    data/processed/logmel/images/<id>_<class>/<clip>.png          every clip as an image (same colour scale for all)
    data/processed/logmel/overview/<id>_<class>.png               the 40 clips of each class on one sheet
    data/processed/logmel/preview_before_after.png / .pdf         raw waveform vs log-mel, one clip per major category

Usage:
    python scripts/preprocess.py               # rebuild everything (~1 min)
    python scripts/preprocess.py --skip-build  # keep the existing .npz, only rewrite clips.csv and the preview
"""
import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.data import (ESC50, LOGMEL_DIR, MAJOR_CATEGORIES, PROCESSED, FeatureConfig, build_cache,  # noqa: E402
                      cache_path, class_names, load_cache, load_meta)


def write_clip_table(d: dict):
    meta = load_meta().set_index("filename")
    rows = meta.loc[d["filename"]].reset_index()
    rows.insert(0, "index", np.arange(len(rows)))
    rows["silence_share"] = d["silence"].round(4)
    rows["major_category"] = [MAJOR_CATEGORIES[t // 10] for t in rows["target"]]
    rows.to_csv(PROCESSED / "clips.csv", index=False)


def plot_before_after(d: dict, cfg: FeatureConfig):
    """One clip per major category: raw 44.1 kHz waveform (top) and the processed log-mel input (bottom)."""
    import librosa
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from src.evaluate import CM_CMAP, NPG, save, set_style
    set_style()
    names = class_names()
    picks = ["dog", "rain", "sneezing", "glass_breaking", "siren"]
    fig, axes = plt.subplots(2, 5, figsize=(7.2, 3.2), gridspec_kw={"height_ratios": [1, 1.6]})
    for col, cls in enumerate(picks):
        i = int(np.flatnonzero(d["y"] == names.index(cls))[0])
        raw, sr = librosa.load(ESC50 / "audio" / d["filename"][i], sr=None, mono=True)
        t = np.arange(len(raw)) / sr
        axes[0, col].plot(t, raw, color=NPG[col], lw=0.4)
        axes[0, col].set(title=cls.replace("_", " "), xlim=(0, 5), xticks=[0, 2.5, 5], yticks=[])
        if col == 0:
            axes[0, col].set_ylabel("Raw\n44.1 kHz")
        im = axes[1, col].imshow(d["X"][i], origin="lower", aspect="auto", cmap=CM_CMAP.reversed(),
                                 extent=(0, 5, 0, cfg.n_mels), vmin=-100, vmax=d["X"].max())
        axes[1, col].set(xlabel="Time (s)", xticks=[0, 2.5, 5], yticks=[0, 64, 128] if col == 0 else [])
        if col == 0:
            axes[1, col].set_ylabel("Log-mel band")
    cb = fig.colorbar(im, ax=axes[1, :], fraction=0.02, pad=0.01)
    cb.set_label("dB")
    save(fig, LOGMEL_DIR, "preview_before_after")


def _to_rgb(spec: np.ndarray, vmin: float, vmax: float, scale: int = 2, cmap=None) -> "np.ndarray":
    """(bands, frames) -> RGB uint8 image, low frequencies at the bottom, same colour scale for every clip."""
    from src.evaluate import CM_CMAP
    z = np.clip((spec - vmin) / (vmax - vmin), 0, 1)[::-1]
    rgb = ((cmap or CM_CMAP.reversed())(z)[..., :3] * 255).astype(np.uint8)
    return rgb.repeat(scale, 0).repeat(scale, 1)  # 128x216 is tiny on screen; nearest-neighbour upscale keeps pixels exact


def export_images(d: dict, out: Path = LOGMEL_DIR, label: str = "log-mel, 0–11 kHz", vmin: float = -100.0,
                  vmax: float = 40.0, cmap=None, unit: str = "dB"):
    """<out>/images/<id>_<class>/<clip>.png: every processed clip as an image (256 x 432 px, one fixed dB scale);
    <out>/overview/<id>_<class>.png: the 40 clips of one class on one sheet, labelled with fold and file name."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from PIL import Image
    from tqdm import tqdm
    from src.evaluate import set_style
    set_style()
    names = class_names()
    for c, name in enumerate(tqdm(names, desc="images")):
        folder = out / "images" / f"{c:02d}_{name}"
        folder.mkdir(parents=True, exist_ok=True)
        idx = np.flatnonzero(d["y"] == c)
        for i in idx:
            Image.fromarray(_to_rgb(d["X"][i], vmin, vmax, cmap=cmap)).save(folder / f"{Path(d['filename'][i]).stem}.png")
        fig, axes = plt.subplots(5, 8, figsize=(7.2, 5.4), gridspec_kw={"hspace": 0.45, "wspace": 0.12})
        for ax, i in zip(axes.flat, idx):
            ax.imshow(_to_rgb(d["X"][i], vmin, vmax, scale=1, cmap=cmap), aspect="auto")
            ax.set_title(f"fold {d['fold'][i]} · {Path(d['filename'][i]).stem}", fontsize=4.5)
            ax.axis("off")
        fig.suptitle(f"{c:02d} {name.replace('_', ' ')} — {label} (bottom to top), 0–5 s, "
                     f"{vmin:.0f} to {vmax:.0f} {unit}", fontsize=8)
        (out / "overview").mkdir(parents=True, exist_ok=True)
        fig.savefig(out / "overview" / f"{c:02d}_{name}.png", dpi=200)  # PNG only: a PDF of 40 bitmaps is ~3 MB each
        plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--skip-build", action="store_true")
    a = ap.parse_args()
    cfg = FeatureConfig()
    path = cache_path(cfg) if a.skip_build else build_cache(cfg)
    d = load_cache(cfg)
    write_clip_table(d)
    plot_before_after(d, cfg)
    export_images(d)
    X, y, silence = d["X"], d["y"], d["silence"]
    print(f"saved {path} ({path.stat().st_size / 1e6:.0f} MB), clips.csv, preview_before_after.png/pdf")
    print(f"X {X.shape} (clips, mel bands, frames), {X.dtype}; log-mel range {X.min():.1f} .. {X.max():.1f} dB")
    print(f"silence share per clip: mean {silence.mean():.1%}, clips with >50% silence {np.mean(silence > 0.5):.1%}")
    names = class_names()
    per_class = sorted(((silence[y == c].mean(), names[c]) for c in range(50)), reverse=True)
    print("most padded classes:", ", ".join(f"{n} {s:.0%}" for s, n in per_class[:6]))


if __name__ == "__main__":
    main()
