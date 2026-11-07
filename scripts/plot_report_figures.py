r"""Report figures for one model trained with several seeds (standard 5-fold CV runs):

  <out>/<name>_learning_curves.(png|pdf)  loss and accuracy every 10 epochs on the training folds without augmentation
                                          and on the test fold (runs trained with --curve-every 10); the mean over every
                                          fold and seed, with a light band from the lowest to the highest run
  <out>/<name>_per_class_f1.(png|pdf)     F1 of every class, one panel per ESC-50 major category, sorted; dots are
                                          the mean over seeds, thin lines the range over seeds

    python scripts/plot_report_figures.py --name multiscale_cnn --title "Multi-scale CNN" \
        --runs model_results/model_multiscale_cnn/final_seed42 model_results/model_multiscale_cnn/final_seed43 \
               model_results/model_multiscale_cnn/final_seed44
"""
import argparse
import json
import sys
import textwrap
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.data import MAJOR_CATEGORIES  # noqa: E402

INK, INK_2, INK_3 = "#0b0b0b", "#52514e", "#8a8984"      # primary / secondary / muted text
GRID, SURFACE = "#e6e5e1", "#fcfcfb"
SERIES = "#2a78d6"                                       # categorical slot 1 (training)
SERIES_2 = "#eb6834"                                     # categorical slot 2 (test / validation fold)
BAND = "#2a78d6"


def style():
    plt.rcParams.update({
        "font.family": "DejaVu Sans", "font.size": 9, "axes.titlesize": 10, "axes.labelsize": 9,
        "axes.edgecolor": INK_3, "axes.labelcolor": INK_2, "axes.linewidth": 0.8,
        "xtick.color": INK_2, "ytick.color": INK_2, "xtick.major.width": 0.8, "ytick.major.width": 0.8,
        "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True, "grid.color": GRID,
        "grid.linewidth": 0.6, "axes.axisbelow": True, "figure.facecolor": SURFACE, "axes.facecolor": SURFACE,
        "savefig.facecolor": SURFACE, "legend.frameon": False,
    })


def smooth(y: np.ndarray, k: int = 15) -> np.ndarray:
    """Centred moving average; the ends use the points available."""
    pad = k // 2
    c = np.convolve(np.pad(y, pad, mode="edge"), np.ones(k) / k, mode="valid")
    return c[:len(y)]


def learning_curves(runs: list[Path], title: str, out: Path):
    folds = [f for run in runs for f in json.loads((run / "history.json").read_text(encoding="utf-8")).values()]
    if "curve_epochs" not in folds[0]:
        # e.g. our committed baseline runs, trained before learning curves were recorded: the per-class F1 still works
        print("these runs have no learning curves (train them with --curve-every 10): learning-curve figure skipped")
        return
    ep = np.array(folds[0]["curve_epochs"])
    final = "curve_test_acc" in folds[0]  # standard 5-fold CV: test fold; development protocol: validation fold
    held_name = "Test fold" if final else "Validation fold"
    rows = {"train_loss": [], "held_loss": [], "train_acc": [], "held_acc": []}
    for f in folds:
        rows["train_loss"].append(f["clean_train_loss"])
        rows["train_acc"].append(np.array(f["clean_train_acc"]) * 100)
        rows["held_loss"].append(f["curve_test_loss"] if final else np.array(f["val_loss"])[ep - 1])
        rows["held_acc"].append((np.array(f["curve_test_acc"]) if final else np.array(f["val_acc"])[ep - 1]) * 100)
    rows = {k: np.array(v) for k, v in rows.items()}

    fig, axes = plt.subplots(1, 2, figsize=(7.6, 3.0))
    # the loss on augmented, mixed batches is left out: it sits above the test loss and only invites questions
    series = (("train_loss", "Training folds, no augmentation", SERIES, "solid"),
              ("held_loss", held_name, SERIES_2, "solid"))
    for key, label, color, ls in series:
        y = rows[key]
        axes[0].fill_between(ep, y.min(0), y.max(0), color=color, alpha=0.12, linewidth=0)
        axes[0].plot(ep, y.mean(0), color=color, linewidth=1.3, linestyle=ls, label=label)
    for key, color in (("train_acc", SERIES), ("held_acc", SERIES_2)):
        y = rows[key]
        axes[1].fill_between(ep, y.min(0), y.max(0), color=color, alpha=0.12, linewidth=0)
        axes[1].plot(ep, y.mean(0), color=color, linewidth=1.3)
        axes[1].annotate(f"{y.mean(0)[-1]:.1f}%", (ep[-1], y.mean(0)[-1]), xytext=(-2, -11 if key == "held_acc" else 5),
                         textcoords="offset points", ha="right", color=INK_2, fontsize=8)
    for ax, name in ((axes[0], "Loss"), (axes[1], "Accuracy (%)")):
        ax.set_title(name.split()[0], loc="left", color=INK, fontweight="bold")
        ax.set_xlabel("Epoch")
        ax.set_ylabel(name)
        ax.set_xlim(0, ep[-1])
    axes[0].set_ylim(bottom=0)
    axes[1].set_ylim(0, 102)
    fig.legend(*axes[0].get_legend_handles_labels(), loc="upper center", ncol=2, fontsize=8, bbox_to_anchor=(0.5, 1.07))
    fig.tight_layout()
    held_note = (f"{held_name}: read every {ep[1] if len(ep) > 1 else 1} epochs for these curves only; it chose nothing ({ep[-1]} epochs fixed "
                 "in advance, the last epoch is reported)." if final else
                 f"{held_name}: one of the training folds held out (three folds train).")
    caption = (f"{title}: mean of {len(folds)} runs ({len(runs)} seeds x {len(folds) // len(runs)} folds); band = lowest to "
               f"highest run. Training accuracy is measured on the training folds without augmentation, in evaluation "
               f"mode (accuracy on the augmented, mixup-mixed batches stays near 50% by construction). {held_note}")
    fig.text(0.01, -0.02, textwrap.fill(caption, 150), fontsize=7.2, color=INK_2, ha="left", va="top")
    save(fig, out)


def per_class_f1(runs: list[Path], title: str, out: Path):
    tables = [pd.read_csv(run / "per_class.csv") for run in runs]
    df = tables[0][["class", "major_category"]].copy()
    f1 = np.array([t["f1"].to_numpy() for t in tables]) * 100
    df["f1"], df["lo"], df["hi"] = f1.mean(0), f1.min(0), f1.max(0)
    df["class"] = df["class"].str.replace("_", " ")
    overall = df["f1"].mean()
    fig, axes = plt.subplots(2, 3, figsize=(10.5, 6.6), sharex=True)
    for ax, cat in zip(axes.flat, MAJOR_CATEGORIES):
        d = df[df["major_category"] == cat].sort_values("f1")
        y = np.arange(len(d))
        # dot plot (not bars): the axis does not start at zero, so lengths must not carry the value
        ax.axvline(overall, color=INK_3, linewidth=0.9, linestyle=(0, (3, 3)), zorder=1)
        ax.hlines(y, d["lo"], d["hi"], color=SERIES, linewidth=2.0, alpha=0.35, zorder=2)
        ax.scatter(d["f1"], y, s=34, color=SERIES, edgecolor=SURFACE, linewidth=1.2, zorder=3)
        ax.set_yticks(y, d["class"])
        ax.tick_params(axis="y", length=0)
        ax.grid(axis="y", color=GRID, linewidth=0.5)
        ax.set_title(cat, loc="left", color=INK, fontweight="bold", fontsize=9.5, pad=14)
        ax.text(0, 1.015, f"mean F1 {d['f1'].mean():.1f}", transform=ax.transAxes, fontsize=8, color=INK_2,
                va="bottom", ha="left")
        ax.set_xlim(50, 100)
        ax.set_xticks(range(50, 101, 10))
        ax.set_ylim(-0.6, len(d) - 0.4)
    axes[0][2].tick_params(labelbottom=True)  # nothing below it: give it its own tick labels
    axes[0][2].set_xlabel("F1 score (%)")
    for ax in axes[1]:
        ax.set_xlabel("F1 score (%)")
    note = axes.flat[5]
    note.axis("off")
    worst = df.sort_values("f1").head(5)
    lines = "\n".join(f"{r['class']:<16} {r['f1']:.1f}" for _, r in worst.iterrows())
    note.text(0.02, 0.95, f"{title}\n\nOverall macro F1: {overall:.1f}%\n(dashed line in every panel)\n\n"
              f"Dots: mean over {len(runs)} seeds\nLight lines: range over seeds\n\nHardest classes:\n{lines}",
              va="top", ha="left", fontsize=8.5, color=INK_2, family="DejaVu Sans Mono", transform=note.transAxes)
    # fixed spacing: the automatic layout gives the gaps (not the plots) the extra width
    fig.subplots_adjust(left=0.105, right=0.985, bottom=0.08, top=0.93, wspace=0.62, hspace=0.30)
    save(fig, out)


def save(fig, out: Path):
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out.with_suffix(".png"), dpi=200, bbox_inches="tight")
    fig.savefig(out.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)
    print("saved", out.with_suffix(".png"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", nargs="+", required=True, help="results folders of the seeds of one model")
    ap.add_argument("--name", required=True)
    ap.add_argument("--title", required=True)
    ap.add_argument("--out", default="results/report_figures")
    a = ap.parse_args()
    style()
    runs = [ROOT / r for r in a.runs]
    learning_curves(runs, a.title, ROOT / a.out / f"{a.name}_learning_curves")
    per_class_f1(runs, a.title, ROOT / a.out / f"{a.name}_per_class_f1")


if __name__ == "__main__":
    main()
