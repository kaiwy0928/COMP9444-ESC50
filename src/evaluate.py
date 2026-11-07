"""Fixed evaluation and plotting for every experiment, so runs can be compared like for like.

Every model writes results/<run>/predictions.npz with
    y_true (N,), y_prob (N, 50), fold (N,), filename (N,)
and, optionally, results/<run>/history.json ({fold: {train_loss, val_loss, train_acc, val_acc, best_epoch}}).
`evaluate_run` turns that into metrics.json, per_class.csv and the figures in results/<run>/figures/.
`compare_runs` puts several runs side by side.

Metrics (reported as mean ± std over the 5 test folds, the convention for ESC-50):
    accuracy, macro precision / recall / F1, Cohen's kappa, top-3 / top-5 accuracy, log loss.
ESC-50 is perfectly balanced, so macro recall equals balanced accuracy.

Usage:
    python -m src.evaluate results/crnn_baseline
    python -m src.evaluate --compare results/crnn_baseline results/crnn_specaug --out results/compare
"""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap  # noqa: E402
from sklearn.metrics import (accuracy_score, cohen_kappa_score, confusion_matrix, log_loss,  # noqa: E402
                             precision_recall_fscore_support, top_k_accuracy_score)

from .data import MAJOR_CATEGORIES, class_names  # noqa: E402

# ---------------- style: Nature Publishing Group palette (ggsci "npg"), single-hue sequential map ----------------

NPG = ["#E64B35", "#4DBBD5", "#00A087", "#3C5488", "#F39B7F", "#8491B4", "#91D1C2", "#DC0000", "#7E6148", "#B09C85"]
CM_CMAP = LinearSegmentedColormap.from_list("npg_blue", ["#FFFFFF", "#D6E4F0", "#8491B4", "#3C5488", "#1B2A4A"])
FIG_W = 7.2  # inches: double-column width in most journals; single column is ~3.5


def set_style():
    plt.rcParams.update({
        "font.family": "sans-serif", "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
        "font.size": 8, "axes.titlesize": 9, "axes.labelsize": 8, "xtick.labelsize": 7, "ytick.labelsize": 7,
        "legend.fontsize": 7, "axes.linewidth": 0.8, "axes.spines.top": False, "axes.spines.right": False,
        "xtick.major.width": 0.8, "ytick.major.width": 0.8, "lines.linewidth": 1.4,
        "axes.prop_cycle": matplotlib.cycler(color=NPG), "savefig.dpi": 300, "savefig.bbox": "tight",
        "pdf.fonttype": 42, "ps.fonttype": 42,  # embed TrueType so text stays editable in the PDF
    })


def save(fig, out: Path, name: str):
    out.mkdir(parents=True, exist_ok=True)
    for ext in ("png", "pdf"):
        fig.savefig(out / f"{name}.{ext}")
    plt.close(fig)


# ---------------- metrics ----------------

def metrics(y_true: np.ndarray, y_prob: np.ndarray) -> dict:
    y_pred = y_prob.argmax(1)
    labels = np.arange(y_prob.shape[1])
    p, r, f1, _ = precision_recall_fscore_support(y_true, y_pred, labels=labels, average="macro", zero_division=0)
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "macro_precision": p, "macro_recall": r, "macro_f1": f1,
        "cohen_kappa": cohen_kappa_score(y_true, y_pred),
        "top3_accuracy": top_k_accuracy_score(y_true, y_prob, k=3, labels=labels),
        "top5_accuracy": top_k_accuracy_score(y_true, y_prob, k=5, labels=labels),
        "log_loss": log_loss(y_true, np.clip(y_prob, 1e-7, 1), labels=labels),
    }


METRIC_NAMES = {"accuracy": "Accuracy", "macro_precision": "Macro precision", "macro_recall": "Macro recall",
                "macro_f1": "Macro F1", "cohen_kappa": "Cohen's κ", "top3_accuracy": "Top-3 accuracy",
                "top5_accuracy": "Top-5 accuracy", "log_loss": "Log loss"}


def load_predictions(run: Path) -> dict:
    with np.load(run / "predictions.npz", allow_pickle=True) as d:
        return {k: d[k] for k in d.files}


def evaluate_run(run: Path) -> dict:
    run = Path(run)
    set_style()
    pred = load_predictions(run)
    y, prob, fold = pred["y_true"], pred["y_prob"], pred["fold"]
    names = class_names()
    per_fold = {int(k): metrics(y[fold == k], prob[fold == k]) for k in np.unique(fold)}
    summary = {m: {"mean": float(np.mean([v[m] for v in per_fold.values()])),
                   "std": float(np.std([v[m] for v in per_fold.values()], ddof=1)) if len(per_fold) > 1 else 0.0}
               for m in METRIC_NAMES}
    pooled = {m: float(v) for m, v in metrics(y, prob).items()}

    y_pred = prob.argmax(1)
    p, r, f1, support = precision_recall_fscore_support(y, y_pred, labels=np.arange(50), zero_division=0)
    per_class = pd.DataFrame({"class": names, "major_category": [MAJOR_CATEGORIES[i // 10] for i in range(50)],
                              "precision": p, "recall": r, "f1": f1, "support": support})
    per_class.to_csv(run / "per_class.csv", index=False, float_format="%.4f")
    cm = confusion_matrix(y, y_pred, labels=np.arange(50))
    np.savetxt(run / "confusion_matrix.csv", cm, fmt="%d", delimiter=",")
    pairs = top_confusions(cm, names, k=10)

    config = run / "config.json"  # name the run from its config, not from the folder name
    name = json.loads(config.read_text(encoding="utf-8")).get("run", run.name) if config.exists() else run.name
    result = {"run": name, "folds": sorted(per_fold), "summary": summary, "pooled": pooled,
              "per_fold": {str(k): {m: float(v) for m, v in d.items()} for k, d in per_fold.items()},
              "top_confusions": pairs}
    (run / "metrics.json").write_text(json.dumps(result, indent=2, ensure_ascii=False))

    fig_dir = run / "figures"
    plot_confusion(cm, names, fig_dir, title=f"{run.name} — accuracy {summary['accuracy']['mean']:.2%} "
                                             f"± {summary['accuracy']['std']:.2%} (5-fold mean ± SD)")
    plot_major_confusion(y, y_pred, fig_dir)
    plot_per_class(per_class, fig_dir)
    plot_per_fold(per_fold, fig_dir)
    if (run / "history.json").exists():
        plot_history(json.loads((run / "history.json").read_text()), fig_dir)
    print_summary(result)
    return result


def top_confusions(cm: np.ndarray, names: list[str], k: int = 10) -> list[dict]:
    off = cm.copy()
    np.fill_diagonal(off, 0)
    idx = np.dstack(np.unravel_index(np.argsort(-off, axis=None)[:k], off.shape))[0]
    return [{"true": names[i], "predicted": names[j], "count": int(off[i, j])} for i, j in idx if off[i, j] > 0]


def print_summary(result: dict):
    s = result["summary"]
    print(f"== {result['run']} ({len(result['folds'])} folds)")
    for m, label in METRIC_NAMES.items():
        print(f"  {label:<16} {s[m]['mean']:.4f} ± {s[m]['std']:.4f}")
    print("  most confused:", "; ".join(f"{p['true']}→{p['predicted']} ({p['count']})" for p in result["top_confusions"][:5]))


# ---------------- figures ----------------

def plot_confusion(cm: np.ndarray, names: list[str], out: Path, title: str = ""):
    """50x50 confusion matrix of the 50-class model. Colour = share of the true class (row-normalised, so the diagonal
    is per-class recall); number = test clips, 5 folds pooled (40 per class). Major categories outlined."""
    norm = cm / cm.sum(1, keepdims=True).clip(min=1)
    fig, ax = plt.subplots(figsize=(11, 10.4))
    im = ax.imshow(norm, cmap=CM_CMAP, vmin=0, vmax=1, interpolation="nearest")
    for i, j in zip(*np.nonzero(cm)):
        ax.text(j, i, str(cm[i, j]), ha="center", va="center", fontsize=4.6 if i == j else 4.2,
                fontweight="bold" if i == j else "normal", color="white" if norm[i, j] > 0.5 else "#1B2A4A")
    labels = [n.replace("_", " ") for n in names]
    ax.set_xticks(range(50), labels, rotation=90, fontsize=6)
    ax.set_yticks(range(50), labels, fontsize=6)
    ax.set_xlabel("Predicted class", fontsize=9)
    ax.set_ylabel("True class", fontsize=9)
    for b in range(0, 51, 10):  # major-category blocks
        ax.axhline(b - 0.5, color="#B09C85", lw=0.7)
        ax.axvline(b - 0.5, color="#B09C85", lw=0.7)
    for side in ("top", "right"):
        ax.spines[side].set_visible(True)
    cb = fig.colorbar(im, ax=ax, fraction=0.03, pad=0.015)
    cb.set_label("Share of the true class (row-normalised)", fontsize=8)
    ax.set_title((f"{title}\n" if title else "") + "50-class confusion matrix — numbers are test clips "
                 "(5 folds pooled, 40 per class); bold diagonal = correct", fontsize=9)
    save(fig, out, "confusion_matrix_50_classes")


def plot_major_confusion(y: np.ndarray, y_pred: np.ndarray, out: Path):
    """Summary only: the 50-class predictions grouped into the 5 major categories of ESC-50 (class id // 10), to see
    whether mistakes stay inside a category. The model itself is always a 50-class classifier."""
    cm = confusion_matrix(y // 10, y_pred // 10, labels=range(5))
    norm = cm / cm.sum(1, keepdims=True)
    fig, ax = plt.subplots(figsize=(3.7, 3.4))
    ax.imshow(norm, cmap=CM_CMAP, vmin=0, vmax=1)
    for i in range(5):
        for j in range(5):
            ax.text(j, i, f"{norm[i, j]:.0%}\n({cm[i, j]})", ha="center", va="center", fontsize=6.5,
                    color="white" if norm[i, j] > 0.55 else "#1B2A4A")
    ax.set_xticks(range(5), MAJOR_CATEGORIES, rotation=35, ha="right")
    ax.set_yticks(range(5), MAJOR_CATEGORIES)
    ax.set_xlabel("Predicted category")
    ax.set_ylabel("True category")
    ax.set_title("50-class predictions grouped by major category\n(400 test clips per category)", fontsize=7.5)
    for side in ("top", "right"):
        ax.spines[side].set_visible(True)
    save(fig, out, "confusion_matrix_5_categories_summary")


def plot_per_class(per_class: pd.DataFrame, out: Path):
    df = per_class.sort_values("f1")
    colors = [NPG[MAJOR_CATEGORIES.index(c)] for c in df["major_category"]]
    fig, ax = plt.subplots(figsize=(FIG_W, 3.2))
    ax.bar(range(50), df["f1"], color=colors, width=0.75)
    ax.set_xticks(range(50), [n.replace("_", " ") for n in df["class"]], rotation=90, fontsize=5.5)
    ax.set_ylabel("F1 score")
    ax.set_ylim(0, 1)
    ax.axhline(df["f1"].mean(), color="#3C5488", lw=0.8, ls="--")
    handles = [plt.Rectangle((0, 0), 1, 1, color=NPG[i]) for i in range(5)]
    ax.legend(handles, MAJOR_CATEGORIES, ncol=5, frameon=False, loc="upper left", bbox_to_anchor=(0, 1.12))
    save(fig, out, "per_class_f1")


def plot_per_fold(per_fold: dict, out: Path):
    """Dot plot: each test fold is one point (marker shape = fold), the thick line is the mean, the error bar is ±1 SD.
    The y-axis is zoomed to the data, which is fine for points (bars would have to start at zero)."""
    keys = ["accuracy", "macro_f1", "cohen_kappa"]
    folds = sorted(per_fold)
    markers = ["o", "s", "^", "D", "v", "P", "X"]
    colors = [NPG[3], NPG[0], NPG[2]]
    fig, ax = plt.subplots(figsize=(3.5, 2.7))
    all_vals = []
    for i, m in enumerate(keys):
        vals = np.array([per_fold[f][m] for f in folds])
        all_vals += vals.tolist()
        mean, sd = vals.mean(), (vals.std(ddof=1) if len(vals) > 1 else 0.0)
        ax.errorbar(i, mean, yerr=sd, color="#333333", capsize=4, lw=0.9, capthick=0.9, zorder=2)
        ax.hlines(mean, i - 0.22, i + 0.22, color=colors[i], lw=2.2, zorder=3)
        for k, (x, v) in enumerate(zip(i + np.linspace(-0.15, 0.15, len(vals)), vals)):
            ax.scatter(x, v, s=18, marker=markers[k % len(markers)], facecolor=colors[i], edgecolor="white", lw=0.5,
                       alpha=0.9, zorder=4)
        ax.annotate(f"{mean:.3f} ± {sd:.3f}", (i, max(mean + sd, vals.max())), xytext=(0, 7),
                    textcoords="offset points", ha="center", fontsize=6.5, color="#333333")
    ax.set_xticks(range(len(keys)), [METRIC_NAMES[m] for m in keys])
    ax.set_xlim(-0.6, len(keys) - 0.4)
    lo, hi = min(all_vals), max(all_vals)
    ax.set_ylim(np.floor((lo - 0.03) * 20) / 20, np.ceil((hi + 0.05) * 20) / 20)
    ax.set_ylabel("Score on each test fold")
    ax.yaxis.grid(True, color="#E6E6E6", lw=0.6)
    ax.set_axisbelow(True)
    handles = [plt.Line2D([], [], marker=markers[k % len(markers)], ls="", color="#7F7F7F", markersize=4,
                          label=f"Fold {f}") for k, f in enumerate(folds)]
    ax.legend(handles=handles, frameon=False, ncol=len(folds), loc="lower center", bbox_to_anchor=(0.5, 1.0),
              fontsize=6.5, handletextpad=0.2, columnspacing=0.8)
    save(fig, out, "per_fold_metrics")


def plot_history(history: dict, out: Path):
    fig, axes = plt.subplots(1, 2, figsize=(FIG_W, 2.5))
    curves = all("curve_epochs" in h for h in history.values())
    for i, (fold, h) in enumerate(sorted(history.items())):
        c = NPG[i % len(NPG)]
        if curves:  # --curve-every: training folds without augmentation (dashed), test or validation fold (solid)
            ep = np.array(h["curve_epochs"])
            held = ((h["curve_test_loss"], h["curve_test_acc"]) if "curve_test_acc" in h else
                    (np.array(h["val_loss"])[ep - 1], np.array(h["val_acc"])[ep - 1]))
            axes[0].plot(ep, h["clean_train_loss"], color=c, lw=1, alpha=0.5, ls="--")
            axes[1].plot(ep, h["clean_train_acc"], color=c, lw=1, alpha=0.5, ls="--")
            axes[0].plot(ep, held[0], color=c, lw=1.2, label=f"Fold {fold}")
            axes[1].plot(ep, held[1], color=c, lw=1.2)
            continue
        ep = np.arange(1, len(h["train_loss"]) + 1)
        axes[0].plot(ep, h["train_loss"], color=c, lw=1, alpha=0.5, ls="--")
        axes[1].plot(ep, h["train_acc"], color=c, lw=1, alpha=0.5, ls="--")
        if "val_loss" not in h:  # --no-val runs: training curves only
            axes[0].lines[-1].set(label=f"Fold {fold}", ls="-", alpha=1)
            axes[1].lines[-1].set(ls="-", alpha=1)
            continue
        axes[0].plot(ep, h["val_loss"], color=c, lw=1.2, label=f"Fold {fold}")
        axes[1].plot(ep, h["val_acc"], color=c, lw=1.2)
        if "best_epoch" in h:
            axes[1].scatter([h["best_epoch"]], [h["val_acc"][h["best_epoch"] - 1]], color=c, s=12, zorder=3)
    if curves:
        held = "test" if all("curve_test_acc" in h for h in history.values()) else "validation"
        axes[0].set(xlabel="Epoch", ylabel="Loss", title=f"Loss (dashed: train, no aug.; solid: {held})")
        axes[1].set(xlabel="Epoch", ylabel="Accuracy", title=f"Accuracy (dashed: train, no aug.; solid: {held})",
                    ylim=(0, 1))
    elif all("val_loss" in h for h in history.values()):
        axes[0].set(xlabel="Epoch", ylabel="Loss", title="Loss (dashed: train, solid: validation)")
        axes[1].set(xlabel="Epoch", ylabel="Accuracy", title="Accuracy (dot: selected epoch)", ylim=(0, 1))
    else:
        axes[0].set(xlabel="Epoch", ylabel="Loss", title="Training loss (no validation fold)")
        axes[1].set(xlabel="Epoch", ylabel="Accuracy", title="Training accuracy (last epoch tested)", ylim=(0, 1))
    axes[0].legend(frameon=False, ncol=1)
    fig.tight_layout()
    save(fig, out, "learning_curves")


# ---------------- comparing runs ----------------

def compare_runs(runs: list[Path], out: Path):
    set_style()
    results = [json.loads((Path(r) / "metrics.json").read_text(encoding="utf-8")) for r in runs]
    if len({r["run"] for r in results}) < len(results):  # same run name in different folders: label with the folder
        for r, path in zip(results, runs):
            r["run"] = f"{Path(path).parent.name}/{r['run']}"
    rows = [{"run": r["run"], **{f"{m}_mean": r["summary"][m]["mean"] for m in METRIC_NAMES},
             **{f"{m}_std": r["summary"][m]["std"] for m in METRIC_NAMES}} for r in results]
    for row, path in zip(rows, runs):  # validation accuracy of the selected epochs: the number to compare settings on
        hist = Path(path) / "history.json"
        vals = [h.get("best_val_acc") for h in json.loads(hist.read_text()).values()] if hist.exists() else []
        row["val_acc_mean"] = float(np.mean(vals)) if vals and None not in vals else float("nan")
    out.mkdir(parents=True, exist_ok=True)
    table = pd.DataFrame(rows)
    table.to_csv(out / "comparison.csv", index=False, float_format="%.4f")
    keys = ["accuracy", "macro_f1", "top5_accuracy"]
    fig, ax = plt.subplots(figsize=(max(3.5, 1.2 * len(results) + 1.5), 2.6))
    w = 0.8 / len(results)
    for i, r in enumerate(results):
        ax.bar(np.arange(len(keys)) + (i - (len(results) - 1) / 2) * w, [r["summary"][m]["mean"] for m in keys], w,
               yerr=[r["summary"][m]["std"] for m in keys], capsize=2, error_kw={"lw": 0.8},
               color=NPG[i % len(NPG)], label=r["run"])
    ax.set_xticks(range(len(keys)), [METRIC_NAMES[m] for m in keys])
    ax.set_ylim(0, 1)
    ax.set_ylabel("Mean ± std over 5 folds")
    ax.legend(frameon=False, loc="upper left", bbox_to_anchor=(1.0, 1.0))
    save(fig, out, "comparison")
    print(table[["run", "val_acc_mean"] + [f"{m}_mean" for m in keys]].to_string(index=False))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("runs", nargs="+", type=Path)
    ap.add_argument("--compare", action="store_true")
    ap.add_argument("--out", type=Path, default=Path("results/compare"))
    a = ap.parse_args()
    if a.compare:
        compare_runs(a.runs, a.out)
    else:
        for r in a.runs:
            evaluate_run(r)


if __name__ == "__main__":
    main()
