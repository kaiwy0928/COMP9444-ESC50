"""5-fold cross-validation training on ESC-50.

For each test fold k: train on three folds, pick the epoch with the best validation accuracy on fold (k mod 5) + 1,
then predict the test fold with that checkpoint. All test predictions go to results/<run>/predictions.npz and are
scored by src/evaluate.py, so every model is evaluated the same way.

The model comes from a model file that defines build_model(in_channels, n_classes): models/model_baseline.py by
default, or any models/model_<name>.py. A build_model that also takes test_fold is told the fold being tested (for models
that use per-fold information chosen without that fold). Results go to --out (default results/<run>/).

Usage (from the repository root):
    python -m src.train --run delta_baseline --features logmel_delta --out baseline/results/delta_baseline
    python -m src.train --run quick --features logmel_delta --folds 1 --epochs 5   # smoke test -> results/quick/
Comparing models: run_model.py calls main([...]) with the common recipe, so this is rarely typed by hand.
"""
import argparse
import importlib.util
import inspect
import json
import os
import random
import sys
import time
from pathlib import Path

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")  # required by cuBLAS for deterministic GPU results

import numpy as np  # noqa: E402
import torch  # noqa: E402
from torch import nn  # noqa: E402
from torch.utils.data import DataLoader  # noqa: E402

from .data import (FEATURE_CHOICES, PROJECT, FeatureConfig, LogMelDataset, fold_split, load_features, norm_stats,
                   with_delta)
from .evaluate import evaluate_run


def load_model_builder(path: Path):
    """Import build_model(in_channels, n_classes) from a model file given by its path."""
    spec = importlib.util.spec_from_file_location(f"model_{path.parent.name}", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.build_model


def rel(path: Path) -> str:
    """Path relative to the repository when possible, for logs and config.json."""
    return path.relative_to(PROJECT).as_posix() if path.is_relative_to(PROJECT) else str(path)


def count_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


class Tee:
    """Write everything printed to the console into results/<run>/train.log as well."""

    def __init__(self, path: Path):
        self.file = open(path, "w", encoding="utf-8")
        self.console = sys.stdout

    def write(self, text):
        self.console.write(text)
        self.file.write(text)
        self.file.flush()

    def flush(self):
        self.console.flush()
        self.file.flush()


def seed_all(seed: int):
    """Same seed + deterministic kernels: a rerun on the same hardware and software gives identical predictions.
    (Different GPUs, drivers or CPU vs GPU can still differ slightly in floating point.)"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True, warn_only=True)


def make_optimizer(args, model):
    if args.optimizer == "sgd":  # as in Zhang et al. (2019): SGD with Nesterov momentum
        return torch.optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum, nesterov=True,
                               weight_decay=args.weight_decay)
    return torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)


def make_scheduler(args, optimizer, steps_per_epoch: int):
    """Learning-rate schedule, stepped after every batch.
    onecycle  warm-up over the first 10% to lr, then cosine decay to ~0 (the baseline)
    cosine    cosine decay from lr to lr/1e4, no warm-up
    sgdr      cosine with warm restarts (Loshchilov & Hutter, 2017): cycles of sgdr_t0, x sgdr_mult each time
    step      lr x step_gamma every step_epochs epochs (Zhang et al., 2019)"""
    total = args.epochs * steps_per_epoch
    if args.scheduler == "cosine":
        return torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=total, eta_min=args.lr * 1e-4)
    if args.scheduler == "sgdr":
        return torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer, T_0=args.sgdr_t0 * steps_per_epoch, T_mult=args.sgdr_mult, eta_min=args.lr * 1e-4)
    if args.scheduler == "step":
        return torch.optim.lr_scheduler.StepLR(optimizer, step_size=args.step_epochs * steps_per_epoch,
                                               gamma=args.step_gamma)
    return torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=args.lr, epochs=args.epochs,
                                               steps_per_epoch=steps_per_epoch, pct_start=0.1)


def window_log_probs(model, x, crop: int, hop: int):
    """Sub-segment evaluation: softmax of every window of `crop` frames, `hop` frames apart (the last window ends at
    the end of the clip), averaged per clip. Returns the log of the averaged probabilities."""
    frames = x.shape[-1]
    starts = list(range(0, frames - crop + 1, hop))
    if starts[-1] != frames - crop:
        starts.append(frames - crop)
    p = torch.stack([torch.softmax(model(x[..., s:s + crop]).float(), 1) for s in starts]).mean(0)
    return torch.log(p.clamp_min(1e-12))


def run_epoch(model, loader, device, optimizer=None, scheduler=None, mixup_alpha: float = 0.0, crop: int = 0,
              hop: int = 0, label_smoothing: float = 0.0):
    """One pass over the loader. Trains when an optimizer is given. Returns (mean loss, accuracy, probabilities).
    With crop > 0, evaluation averages windows of `crop` frames (training crops are made by the dataset).
    label_smoothing applies to the training loss only, so validation losses stay comparable."""
    train = optimizer is not None
    model.train(train)
    loss_fn = nn.CrossEntropyLoss(label_smoothing=label_smoothing) if train and label_smoothing else nn.CrossEntropyLoss()
    total, correct, loss_sum, probs = 0, 0, 0.0, []
    with torch.set_grad_enabled(train):
        for x, y in loader:
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            if train and mixup_alpha > 0:
                lam = float(np.random.beta(mixup_alpha, mixup_alpha))
                perm = torch.randperm(x.size(0), device=device)
                logits = model(lam * x + (1 - lam) * x[perm])
                loss = lam * loss_fn(logits, y) + (1 - lam) * loss_fn(logits, y[perm])
            elif not train and crop:
                logits = window_log_probs(model, x, crop, hop)
                loss = nn.functional.nll_loss(logits, y)
            else:
                logits = model(x)
                loss = loss_fn(logits, y)
            if train:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 5.0)
                optimizer.step()
                scheduler.step()
            loss_sum += loss.item() * x.size(0)
            correct += (logits.argmax(1) == y).sum().item()
            total += x.size(0)
            if not train:
                probs.append(torch.softmax(logits.float(), 1).cpu().numpy())
    return loss_sum / total, correct / total, (np.concatenate(probs) if probs else None)


def train_fold(args, data: dict, test_fold: int, device) -> tuple[np.ndarray, np.ndarray, dict]:
    seed_all(args.seed + test_fold)
    train_folds, val_fold, _ = fold_split(test_fold)
    if args.no_val:  # final evaluation: the validation fold joins training, the last epoch is tested
        train_folds, val_fold = sorted(train_folds + [val_fold]), None
    X, y, fold = data["X"], data["y"], data["fold"]
    tr, va, te = np.isin(fold, train_folds), fold == val_fold, fold == test_fold
    mean, std = norm_stats(X[tr])  # training folds only
    aug = (dict(freq_masks=args.n_masks, time_masks=args.n_masks, freq_width=args.mask_freq_width,
                time_width=args.mask_time_width) if args.spec_augment else None)
    X_aug = data["X_aug"][tr] if "X_aug" in data else None
    ds = {"train": LogMelDataset(X[tr], y[tr], mean, std, spec_augment=aug, X_aug=X_aug, time_shift=args.time_shift,
                                  crop=args.crop_frames),
          "val": LogMelDataset(X[va], y[va], mean, std), "test": LogMelDataset(X[te], y[te], mean, std)}
    if args.no_val:
        del ds["val"]
    loaders = {k: DataLoader(v, batch_size=args.batch_size, shuffle=(k == "train"), num_workers=0,
                             pin_memory=True, drop_last=(k == "train")) for k, v in ds.items()}

    fold_kw = {"test_fold": test_fold} if "test_fold" in inspect.signature(args.build_model).parameters else {}
    model = args.build_model(in_channels=args.in_channels, **fold_kw).to(device)
    optimizer = make_optimizer(args, model)
    scheduler = make_scheduler(args, optimizer, len(loaders["train"]))
    hist = {"train_loss": [], "val_loss": [], "train_acc": [], "val_acc": []}
    # learning-curve loaders get their own generator: iterating a DataLoader draws from the global RNG otherwise
    curve = ({"train": DataLoader(LogMelDataset(X[tr], y[tr], mean, std), batch_size=64, num_workers=0,
                                  generator=torch.Generator()),
              "test": DataLoader(ds["test"], batch_size=args.batch_size, num_workers=0, generator=torch.Generator())}
             if args.curve_every else None)
    best_acc, best_state, best_epoch = -1.0, None, 0
    for epoch in range(1, args.epochs + 1):
        tl, ta, _ = run_epoch(model, loaders["train"], device, optimizer, scheduler, args.mixup,
                              label_smoothing=args.label_smoothing)
        if args.curve_every and (epoch % args.curve_every == 0 or epoch in (1, args.epochs)):
            # learning curves. The training folds without augmentation, in eval mode: the real training accuracy (the
            # per-epoch train_acc is measured on augmented, mixed batches). With --no-val also the test fold: read for
            # the curve only, never for a choice (fixed epochs, the last one is reported). No randomness is drawn, so
            # training is unchanged.
            cl, ca, _ = run_epoch(model, curve["train"], device)
            rows = [("curve_epochs", epoch), ("clean_train_loss", round(cl, 5)), ("clean_train_acc", round(ca, 5))]
            if args.no_val:
                ml, ma, _ = run_epoch(model, curve["test"], device, crop=args.crop_frames, hop=args.crop_hop)
                rows += [("curve_test_loss", round(ml, 5)), ("curve_test_acc", round(ma, 5))]
            for k, v in rows:
                hist.setdefault(k, []).append(v)
        if args.no_val:  # no validation and no early stopping; the test fold chooses nothing
            hist["train_loss"].append(round(tl, 5))
            hist["train_acc"].append(round(ta, 5))
            if epoch % 10 == 0 or epoch == args.epochs:
                print(f"  fold {test_fold} epoch {epoch:3d}: train loss {tl:.3f} acc {ta:.3f}")
            continue
        vl, vacc, _ = run_epoch(model, loaders["val"], device, crop=args.crop_frames, hop=args.crop_hop)
        for k, v in zip(hist, (tl, vl, ta, vacc)):
            hist[k].append(round(v, 5))
        if vacc > best_acc:
            best_acc, best_epoch = vacc, epoch
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        if epoch % 10 == 0 or epoch == args.epochs:
            print(f"  fold {test_fold} epoch {epoch:3d}: train loss {tl:.3f} acc {ta:.3f} | "
                  f"val loss {vl:.3f} acc {vacc:.3f} (best {best_acc:.3f} @ {best_epoch})")
    if args.no_val:
        del hist["val_loss"], hist["val_acc"]
        best_state, best_epoch, best_acc = model.state_dict(), args.epochs, None
    model.load_state_dict(best_state)
    _, test_acc, prob = run_epoch(model, loaders["test"], device, crop=args.crop_frames, hop=args.crop_hop)
    hist["best_epoch"] = best_epoch
    hist["best_val_acc"] = best_acc
    hist["test_acc"] = test_acc
    print(f"  fold {test_fold}: test accuracy {test_acc:.4f} "
          + (f"(trained on folds {train_folds}, last epoch {best_epoch})" if args.no_val
             else f"(val fold {val_fold}, epoch {best_epoch})"))
    if args.save_models:
        torch.save(best_state, args.out / f"model_fold{test_fold}.pt")
    return np.flatnonzero(te), prob, hist


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True, help="experiment name")
    ap.add_argument("--model-file", type=Path, default=PROJECT / "models" / "model_baseline.py",
                    help="Python file defining build_model(in_channels, n_classes)")
    ap.add_argument("--out", type=Path, help="output folder (default results/<run>/)")
    ap.add_argument("--features", default="logmel", choices=FEATURE_CHOICES, help="input representation")
    ap.add_argument("--folds", type=int, nargs="+", default=[1, 2, 3, 4, 5], help="test folds to run")
    ap.add_argument("--epochs", type=int, default=80)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--weight-decay", type=float, default=1e-3)
    ap.add_argument("--optimizer", default="adamw", choices=["adamw", "sgd"])
    ap.add_argument("--momentum", type=float, default=0.9, help="SGD Nesterov momentum")
    ap.add_argument("--scheduler", default="onecycle", choices=["onecycle", "cosine", "sgdr", "step"],
                    help="learning-rate schedule (see make_scheduler)")
    ap.add_argument("--sgdr-t0", type=int, default=20, help="sgdr: first cycle length in epochs")
    ap.add_argument("--sgdr-mult", type=int, default=2, help="sgdr: each cycle is this many times longer")
    ap.add_argument("--step-epochs", type=int, default=100, help="step: epochs between decays")
    ap.add_argument("--step-gamma", type=float, default=0.1, help="step: decay factor")
    ap.add_argument("--spec-augment", action="store_true", help="SpecAugment time/frequency masking (training only)")
    ap.add_argument("--n-masks", type=int, default=2, help="SpecAugment: frequency masks and time masks per clip")
    ap.add_argument("--mask-freq-width", type=int, default=16, help="SpecAugment: widest frequency mask (bands)")
    ap.add_argument("--mask-time-width", type=int, default=24, help="SpecAugment: widest time mask (frames)")
    ap.add_argument("--crop-frames", type=int, default=0,
                    help="sub-segment training on random windows of this many frames, window-averaged evaluation")
    ap.add_argument("--no-val", action="store_true",
                    help="final evaluation: train on all 4 other folds for --epochs, test the last epoch (no early stopping)")
    ap.add_argument("--label-smoothing", type=float, default=0.0, help="training loss label smoothing; 0 = off")
    ap.add_argument("--crop-hop", type=int, default=0, help="evaluation window hop (default crop-frames / 2)")
    ap.add_argument("--mixup", type=float, default=0.0, help="mixup alpha; 0 = off")
    ap.add_argument("--wave-aug", action="store_true", help="offline pitch / stretch / noise copies (training only)")
    ap.add_argument("--time-shift", action="store_true", help="random circular time shift (training only)")
    ap.add_argument("--all-aug", action="store_true", help="= --wave-aug --time-shift --spec-augment --mixup 0.4")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--save-models", action="store_true")
    ap.add_argument("--curve-every", type=int, default=0,
                    help="learning curves: every N epochs also evaluate the training folds without augmentation and, "
                         "with --no-val, the test fold (recorded only, never used for a choice); 0 = off")
    args = ap.parse_args(argv)
    args.crop_hop = args.crop_hop or args.crop_frames // 2
    if args.all_aug:
        args.wave_aug = args.time_shift = args.spec_augment = True
        args.mixup = args.mixup or 0.4
    args.out = PROJECT / (args.out or Path("results") / args.run)  # relative paths are relative to the repository
    args.model_file = PROJECT / args.model_file
    args.build_model = load_model_builder(args.model_file)
    args.out.mkdir(parents=True, exist_ok=True)
    sys.stdout = Tee(args.out / "train.log")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if args.wave_aug and args.features not in ("logmel", "logmel_delta"):
        raise SystemExit("the offline waveform copies exist for log-mel (and its delta) only")
    data, feat_name = load_features(args.features)
    args.in_channels = data["X"].shape[1] if data["X"].ndim == 4 else 1
    if args.wave_aug:
        from .augment import AugConfig, load_aug_cache
        data["X_aug"] = load_aug_cache(FeatureConfig(), AugConfig())
        if args.features == "logmel_delta":  # each copy gets its delta channel exactly like the original clips
            data["X_aug"] = with_delta(data["X_aug"])
    n_params = count_params(args.build_model(in_channels=args.in_channels))
    model_file = rel(args.model_file)
    print(f"{args.run}: {model_file}, {n_params / 1e6:.2f}M parameters, device {device}, folds {args.folds}")
    config = {k: (str(v) if isinstance(v, Path) else v) for k, v in vars(args).items() if k != "build_model"}
    config.update(features=feat_name, n_params=n_params, model_file=model_file, out=rel(args.out))
    if args.wave_aug:
        from dataclasses import asdict
        from .augment import AugConfig
        config["wave_aug_config"] = asdict(AugConfig())
    (args.out / "config.json").write_text(json.dumps(config, indent=2))

    t0 = time.time()
    idx_all, prob_all, history = [], [], {}
    for k in args.folds:
        idx, prob, hist = train_fold(args, data, k, device)
        idx_all.append(idx)
        prob_all.append(prob)
        history[str(k)] = hist
    idx = np.concatenate(idx_all)
    np.savez_compressed(args.out / "predictions.npz", y_true=data["y"][idx], y_prob=np.concatenate(prob_all),
                        fold=data["fold"][idx], filename=data["filename"][idx])
    (args.out / "history.json").write_text(json.dumps(history))
    print(f"training took {(time.time() - t0) / 60:.1f} min")
    evaluate_run(args.out)


if __name__ == "__main__":
    main()
