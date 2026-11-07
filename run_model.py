r"""Train and evaluate one model file with the common recipe, so different models are compared under the same
conditions (GUIDE.md, Parts 2 and 3). Only the model changes; the data, folds, features, augmentation and evaluation are fixed.

Evaluation = the standard ESC-50 5-fold cross-validation: for each test fold, train on the other four folds for a
fixed number of epochs and test the last epoch once (the test fold never chooses anything). Run three seeds.

    python run_model.py models/model_resnet18.py --seeds 42 43 44        # the numbers for the report
    python run_model.py models/model_resnet18.py --folds 1 --epochs 3    # quick smoke test

Optional, for choosing fine-tuning settings without touching the test folds: --protocol dev trains on three folds and
uses the fourth as validation (it picks the epoch); compare settings on its validation accuracy.

Results: model_results/<model file name>/<protocol>_seed<seed>/ (metrics.json, predictions.npz, train.log, figures);
runs with --folds go to smoke_<protocol>_seed<seed>/ and are left out of scripts/summarize_models.py.
Report figures (learning curves, per-class F1) from the seeds of one model:
    python scripts/plot_report_figures.py --name resnet18 --title "ResNet-18" \
        --runs model_results/model_resnet18/final_seed42 model_results/model_resnet18/final_seed43 \
               model_results/model_resnet18/final_seed44
Fine-tuning settings are yours to choose: --epochs, --lr, --weight-decay, and any option of src/train.py after "--",
e.g.  python run_model.py models/model_x.py -- --optimizer sgd --momentum 0.9
"""
import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
from src.train import main as train  # noqa: E402

# the common recipe: augmentation of our best model (SpecAugment Policy2 of Zhang et al. 2019 + mixup)
RECIPE = ["--features", "logmel_delta", "--batch-size", "32",
          "--spec-augment", "--n-masks", "2", "--mask-freq-width", "36", "--mask-time-width", "24",
          "--mixup", "0.2",
          "--curve-every", "10"]  # learning curves (training folds without augmentation, test fold); changes nothing


def main(argv: list[str] | None = None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("model", help="model file, e.g. models/model_resnet18.py")
    ap.add_argument("--protocol", choices=["final", "dev"], default="final",
                    help="final = standard 5-fold CV (4 training folds, last epoch tested); dev = optional tuning run")
    ap.add_argument("--seeds", type=int, nargs="+", default=[42])
    ap.add_argument("--epochs", type=int, default=450)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--weight-decay", type=float, default=1e-3)
    ap.add_argument("--folds", type=int, nargs="+", default=None, help="only these test folds (smoke tests)")
    args, extra = ap.parse_known_args(argv)
    extra = [a for a in extra if a != "--"]

    model = Path(args.model).resolve()
    name = model.stem
    for seed in args.seeds:
        tag = f"smoke_{args.protocol}" if args.folds else args.protocol  # partial runs never enter the summary
        out = ROOT / "model_results" / name / f"{tag}_seed{seed}"
        argv = ["--run", f"{name}_{args.protocol}_seed{seed}", "--model-file", str(model), "--out", str(out),
                "--epochs", str(args.epochs), "--lr", str(args.lr), "--weight-decay", str(args.weight_decay),
                "--seed", str(seed), *RECIPE, *(["--no-val"] if args.protocol == "final" else []),
                *(["--folds", *map(str, args.folds)] if args.folds else []), *extra]
        train(argv)
        sys.stdout = sys.__stdout__  # src/train.py copies the console into this run's train.log


if __name__ == "__main__":
    main()
