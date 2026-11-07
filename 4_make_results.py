"""Step 4: the results of the report from model_results/ (nothing is trained; about a minute).

- one table of all five models: test accuracy (mean +- SD over seeds 42/43/44), macro F1, parameters
  -> printed and saved to model_results/summary.csv
- for every model: learning curves and per-class F1 over its three seeds -> results/report_figures/

Works straight after downloading the repository (our results are in model_results/), or after step 3 with yours.
Click Run, or: python 4_make_results.py
"""
from src.runner import ROOT, run

# model file name -> title on the figures
MODELS = {"baseline": "Baseline CRNN", "multiscale_cnn": "Multi-scale CNN + SE (ours)",
          "mobilenetv3_small": "MobileNetV3-Small", "efficientnet_b0": "EfficientNet-B0",
          "resnet18": "ResNet-18", "panns_cnn14": "PANNs CNN14 (AudioSet pretrained)"}

if __name__ == "__main__":
    run("scripts/summarize_models.py")
    for name, title in MODELS.items():
        runs = sorted((ROOT / "model_results" / f"model_{name}").glob("final_seed*"))
        if not runs:
            print(f"(no final runs for {name} yet: train it with 3_train_models.py)")
            continue
        run("scripts/plot_report_figures.py", "--name", name, "--title", title,
            "--runs", *[r.relative_to(ROOT).as_posix() for r in runs])
    print(f"\nStep 4 finished: table in model_results/summary.csv, figures in {ROOT / 'results' / 'report_figures'}")
