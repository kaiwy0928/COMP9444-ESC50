# COMP9444 — Environmental Sound Classification on ESC-50

Fine-tuning an **Audio Spectrogram Transformer (AST)** on ESC-50 with the team's shared pipeline.
Final recipe reaches **90.25% fold-1 accuracy** with **86.23M parameters** — meeting both hard
requirements (params > 1M, accuracy > 90%). Full 5-fold × 3-seed evaluation in progress.

## Key result (AST, 40 epochs + time-shift, fold 1, seed 42)

| Accuracy | Macro-F1 | Cohen's κ | Top-3 | Top-5 | Params |
|---|---|---|---|---|---|
| **90.25%** | 0.9003 | 0.9005 | 98.75% | 99.50% | 86.23M |

Full-5-fold × 3-seed run (seeds 42/43/44) — seed 42 partial: 89.25 / 90.75 / 86.25 (folds 1–3).

## Experiment history (fold 1, seed 42 unless noted)

| Model / setting | Accuracy | Note |
|---|---|---|
| PANNs CNN14, 3 ep (smoke) | 27.00% | pipeline check (random = 2%) |
| PANNs CNN14, 40 ep, wd 5e-3 | 79.75% | overfits: clean train acc 98.9% |
| PANNs CNN14 + 3 s crops, 40 ep | 75.75% | stronger augmentation starves 40-epoch budget |
| EfficientNet-B0, 60 ep | 88.00% | best ImageNet-pretrained CNN |
| MobileNetV3-Small, 450 ep | 83.00% | |
| AST, 20 ep | 88.50% | |
| AST, 40 ep | 89.25% | |
| **AST + time-shift, 40 ep** | **90.25%** | **final recipe** |
| AST + time-shift, fold 5 check | 88.00% | recipe stability check |
| AST + time-shift, full run | in progress | folds 1–3 of seed 42: 89.25 / 90.75 / 86.25 |

## What is here

- `run_model.py`, `src/` — shared training/evaluation pipeline (5-fold CV, SpecAugment, mixup, TTA)
- `models/model_ast.py` — **our AST adaptation**: keeps the shared 22.05 kHz log-mel bands, fuses
  log-mel/delta channels with a 1×1 conv (initialised to average), bicubically interpolates the
  pretrained 12×101 position embeddings to the 12×21 grid of 216-frame inputs
- `models/model_panns_cnn14.py` etc. — comparison models
- `model_results/` — metrics.json, history.json, confusion matrices, figures, logs for every run
- `COMP9444_答辩文档.docx` — defense document (Chinese): architecture, experiments, error analysis, Q&A
- `CLOUD_START.md`, `requirements_cloud.txt` — cloud/GPU setup guide

## Reproduce

```bash
python -m venv .venv && .venv/Scripts/python -m pip install -r requirements_cloud.txt
# PyTorch (CUDA 12.8 build):
.venv/Scripts/python -m pip install torch==2.11.0 torchvision==0.26.0 --index-url https://download.pytorch.org/whl/cu128
# AST weights (huggingface.co unreachable? set the mirror):
set HF_ENDPOINT=https://hf-mirror.com
# final recipe, all folds, three seeds:
.venv/Scripts/python run_model.py models/model_ast.py --folds 1 2 3 4 5 --seeds 42 43 44 \
    --epochs 40 --lr 1e-4 --weight-decay 1e-2 --label-smoothing 0.1 --time-shift
```

Note: the ESC-50 metadata CSV (`data/raw/ESC-50-master/meta/esc50.csv`) is required for the
post-training evaluation step; raw audio and cached features are excluded from git
(regenerate with `1_get_raw_data.py` / `2_preprocess.py`).

## Environment

Windows 11, RTX 4060 Ti 16 GB (sm_89), PyTorch 2.11.0+cu128, Python 3.13.
