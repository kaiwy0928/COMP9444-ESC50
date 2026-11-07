# COMP9444 — Environmental Sound Classification on ESC-50

Fine-tuning an **Audio Spectrogram Transformer (AST)** on ESC-50 with the team's shared pipeline.
Final result: **90.13 ± 0.85%** five-fold accuracy over three seeds with **86.23M parameters** —
meeting both hard requirements (params > 1M, accuracy > 90%).

## Final results (5-fold CV, 3 seeds, 15 runs)

| Seed | Fold 1 | Fold 2 | Fold 3 | Fold 4 | Fold 5 | Mean |
|---|---|---|---|---|---|---|
| 42 | 89.25 | 90.75 | 86.25 | 91.50 | 88.25 | 89.20% |
| 43 | 90.50 | 92.00 | 89.50 | 92.75 | 89.50 | 90.85% |
| 44 | 92.25 | 90.00 | 89.25 | 92.75 | 87.50 | 90.35% |
| **all** | | | | | | **90.13 ± 0.85%** |

Pooled over the three seeds: macro-F1 **0.8998**, Cohen's κ **0.8993**, top-3 **97.87%**, top-5 **98.70%**.
Standard deviation is across the three seed-level five-fold means (not over independent datasets).

## Recipe

AST (DeiT ViT-B, AudioSet-pretrained `MIT/ast-finetuned-audioset-10-10-0.4593`, 86.23M params),
40 epochs, AdamW, lr 1e-4, weight decay 1e-2, label smoothing 0.1, batch 32, OneCycle,
SpecAugment (2 masks, freq 36 / time 24) + mixup 0.2 + **waveform time-shift**.

## Experiment history (development runs, fold 1, seed 42 unless noted)

| Model / setting | Accuracy | Note |
|---|---|---|
| PANNs CNN14, 3 ep (smoke) | 27.00% | pipeline check (random = 2%) |
| PANNs CNN14, 40 ep, wd 5e-3 | 79.75% | overfits: clean train acc 98.9% |
| PANNs CNN14 + 3 s crops, 40 ep | 75.75% | stronger augmentation starves a 40-epoch budget |
| EfficientNet-B0, 60 ep | 88.00% | best ImageNet-pretrained CNN |
| MobileNetV3-Small, 450 ep | 83.00% | |
| AST, 20 ep | 88.50% | |
| AST, 40 ep | 89.25% | |
| AST + time-shift, 40 ep | 90.25% | final recipe, single-fold check |
| AST + time-shift, fold 5 only | 88.00% | recipe stability check |
| **AST + time-shift, 15 runs** | **90.13 ± 0.85%** | **final** |

## What is here

- `run_model.py`, `src/` — shared training/evaluation pipeline (5-fold CV, SpecAugment, mixup, TTA)
- `models/model_ast.py` — **our AST adaptation**: keeps the shared 22.05 kHz log-mel bands, fuses
  log-mel/delta channels with a 1×1 conv (initialised to average), bicubically interpolates the
  pretrained 12×101 position embeddings to the 12×21 grid of 216-frame inputs
- `models/model_panns_cnn14.py` etc. — comparison models
- `model_results/` — metrics.json, history.json, confusion matrices, figures, logs for every run;
  the 15 final runs are `model_results/model_ast/smoke_final_seed{42,43,44}/`
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
