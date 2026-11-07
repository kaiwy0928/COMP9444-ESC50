# models/

One file per model: `model_<name>.py`, e.g. `model_resnet18.py`. This is the only folder where you add code.

Every file defines `build_model(in_channels, n_classes)` and returns a `torch.nn.Module` that maps
`(batch, in_channels, 128, 216)` spectrograms to `(batch, n_classes)` logits (`in_channels` = 2: log-mel + delta,
`n_classes` = 50).

| File | Model | Parameters | 5-fold CV, 3 seeds |
|---|---|---|---|
| `model_baseline.py` | baseline CRNN (CNN + BiLSTM) | 1.25M | 86.85 ± 0.58% |
| `model_multiscale_cnn.py` | our final model: multi-scale CNN + SE, no BiLSTM | 0.84M | 88.77 ± 0.40% |
| `model_mobilenetv3_small.py` | MobileNetV3-Small, ImageNet-pretrained | 1.57M | 83.53 ± 0.49% |
| `model_efficientnet_b0.py` | EfficientNet-B0, ImageNet-pretrained | 4.07M | 87.27 ± 0.81% |
| `model_resnet18.py` | ResNet-18, ImageNet-pretrained | 11.20M | 83.07 ± 0.55% |

Train and evaluate a model with the common recipe, then put all models in one table:

    python run_model.py models/model_<name>.py --folds 1 --epochs 3    # smoke test
    python run_model.py models/model_<name>.py --seeds 42 43 44        # results -> model_results/model_<name>/
    python scripts/summarize_models.py

Full steps: `GUIDE.md`, Part 2.

---

每个模型一个文件 `model_<名字>.py`（例如 `model_resnet18.py`），这是唯一需要写代码的文件夹。文件里定义 `build_model(in_channels, n_classes)`，输入 `(batch, 2, 128, 216)` 的谱图，输出 `(batch, 50)`。完整步骤见 `GUIDE.md` 第二部分。
