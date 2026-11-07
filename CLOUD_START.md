# CLOUD_START - running COMP9444 on the rented GPU box

Everything needed is inside `COMP9444_cloud_bundle.zip`. Raw ESC-50 audio is
not included: the two cached feature arrays already hold the preprocessed
log-mel spectrograms and their deltas, which is all training reads.

Target hardware in this note: **RTX 5060 Ti (Blackwell, sm_120, 16 GB)**.

---

## 0. Why the CUDA build matters here

The 5060 Ti is compute capability **sm_120**. A plain `pip install torch` gives
a build compiled for older architectures and PyTorch will silently fall back to
the CPU. It must be **CUDA 12.8 or newer**:

| item | requirement |
|---|---|
| NVIDIA driver | >= 570.65 |
| PyTorch | >= 2.7 with cu128 (2.11.0 validated here) |
| CUDA runtime | 12.8 |

Check the driver first:

```bat
nvidia-smi
```

The top-right line must read `CUDA Version: 12.8` or higher. If it shows 12.7
or lower, update the driver before anything else.

---

## 1. Unpack

Unzip `COMP9444_cloud_bundle.zip` into a short path with no spaces, e.g.

```
D:\comp9444
```

The resulting layout is:

```
D:\comp9444\
├── run_model.py                 training entry point
├── src\                         library code
├── models\model_panns_cnn14.py  the model being evaluated
├── models\pretrained\Cnn14_mAP=0.431.pth
├── scripts\                     helpers (study, figures, summary)
├── cloud\                       cloud helper scripts
├── data\processed\              cached features (log-mel + delta)
└── requirements.txt
```

---

## 2. Environment

```bat
conda create -n comp9444 python=3.11 -y
conda activate comp9444
python -m pip install --upgrade pip
python -m pip install torch==2.11.0 torchvision==0.26.0 --index-url https://download.pytorch.org/whl/cu128
python -m pip install -r requirements_cloud.txt
```

**Use `requirements_cloud.txt`, not `requirements.txt`.** The latter pins
`torch==2.11.0` with no index URL, so pip resolves it from PyPI to the
CPU-only build and quietly replaces the cu128 build installed on the line
above. `requirements_cloud.txt` is the identical list with the two torch lines
removed; it ships inside the bundle. Symptom if this goes wrong: the GPU is
detected by `nvidia-smi` but training prints `device cpu`.

Verify (must print `True` and the GPU name):

```bat
python -c "import torch; print(torch.__version__, torch.cuda.is_available(), torch.cuda.get_device_name(0))"
```

---

## 3. Smoke test - 3 epochs, one fold

Confirms the data, the checkpoint and the GPU path all work before committing
to a long run.

```bat
python run_model.py models/model_panns_cnn14.py --folds 1 --seeds 42 --epochs 3 --lr 4e-4
```

Expected: `[panns_cnn14] loaded 79 tensors / 79.7M parameters`, `device cuda`,
and a test accuracy around 40-60% after three epochs. Anything in the 2% range
means the images/features are not being read correctly.

---

## 4. Pick the regularisation setting

The previous local run peaked at **0.8325** with full fine-tuning and
overfitted (train 100% / test 83%), so the next step is finding a recipe that
clears 90%.

```bat
python run_model.py models/model_panns_cnn14.py --folds 1 --seeds 42 --epochs 40 --lr 4e-4 --weight-decay 5e-3 --label-smoothing 0.1
```

Each such run is one fold x 40 epochs (about 4-6 minutes on a 5060 Ti). Compare
a few settings by changing `--weight-decay`, `--label-smoothing` and `--lr`:

| tag | lr | weight decay | label smoothing |
|---|---|---|---|
| reg_light | 4e-4 | 1e-3 | 0.05 |
| reg_mid | 4e-4 | 5e-3 | 0.1 |
| reg_heavy | 3e-4 | 1e-2 | 0.15 |
| baseline | 4e-4 | 1e-3 | 0.0 |

Always judge these on the **same fold and the same seed**; comparing a fold-1
number with a fold-3 number is meaningless.

---

## 5. Full grid - 5 folds x 3 seeds

Only once a setting clears 90% on the single fold. The protocol name `final`
runs the standard 5-fold cross-validation.

```bat
python run_model.py models/model_panns_cnn14.py --protocol final --seeds 42 43 44 --epochs 40 --lr 4e-4 --weight-decay 5e-3 --label-smoothing 0.1
```

Then the aggregate table:

```bat
python scripts/summarize_models.py
```

On a 5060 Ti the full grid is roughly 40-70 minutes.

**Run it detached** so a dropped remote-desktop session cannot kill it:

```bat
start /b cmd /c "python run_model.py ... > grid.log 2>&1"
```

or in PowerShell:

```powershell
Start-Process -NoNewWindow python -ArgumentList "run_model.py","models/model_panns_cnn14.py","--protocol","final","--seeds","42","43","44","--epochs","40","--lr","4e-4","--weight-decay","5e-3","--label-smoothing","0.1" -RedirectStandardOutput grid.log
```

---

## 6. What to copy back

Only the results matter, and they are small:

```
model_results\model_panns_cnn14\final_seed42\
final_seed43\
final_seed44\
```

Each holds `metrics.json`, `history.json`, `train.log` and figures. Zip those
three folders and bring them back - a few MB in total.

Then on the local machine:

```bat
python scripts/summarize_models.py
python 4_make_results.py
```

---

## 7. Expected numbers, so you know if something is wrong

| stage | reference |
|---|---|
| 3-epoch smoke test | 40-60% test accuracy |
| 40-epoch single fold, no regularisation | 83.25% (measured locally) |
| 40-epoch single fold, with regularisation | target 88-92% |
| full grid, 5 folds x 3 seeds | mean of the above |

If the smoke test prints `device cpu`, the CUDA build is wrong - go back to
section 0. Do not continue on the CPU: 15 runs would take about two days.
