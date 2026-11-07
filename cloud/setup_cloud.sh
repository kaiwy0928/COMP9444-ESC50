#!/usr/bin/env bash
# One-shot setup for the cloud GPU instance (Linux shell, e.g. Git Bash on a
# Windows remote desktop).
#
# Usage:  bash cloud/setup_cloud.sh
#
# IMPORTANT: never run `pip install -r requirements.txt` on the cloud box.
# That file pins torch==2.11.0 without an index URL, so pip resolves it from
# PyPI to the CPU-only build and silently replaces the CUDA build that the
# RTX 5060 Ti (sm_120) needs. requirements_cloud.txt is the same list with the
# torch lines stripped out.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
echo "== project root: $ROOT"

if command -v python >/dev/null 2>&1; then PY=python; else PY=python3; fi
"$PY" - <<'EOF'
import sys
print("python", sys.version.split()[0])
if sys.version_info[:2] != (3, 11):
    print("NOTE: 3.11 is what this project was validated on")
EOF

# ---------------------------------------------------------------------------
# 1. Dependencies, torch excluded on purpose (see the header).
# ---------------------------------------------------------------------------
if [ ! -f requirements_cloud.txt ]; then
    grep -vE '^(torch|torchvision)==' requirements.txt > requirements_cloud.txt
fi
"$PY" -m pip install -q --upgrade pip
"$PY" -m pip install -q -r requirements_cloud.txt

# ---------------------------------------------------------------------------
# 2. GPU check. Blackwell needs a cu128 build; anything older falls back to CPU
#    without raising an error, which is the most common way this project stalls.
# ---------------------------------------------------------------------------
"$PY" - <<'EOF'
import torch
print("torch", torch.__version__, "| built for cuda", torch.version.cuda,
      "| available", torch.cuda.is_available())
if not torch.cuda.is_available():
    raise SystemExit(
        "CUDA NOT AVAILABLE. On a 5060 Ti (sm_120) you need torch>=2.7 built for "
        "cu128 and a driver >= 570.65. Reinstall with:\n"
        "  pip install torch==2.11.0 torchvision==0.26.0 "
        "--index-url https://download.pytorch.org/whl/cu128"
    )
p = torch.cuda.get_device_properties(0)
print(f"device {p.name}  {p.total_memory / 2**30:.1f} GiB  sm_{p.major}{p.minor}")
free, total = torch.cuda.mem_get_info()
print(f"free VRAM {free / 2**30:.1f} / {total / 2**30:.1f} GiB")
EOF

# ---------------------------------------------------------------------------
# 3. Artefacts: two cached feature arrays plus the PANNs checkpoint.
# ---------------------------------------------------------------------------
"$PY" - <<'EOF'
from pathlib import Path
need = {
    "data/processed/logmel/logmel_sr22050_m128_fft1024_hop512.npz": 150e6,
    "data/processed/delta/logmel_sr22050_m128_fft1024_hop512_delta9.npz": 150e6,
    "models/pretrained/Cnn14_mAP=0.431.pth": 300e6,
}
ok = True
for rel, min_size in need.items():
    p = Path(rel)
    if not p.exists():
        print(f"MISSING   {rel}"); ok = False
    elif p.stat().st_size < min_size:
        print(f"TRUNCATED {rel} ({p.stat().st_size/1e6:.0f} MB)"); ok = False
    else:
        print(f"ok  {p.stat().st_size/1e6:7.0f} MB  {rel}")
if not ok:
    raise SystemExit("artefacts incomplete - re-upload before training")
EOF

echo
echo "== setup done."
echo "== next: python run_model.py models/model_panns_cnn14.py --folds 1 --seeds 42 --epochs 3 --lr 4e-4"
