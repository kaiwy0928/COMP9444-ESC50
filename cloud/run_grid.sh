#!/usr/bin/env bash
# Full result grid: 5 folds x 3 seeds, 40 epochs, the regularised recipe.
#
# Run it detached so an SSH disconnect cannot kill it:
#     nohup bash cloud/run_grid.sh > cloud/grid.log 2>&1 &
#     tail -f cloud/grid.log
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

mkdir -p cloud
STAMP=$(date +%Y%m%d_%H%M%S)
LOG="cloud/grid_${STAMP}.log"

echo "== full grid start $(date '+%F %T')" | tee -a "$LOG"

# 40 epochs, stronger regularisation than the 0.8325 baseline:
# dropout 0.5 lives in the model file, plus weight decay and label smoothing.
# --protocol final = standard 5-fold CV; the three seeds give mean +/- spread.
python -u run_model.py models/model_panns_cnn14.py \
    --protocol final \
    --seeds 42 43 44 \
    --epochs 40 \
    --lr 4e-4 \
    --weight-decay 5e-3 \
    --label-smoothing 0.1 \
    2>&1 | tee -a "$LOG"

echo "== full grid done $(date '+%F %T')" | tee -a "$LOG"

# Per-fold and aggregate numbers, straight to the console.
python scripts/summarize_models.py 2>&1 | tee -a "$LOG" || true
