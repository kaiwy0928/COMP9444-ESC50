#!/usr/bin/env bash
# Single-fold hyper-parameter sweep. One fold x 40 epochs per point, so a
# 24 GB card clears the whole sweep well inside an hour.
#
# Purpose: find which regularisation strength actually clears 90% before
# spending time on the full 5-fold x 3-seed grid.
#
#     nohup bash cloud/run_sweep.sh > cloud/sweep.log 2>&1 &
#     tail -f cloud/sweep.log
set -uo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

mkdir -p cloud
STAMP=$(date +%Y%m%d_%H%M%S)
LOG="cloud/sweep_${STAMP}.log"
OUT="model_results/model_panns_cnn14"
echo "== sweep start $(date '+%F %T')" | tee -a "$LOG"

# --folds 1 always writes to ${OUT}/smoke_final_seed42, so each point is moved
# aside before the next one starts; otherwise they overwrite each other.
run () {   # $1=tag  $2=lr  $3=weight-decay  $4=label-smoothing
    local tag="$1" lr="$2" wd="$3" ls="$4"
    echo "---- $tag  lr=$lr wd=$wd ls=$ls  $(date '+%T')" | tee -a "$LOG"

    python -u run_model.py models/model_panns_cnn14.py \
        --folds 1 --seeds 42 --epochs 40 \
        --lr "$lr" --weight-decay "$wd" --label-smoothing "$ls" \
        2>&1 | tee -a "$LOG" | grep -aE "test accuracy" | tail -1

    if [ -d "${OUT}/smoke_final_seed42" ]; then
        rm -rf "${OUT}/sweep_${tag}"
        mv "${OUT}/smoke_final_seed42" "${OUT}/sweep_${tag}"
        echo "   -> ${OUT}/sweep_${tag}" | tee -a "$LOG"
    fi
}

# a: the recipe that was interrupted on the local CPU box
run reg_mid     4e-4 5e-3 0.1
# b: milder regularisation, keeps more capacity
run reg_light   4e-4 1e-3 0.05
# c: heavier regularisation
run reg_heavy   3e-4 1e-2 0.15
# d: lower learning rate, same regularisation
run reg_lowlr   2e-4 5e-3 0.1
# e: the original 0.8325 baseline, for a like-for-like reference
run baseline    4e-4 1e-3 0.0

echo "== sweep done $(date '+%F %T')" | tee -a "$LOG"
echo "Pick the best tag and copy its lr / wd / ls into cloud/run_grid.sh" | tee -a "$LOG"
