#!/bin/bash
#SBATCH --mem=2G
#SBATCH --time=0-00:10:00
#SBATCH --output=logs/%x_%A_%a.out
# Set your HPC allocation account before submitting, e.g.:
#   export SLURM_ACCOUNT=def-yourpi; sbatch --account="$SLURM_ACCOUNT" ...
# (left unset here rather than hardcoded to any specific account)

# Unified LFC generation script - both the SLURM array-task body AND its own
# submission-loop driver, replacing the project's original 7 ad hoc wrapper
# scripts (run.sh, submit_all.sh, submit_all_ws.sh, submit_all_50_99.sh,
# submit_all_ws_50_99.sh, plus two 100_seed_nets/-local k816/krest variants).
#
# One array task = one (k, p) combination generated for a single seed and
# model type, calling generate_lfc.py to produce exactly one .gt file.
#
# Usage (submission loop - NOT via sbatch, just `bash`):
#   bash submit_lfc_generate.sh <model_type: mhk|ws|ke> <seed_start> <seed_end> [k_values...]
#   e.g. bash submit_lfc_generate.sh mhk 0 99 2 4 6 8 10 12 14 16 18 20 22 24 26 28 30 32
#   e.g. bash submit_lfc_generate.sh ke 1 1 8 10 12   # single-seed baseline sweep
#
# This submits one throttled, chunked sbatch array per seed (via `sbatch
# --export=... "$0"`), so SLURM re-invokes this same file as the array task
# below once for each (k, p) combination.
#
# Optional env vars: NODES (default 240), N_P (default 100 p-values per k),
# CHUNK_SIZE (default 400 tasks/sbatch call), MAX_IN_QUEUE (default 900).

set -euo pipefail

if [ -n "${SLURM_ARRAY_TASK_ID:-}" ]; then
    # ------------------------- SLURM array-task body -------------------------
    module load StdEnv/2020 gcc/9.3.0 graph-tool/2.56 python/3.10

    N_P=${N_P:-100}
    NODES=${NODES:-240}
    TASK_ID=$SLURM_ARRAY_TASK_ID
    P_IDX=$(( TASK_ID % N_P ))
    K_IDX=$(( TASK_ID / N_P ))

    read -ra K_ARR <<< "$K_VALUES"
    K=${K_ARR[$K_IDX]}

    # p sweep matches np.linspace(0.001, 1, 100) from the original generation script
    P=$(python3 -c "import numpy as np; print(np.linspace(0.001, 1, $N_P)[$P_IDX])")

    echo "seed=$SEED model_type=$MODEL_TYPE nodes=$NODES k=$K p=$P (array task $TASK_ID)"

    SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
    python "$SCRIPT_DIR/generate_lfc.py" --seed "$SEED" --model_type "$MODEL_TYPE" --k "$K" --p "$P" --nodes "$NODES"
    exit 0
fi

# ------------------------------ submission loop ------------------------------
MODEL_TYPE=${1:?"usage: bash submit_lfc_generate.sh <model_type> <seed_start> <seed_end> [k_values...]"}
SEED_START=${2:?}
SEED_END=${3:?}
shift 3
K_VALUES_ARGS=("$@")
if [ ${#K_VALUES_ARGS[@]} -eq 0 ]; then
    K_VALUES_ARGS=(2 4 6 8 10 12 14 16 18 20 22 24 26 28 30 32)
fi
K_VALUES="${K_VALUES_ARGS[*]}"

NODES=${NODES:-240}
N_P=${N_P:-100}
CHUNK_SIZE=${CHUNK_SIZE:-400}
MAX_IN_QUEUE=${MAX_IN_QUEUE:-900}
POLL_INTERVAL=${POLL_INTERVAL:-30}
TOTAL_TASKS=$(( ${#K_VALUES_ARGS[@]} * N_P ))

mkdir -p logs

current_queue_depth() {
    # -r expands pending array jobs into individual task lines, matching what
    # actually counts toward MaxSubmitJobsPerUser (a compact "[0-399]" range
    # would otherwise undercount).
    squeue -u "$USER" -r -h -o "%i" | wc -l
}

SELF="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/$(basename "${BASH_SOURCE[0]}")"

for SEED in $(seq "$SEED_START" "$SEED_END"); do
    START=0
    while [ "$START" -lt "$TOTAL_TASKS" ]; do
        END=$(( START + CHUNK_SIZE - 1 ))
        if [ "$END" -ge "$TOTAL_TASKS" ]; then
            END=$(( TOTAL_TASKS - 1 ))
        fi
        CHUNK_LEN=$(( END - START + 1 ))

        while [ "$(( $(current_queue_depth) + CHUNK_LEN ))" -gt "$MAX_IN_QUEUE" ]; do
            echo "Queue near cap, waiting ${POLL_INTERVAL}s before submitting model_type=${MODEL_TYPE} seed=${SEED} array=${START}-${END} ..."
            sleep "$POLL_INTERVAL"
        done

        sbatch --job-name="LFC_${MODEL_TYPE}_seed${SEED}" --array="${START}-${END}%50" \
            --export="SEED=${SEED},MODEL_TYPE=${MODEL_TYPE},NODES=${NODES},N_P=${N_P},K_VALUES=${K_VALUES}" \
            "$SELF"
        echo "Submitted model_type=${MODEL_TYPE} seed=${SEED} array=${START}-${END} (${CHUNK_LEN} tasks)"

        START=$(( END + 1 ))
    done
done

echo "All submissions complete."
