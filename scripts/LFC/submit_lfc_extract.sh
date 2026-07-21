#!/bin/bash
#SBATCH --mem=2G
#SBATCH --time=0-00:20:00
#SBATCH --output=logs/%x_%A_%a.out
#SBATCH --job-name=LFC_extract
# Set your HPC allocation account before submitting, e.g.:
#   export SLURM_ACCOUNT=def-yourpi; sbatch --account="$SLURM_ACCOUNT" ...

# Unified LFC extraction script - both the SLURM array-task body AND its own
# submission driver, replacing submit_props_csv_gen.sh and the
# 100_seed_nets/-local submit_props_csv_gen_{k816,mhk,ws}.sh variants.
#
# One array task = extract_lfc_csv.py restricted to a single k (all model
# types under that k, via --root nets/LFC/<nodes>).
#
# Usage (submission driver - NOT via sbatch, just `bash`):
#   bash submit_lfc_extract.sh [nodes] [k_values...]
#   e.g. bash submit_lfc_extract.sh 240 2 4 6 8 10 12 14 16 18 20 22 24 26 28 30 32

set -euo pipefail

if [ -n "${SLURM_ARRAY_TASK_ID:-}" ]; then
    module load StdEnv/2020 gcc/9.3.0 graph-tool/2.56 python/3.10

    read -ra K_ARR <<< "$K_VALUES"
    K=${K_ARR[$SLURM_ARRAY_TASK_ID]}

    echo "nodes=$NODES k=$K (array task $SLURM_ARRAY_TASK_ID)"

    SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
    python "$SCRIPT_DIR/extract_lfc_csv.py" --root "nets/LFC/${NODES}" --k "$K"
    exit 0
fi

NODES=${1:-240}
shift || true
K_VALUES_ARGS=("$@")
if [ ${#K_VALUES_ARGS[@]} -eq 0 ]; then
    K_VALUES_ARGS=(2 4 6 8 10 12 14 16 18 20 22 24 26 28 30 32)
fi
K_VALUES="${K_VALUES_ARGS[*]}"
LAST_IDX=$(( ${#K_VALUES_ARGS[@]} - 1 ))

mkdir -p logs

SELF="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/$(basename "${BASH_SOURCE[0]}")"
sbatch --array="0-${LAST_IDX}" --export="NODES=${NODES},K_VALUES=${K_VALUES}" "$SELF"
echo "Submitted extraction for nodes=${NODES}, k in [${K_VALUES}]"
