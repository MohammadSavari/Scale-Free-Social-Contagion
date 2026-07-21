#!/bin/bash
#SBATCH --mem=2G
#SBATCH --time=0-00:20:00
#SBATCH --output=logs/%x_%A_%a.out
#SBATCH --job-name=LFC_largeN_generate
# Set your HPC allocation account before submitting, e.g.:
#   export SLURM_ACCOUNT=def-yourpi; sbatch --account="$SLURM_ACCOUNT" ...

# Unified large-N LFC generation script - both the SLURM array-task body AND
# its own submission driver, replacing the broken graph_gen.sh /
# submit_all_graphs.sh / subjob_generate.sh chain (subjob_generate.sh never
# existed in the original project - that chain could not actually run).
#
# One array task = one (nodes, degree) combination, generating a full
# 100-p-value sweep via generate_graph.py with a deterministic, recorded
# seed (BASE_SEED + task index) so every task's realization is reproducible
# and distinct from its siblings.
#
# Usage (submission driver - NOT via sbatch, just `bash`):
#   bash submit_largeN_generate.sh <base_seed> [nodes_list] [degrees_list]
#   e.g. bash submit_largeN_generate.sh 1000 "1000 5000" "2 4 6 8 10 12 14 16 18 20 22 24 26 28 30 32"

set -euo pipefail

if [ -n "${SLURM_ARRAY_TASK_ID:-}" ]; then
    module load StdEnv/2020 gcc/9.3.0 graph-tool/2.56 python/3.10

    read -ra NODES_ARR <<< "$NODES_LIST"
    read -ra DEGREES_ARR <<< "$DEGREES_LIST"
    N_DEGREES=${#DEGREES_ARR[@]}

    TASK_ID=$SLURM_ARRAY_TASK_ID
    D_IDX=$(( TASK_ID % N_DEGREES ))
    N_IDX=$(( TASK_ID / N_DEGREES ))

    N=${NODES_ARR[$N_IDX]}
    D=${DEGREES_ARR[$D_IDX]}
    SEED=$(( BASE_SEED + TASK_ID ))

    echo "nodes=$N degree=$D seed=$SEED (array task $TASK_ID)"

    SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
    python "$SCRIPT_DIR/generate_graph.py" "$N" "$D" --seed "$SEED"
    exit 0
fi

BASE_SEED=${1:?"usage: bash submit_largeN_generate.sh <base_seed> [nodes_list] [degrees_list]"}
NODES_LIST=${2:-"1000 5000"}
DEGREES_LIST=${3:-"2 4 6 8 10 12 14 16 18 20 22 24 26 28 30 32"}

read -ra NODES_ARR <<< "$NODES_LIST"
read -ra DEGREES_ARR <<< "$DEGREES_LIST"
TOTAL_TASKS=$(( ${#NODES_ARR[@]} * ${#DEGREES_ARR[@]} ))
LAST_IDX=$(( TOTAL_TASKS - 1 ))

mkdir -p logs

SELF="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/$(basename "${BASH_SOURCE[0]}")"
sbatch --array="0-${LAST_IDX}%50" \
    --export="BASE_SEED=${BASE_SEED},NODES_LIST=${NODES_LIST},DEGREES_LIST=${DEGREES_LIST}" \
    "$SELF"
echo "Submitted large-N generation: nodes=[${NODES_LIST}] degrees=[${DEGREES_LIST}] base_seed=${BASE_SEED} (${TOTAL_TASKS} tasks)"
