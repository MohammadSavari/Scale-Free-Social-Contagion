#!/bin/bash
#SBATCH --mem=32G
#SBATCH --time=1-12:00:00
#SBATCH --output=logs/%x_%A_%a.out
#SBATCH --job-name=LFC_largeN_gains
# Set your HPC allocation account before submitting, e.g.:
#   export SLURM_ACCOUNT=def-yourpi; sbatch --account="$SLURM_ACCOUNT" ...

# Unified large-N gains-computation script - both the SLURM array-task body
# AND its own submission driver, replacing process_graphs.sh's manual
# per-file `sbatch` loop and the broken lap_submit_degrees.sh/lap_prop.sh
# batch-mode alternative (which called a LFC_gains_batch.py that never
# existed in the original project).
#
# One array task = compute_gains.py applied to a single .gt file, with
# centrality = 5% of the node count (matching the original convention).
#
# Usage (submission driver - NOT via sbatch, just `bash`):
#   bash submit_largeN_gains.sh <nodes> [degrees_list]
#   e.g. bash submit_largeN_gains.sh 1000 "2 4 6 8 10 12 14 16 18 20 22 24 26 28 30 32"
#
# Builds a manifest of every matching .gt file under nets/LFC/<nodes>/mhk/<degree>/
# (run this from the package root, e.g. `bash LFC_largeN/submit_largeN_gains.sh ...`)
# and submits one array task per file.

set -euo pipefail

if [ -n "${SLURM_ARRAY_TASK_ID:-}" ]; then
    module load StdEnv/2020 gcc/9.3.0 graph-tool/2.56 python/3.10

    GT_FILE=$(sed -n "$((SLURM_ARRAY_TASK_ID + 1))p" "$MANIFEST")
    DEGREE=$(basename "$(dirname "$GT_FILE")")
    NODECOUNT=$(echo "$GT_FILE" | grep -oE '/LFC/[0-9]+/' | grep -oE '[0-9]+')
    CENTRALITY=$(( NODECOUNT / 20 ))

    echo "gt_file=$GT_FILE nodes=$NODECOUNT degree=$DEGREE centrality=$CENTRALITY (array task $SLURM_ARRAY_TASK_ID)"

    SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
    python "$SCRIPT_DIR/compute_gains.py" "$NODECOUNT" "$DEGREE" "$CENTRALITY" "$GT_FILE"
    exit 0
fi

NODES=${1:?"usage: bash submit_largeN_gains.sh <nodes> [degrees_list]"}
DEGREES_LIST=${2:-"2 4 6 8 10 12 14 16 18 20 22 24 26 28 30 32"}

mkdir -p logs
MANIFEST="logs/largeN_gains_manifest_${NODES}.txt"
: > "$MANIFEST"

for D in $DEGREES_LIST; do
    DIR="nets/LFC/${NODES}/mhk/${D}"
    if [ -d "$DIR" ]; then
        find "$DIR" -maxdepth 1 -name "*.gt" -type f >> "$MANIFEST"
    else
        echo "warning: directory not found: $DIR"
    fi
done

N_FILES=$(wc -l < "$MANIFEST")
if [ "$N_FILES" -eq 0 ]; then
    echo "no .gt files found for nodes=${NODES} degrees=[${DEGREES_LIST}] - nothing to submit"
    exit 1
fi
LAST_IDX=$(( N_FILES - 1 ))

SELF="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/$(basename "${BASH_SOURCE[0]}")"
sbatch --array="0-${LAST_IDX}%50" --export="MANIFEST=${MANIFEST}" "$SELF"
echo "Submitted gains computation for nodes=${NODES}, degrees=[${DEGREES_LIST}] (${N_FILES} files, manifest: ${MANIFEST})"
