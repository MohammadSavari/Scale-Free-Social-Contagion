#!/bin/bash
#SBATCH --mem=4G
#SBATCH --time=0-00:30:00
#SBATCH --array=0-99%50
#SBATCH --output=logs/%x_%A_%a.out
# Set your HPC allocation account before submitting, e.g.:
#   export SLURM_ACCOUNT=def-yourpi; sbatch --account="$SLURM_ACCOUNT" ...

# Unified LTM cascade submission script - both the SLURM array-task body AND
# its own submission driver that loops all 6 canonical (net, k) combos,
# replacing the manual one-off
#   sbatch --export=NET=ws,K=16 --job-name=ltm_polspeed_ws16 submit_ltm_polspeed.sh
# invocations that had no submission-loop wrapper.
#
# One array task = one realization seed's LTM cascade simulation for ONE
# (net, k) combo, all four selectors (top5, top10, bot5, bot10) together,
# reading nets/LTM/1000/<net>/<k>_seed<seed>/*.gt (produced by
# generate_ltm.py) and writing
# nets/LTM/1000/<net>/<k>_seed<seed>_{top5,top10,bot5,bot10}.csv.
# Runs on a compute node (never the login node).
#
# Usage (submission driver - NOT via sbatch, just `bash`):
#   bash submit_ltm_cascade_all.sh              # submits all 6 canonical combos
#   bash submit_ltm_cascade_all.sh ws 16         # submits just one combo

set -euo pipefail

if [ -n "${SLURM_ARRAY_TASK_ID:-}" ]; then
    source ~/jupy/bin/activate
    module load StdEnv/2020 gcc/9.3.0 graph-tool/2.56 python/3.10

    : "${NET:?Set NET via --export=NET=ws|mhk|ke}"
    : "${K:?Set K via --export=K=<degree>}"

    SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
    python "$SCRIPT_DIR/run_ltm_cascade.py" --net "$NET" --k "$K" --seed "$SLURM_ARRAY_TASK_ID" \
        --selectors top:5 top:10 bot:5 bot:10
    exit 0
fi

mkdir -p logs
SELF="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/$(basename "${BASH_SOURCE[0]}")"

if [ "$#" -ge 2 ]; then
    COMBOS="$1 $2"
else
    COMBOS="ws 16
ws 8
mhk 16
mhk 8
ke 16
ke 8"
fi

echo "$COMBOS" | while read -r NET K; do
    [ -z "$NET" ] && continue
    sbatch --job-name="ltm_polspeed_${NET}${K}" --export="NET=${NET},K=${K}" "$SELF"
    echo "Submitted cascade simulation for net=${NET} k=${K}"
done
