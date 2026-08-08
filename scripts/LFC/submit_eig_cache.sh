#!/bin/bash
#SBATCH --mem=2G
#SBATCH --time=0-01:00:00
#SBATCH --array=0-1
#SBATCH --output=logs/%x_%A_%a.out
#SBATCH --job-name=LFC_eig_cache
# Set your HPC allocation account before submitting, e.g.:
#   export SLURM_ACCOUNT=def-yourpi; sbatch --account="$SLURM_ACCOUNT" ...

# One array task per network type (mhk, ws), precomputing/caching pooled
# Laplacian eigenvalues for k=16 - the only k the eigenspectrum-histogram
# figure panels plot. Each task opens many individual .gt files (one per
# seed realization x p value) to build the pool; this is slow enough on
# Lustre that it must run on a compute node, not interactively on a login
# node. Writes '<k>_eig_cache.npz' next to the seed directories under
# nets/LFC/240/<network>/, which lfc_data_loader.pooled_eig_laplacians()
# reads instead of re-scanning on every notebook re-execution.

set -euo pipefail

module load StdEnv/2020 gcc/9.3.0 graph-tool/2.56 python/3.10

NETWORKS=(mhk ws)
NETWORK=${NETWORKS[$SLURM_ARRAY_TASK_ID]}

echo "network=$NETWORK k=16 (array task $SLURM_ARRAY_TASK_ID)"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
python "$SCRIPT_DIR/precompute_eig_cache.py" --network "$NETWORK" --k 16 --root nets
