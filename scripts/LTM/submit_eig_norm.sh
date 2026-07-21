#!/bin/bash
#SBATCH --job-name=LTM_eig_norm
#SBATCH --mem=4G
#SBATCH --time=0-00:30:00
#SBATCH --array=0-8
#SBATCH --output=logs/%x_%A_%a.out
# Set your HPC allocation account before submitting, e.g.:
#   export SLURM_ACCOUNT=def-yourpi; sbatch --account="$SLURM_ACCOUNT" ...

set -euo pipefail

module load StdEnv/2020 gcc/9.3.0 graph-tool/2.56 python/3.10

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
python "$SCRIPT_DIR/precompute_eig_norm.py" --index "$SLURM_ARRAY_TASK_ID" --root nets
