#!/bin/bash
#SBATCH --job-name=LTM_eig_norm
#SBATCH --mem=4G
#SBATCH --time=0-00:30:00
# 18 combos: for each k in {16, 8}, 5 ws (the 4 CC-matched p plus the
# unmatched p=0 ring lattice) and 4 mhk. Confirm the count with
#   python scripts/LTM/precompute_eig_norm.py --n-combos
#SBATCH --array=0-17
#SBATCH --output=logs/%x_%A_%a.out
# Set your HPC allocation account before submitting, e.g.:
#   export SLURM_ACCOUNT=def-yourpi; sbatch --account="$SLURM_ACCOUNT" ...

set -euo pipefail

module load StdEnv/2020 gcc/9.3.0 graph-tool/2.56 python/3.10

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
python "$SCRIPT_DIR/precompute_eig_norm.py" --index "$SLURM_ARRAY_TASK_ID" --root nets
