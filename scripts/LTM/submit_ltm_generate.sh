#!/bin/bash
#SBATCH --job-name=LTM_realizations
#SBATCH --mem=4G
#SBATCH --time=0-02:00:00
#SBATCH --array=0-99%50
#SBATCH --output=logs/%x_%A_%a.out
# Set your HPC allocation account before submitting, e.g.:
#   export SLURM_ACCOUNT=def-yourpi; sbatch --account="$SLURM_ACCOUNT" ...

# One array task = one realization (SLURM_ARRAY_TASK_ID = seed 0-99) of all
# 6 (network, k) combos used by the LTM figure notebooks (ws@16, ws@8,
# mhk@16, mhk@8, ke@16, ke@8), each swept over only the p-values those
# notebooks' pick_props actually select - see generate_ltm.py's docstring.
# Output: nets/LTM/1000/<net>/<k>_seed<task_id>/p<value>.gt
# Idempotent (skips .gt files that already exist), so a rerun/resume after
# a partial failure is always safe.

set -euo pipefail

module load StdEnv/2020 gcc/9.3.0 graph-tool/2.56 python/3.10

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
python "$SCRIPT_DIR/generate_ltm.py" --seed "$SLURM_ARRAY_TASK_ID" --root nets
