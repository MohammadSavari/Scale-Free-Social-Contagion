#!/bin/bash
#SBATCH --job-name=LTM_extract
#SBATCH --mem=4G
#SBATCH --time=0-00:30:00
#SBATCH --output=logs/%x_%j.out
# Set your HPC allocation account before submitting, e.g.:
#   export SLURM_ACCOUNT=def-yourpi; sbatch --account="$SLURM_ACCOUNT" ...

set -euo pipefail

module load StdEnv/2020 gcc/9.3.0 graph-tool/2.56 python/3.10

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
python "$SCRIPT_DIR/extract_ltm_csv.py" --root nets
