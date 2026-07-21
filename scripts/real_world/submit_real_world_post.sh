#!/bin/bash
#SBATCH --mem=4G
#SBATCH --time=00:30:00
#SBATCH --job-name=LFC_real_world_post
#SBATCH --output=logs/%x_%A.out
# Set your HPC allocation account before submitting, e.g.:
#   export SLURM_ACCOUNT=def-yourpi; sbatch --account="$SLURM_ACCOUNT" ...

# Runs after submit_real_world_generate.sh's array (submit with
# --dependency=afterany so this still runs even if some array tasks timed
# out; it just processes whichever .gt files exist). Rebuilds the CSVs and
# the summary figure from whatever nets/real_world/**/*.gt files exist at
# that point.

set -euo pipefail

source ~/jupy/bin/activate
module load StdEnv/2020 gcc/9.3.0 graph-tool/2.56 python/3.10

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
python "$SCRIPT_DIR/extract_real_world_csv.py"
python "$SCRIPT_DIR/generate_real_world_figure.py"
