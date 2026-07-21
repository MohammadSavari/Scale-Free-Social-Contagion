#!/bin/bash
#SBATCH --mem=4G
#SBATCH --time=10:00:00
#SBATCH --array=0-7
#SBATCH --job-name=LFC_real_world_gen
#SBATCH --output=logs/%x_%A_%a.out
# Set your HPC allocation account before submitting, e.g.:
#   export SLURM_ACCOUNT=def-yourpi; sbatch --account="$SLURM_ACCOUNT" ...

# One array task = one real-world network, generated via
# `generate_real_world.py --only <name>` (fetch from Netzschleuder + compute
# the same properties generate_lfc.py computes for the synthetic sweep).
# Each task gets its own independent 10h walltime, so a slow network timing
# out does not affect the others - unlike one long serial script.
#
# 'power' (Western US power grid, 4941 nodes) is deliberately NOT in this
# array: gain-computation cost scales roughly as N^4, and extrapolating
# from celegans_metabolic's measured runtime (453 nodes, ~2.3 min) puts
# power at ~540h - wildly impractical here. See README.md's "Known
# limitations" section.

set -euo pipefail

source ~/jupy/bin/activate
module load StdEnv/2020 gcc/9.3.0 graph-tool/2.56 python/3.10

NETWORKS=(celegansneural dolphins football uni_email collins_yeast polblogs faa_routes interactome_yeast)
NAME=${NETWORKS[$SLURM_ARRAY_TASK_ID]}

echo "task=$SLURM_ARRAY_TASK_ID network=$NAME"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
python "$SCRIPT_DIR/generate_real_world.py" --only "$NAME"
