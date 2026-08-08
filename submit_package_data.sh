#!/bin/bash
#SBATCH --time=3:00:00
#SBATCH --ntasks=1
#SBATCH --mem=8G
#SBATCH --job-name=package_data
#SBATCH --output=logs/package_data_%j.log
# Set your HPC allocation account before submitting, e.g.:
#   export SLURM_ACCOUNT=def-yourpi; sbatch --account="$SLURM_ACCOUNT" submit_package_data.sh
# (left unset here rather than hardcoded to any specific account)

# Runs package_data.sh on a compute node: tars nets/{LFC,LTM,real_world}
# subdirs into data/*.tar.gz.part### chunks (<=10MB each). Pure tar/gzip/
# split I/O work - no graph_tool/networkx module load needed.

#   sbatch --account="$SLURM_ACCOUNT" submit_package_data.sh                 # all
#   sbatch --account="$SLURM_ACCOUNT" submit_package_data.sh LFC_5000_mhk    # subset

set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
mkdir -p logs
bash package_data.sh "$@"
