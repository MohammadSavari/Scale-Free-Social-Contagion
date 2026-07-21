#!/bin/bash
#SBATCH --account=def-rbouffan
#SBATCH --time=3:00:00
#SBATCH --ntasks=1
#SBATCH --mem=8G
#SBATCH --job-name=package_data
#SBATCH --output=package_data_%j.log

# Runs package_data.sh on a compute node: tars nets/{LFC,LTM,real_world}
# subdirs into data/*.tar.gz.part### chunks (<=10MB each). Pure tar/gzip/
# split I/O work - no graph_tool/networkx module load needed.

set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
bash package_data.sh
