# LFC

Generates the LFC (spectral-gain) network sweep at N=240 — ws/mhk/ke
network types, degree sweep, one `.gt` file per (k, p, seed).

## Prerequisites

- Run every command below from the **repo root**
  (`temp_Scale-Free-Social-Contagion_clone/`), not from inside this
  folder — output paths (`nets/...`) and `--output=logs/...` are relative
  to the submission cwd.
- `logs/` must exist at the repo root (create with `mkdir -p logs` if
  missing).
- No `--account` is hardcoded in any `.sh` here — set
  `export SLURM_ACCOUNT=def-yourpi` and pass `--account="$SLURM_ACCOUNT"`
  on the `sbatch` command line.
- Standard environment, before running anything interactively:
  ```bash
  source ~/jupy/bin/activate
  module load StdEnv/2020 gcc/9.3.0 graph-tool/2.56 python/3.10
  ```

## Scripts (run in this order)

1. **`submit_lfc_generate.sh`** — generates the `.gt` files (via
   `generate_lfc.py`). Submission driver, not a plain `sbatch` target:
   ```bash
   bash submit_lfc_generate.sh <model_type: mhk|ws|ke> <seed_start> <seed_end> [k_values...]
   # e.g. bash submit_lfc_generate.sh mhk 0 99 2 4 6 8 10 12 14 16 18 20 22 24 26 28 30 32
   ```

2. **`submit_lfc_extract.sh`** — rebuilds `_props.csv` and
   `_{top,bot}_5_corr_gains_degree.csv` from the `.gt` files (via
   `extract_lfc_csv.py`):
   ```bash
   bash submit_lfc_extract.sh [nodes] [k_values...]
   # e.g. bash submit_lfc_extract.sh 240 2 4 6 8 10 12 14 16 18 20 22 24 26 28 30 32
   ```

3. **`submit_eig_cache.sh`** — precomputes/caches pooled Laplacian
   eigenvalues per (network, k=16) into `<k>_eig_cache.npz` (via
   `precompute_eig_cache.py`, which reuses `../pooling/lfc_data_loader.py`):
   ```bash
   sbatch --account="$SLURM_ACCOUNT" submit_eig_cache.sh
   ```

`generate_lfc.py`, `extract_lfc_csv.py`, and `precompute_eig_cache.py` can
also be run directly (see each file's docstring/`--help`) for a single
combo, e.g. for testing on a small case.
