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

## Seed values used for the released data

- **ws & mhk, k=8 and k=16**: seeds **1-100** (`8_seed1`...`8_seed100`,
  `16_seed1`...`16_seed100`), i.e. `submit_lfc_generate.sh {ws,mhk} 1 100 8 16`
  — 100 realizations each, pooled for the banded figures.
- **ws & mhk, all other k-values** (2, 4, 6, 10, 12, 14, 18, 20, 22, 24,
  26, 28, 30, 32): **seed 1 only** (`<k>_seed1`), i.e.
  `submit_lfc_generate.sh {ws,mhk} 1 1 <k>` — a single realization each.
- **ke**: laid out differently on disk (`nets/LFC/240/ke/<k>/`, no
  `_seed` suffix on the directory) since `ke_network(n, m, seed)` doesn't
  actually depend on `p`. Each k has 100 `.gt` files, one per p-value in
  the sweep, and each was generated with an independent random seed
  (`--seed` omitted, so `generate_lfc.py` fell back to
  `np.random.randint(2**63)`) rather than the ws/mhk sequential 1-100
  scheme. The seed actually used for a given file is recorded on that
  file's own `seed` graph property, not reproducible from a single number
  here.
