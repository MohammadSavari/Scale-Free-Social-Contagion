# LTM

Generates the LTM (linear threshold model) cascade realizations sweep at
N=1000 — ws/mhk/ke network types, k=8 and k=16, 100 seed realizations
each, plus the polarization-speed cascade simulation and eigenvalue
precompute that feed the LTM figures.

## Prerequisites

Same as `../LFC/README.md`: run from the **repo root**, needs `logs/`
there, no `--account` hardcoded (supply via `SLURM_ACCOUNT`/`--account`),
and the standard `source ~/jupy/bin/activate` + `module load` environment
(note: `submit_ltm_cascade_all.sh`, `submit_ltm_extract.sh`, and
`submit_eig_norm.sh` import pandas — the venv activation matters here).

## Scripts (run in this order)

1. **`submit_ltm_generate.sh`** — array of 100 tasks (one per seed 0-99),
   each generating all 6 (network, k) combos' `.gt` realizations at their
   figure-relevant p-values (via `generate_ltm.py`). Idempotent — safe to
   resubmit after a partial failure:
   ```bash
   sbatch --account="$SLURM_ACCOUNT" submit_ltm_generate.sh
   ```

2. **`submit_ltm_extract.sh`** — rebuilds CC/T/SP/Rg CSVs from the `.gt`
   files (via `extract_ltm_csv.py`):
   ```bash
   sbatch --account="$SLURM_ACCOUNT" submit_ltm_extract.sh
   ```

3. **`submit_ltm_cascade_all.sh`** — submission driver + array-task body
   that runs the LTM cascade simulation (via `run_ltm_cascade.py`) for all
   6 canonical (network, k) combos, one 100-task array per combo:
   ```bash
   bash submit_ltm_cascade_all.sh              # submits all 6 combos
   bash submit_ltm_cascade_all.sh ws 16         # submits just one combo
   ```

4. **`submit_eig_norm.sh`** — precomputes normalized-Laplacian eigenvalue
   arrays for the 9 (network, p) combos the eigenspectrum figure panels
   need (via `precompute_eig_norm.py`). Independent of step 3 — can run
   any time after step 1:
   ```bash
   sbatch --account="$SLURM_ACCOUNT" submit_eig_norm.sh
   ```

`../pooling/lfc_data_loader.py` is *not* used by anything in this folder
(`precompute_eig_norm.py` reimplements its own pooling inline) — it's
only consumed by `../LFC/precompute_eig_cache.py`.
