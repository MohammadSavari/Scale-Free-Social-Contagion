# LFC_largeN

Large-N scaling companion to `../LFC/`'s N=240 sweep — generates the mhk
network type only, at N=1000 and N=5000, to check whether the LFC
spectral-gain results hold at scale.

## Prerequisites

Same as `../LFC/README.md`: run from the **repo root**, needs `logs/`
there, no `--account` hardcoded (supply via `SLURM_ACCOUNT`/`--account`),
and the standard `source ~/jupy/bin/activate` + `module load` environment.

## Scripts (run in this order)

1. **`submit_largeN_generate.sh`** — generates `.gt` files + `_props.csv`
   directly (via `generate_graph.py`), one array task per (nodes, degree)
   combo, with a deterministic per-task seed:
   ```bash
   bash submit_largeN_generate.sh <base_seed> [nodes_list] [degrees_list]
   # e.g. bash submit_largeN_generate.sh 1000 "1000 5000" "2 4 6 8 10 12 14 16 18 20 22 24 26 28 30 32"
   ```

2. **`submit_largeN_gains.sh`** — computes gains (via `compute_gains.py`)
   for every `.gt` file under `nets/LFC/<nodes>/mhk/<degree>/`, one array
   task per file, writing into a `<degree>_gain/` subfolder next to each
   input file:
   ```bash
   bash submit_largeN_gains.sh <nodes> [degrees_list]
   # e.g. bash submit_largeN_gains.sh 1000 "2 4 6 8 10 12 14 16 18 20 22 24 26 28 30 32"
   ```

3. **`combine_gains.py`** — merges all `<degree>_gain/*.csv` for a given
   node count into one `<degree>_gains.csv` apiece. No submit wrapper
   (cheap enough to run as a single job/kernel call, not an array) — still
   pandas/file-I/O work over many small CSVs, so run it via `sbatch` or the
   persistent Jupyter kernel, never directly on the login node:
   ```bash
   python combine_gains.py --nodes 1000 --centrality 50
   # --centrality must match what compute_gains.py used (5% of --nodes)
   ```
