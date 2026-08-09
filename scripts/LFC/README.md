# LFC

Generates the LFC (spectral-gain) network sweep — ws/mhk/ke network types,
degree sweep, one `.gt` file per (k, p, seed). N=240 is the default; the
same scripts cover N=1000 and N=5000 via `NODES=` (see "Large N" below).

## What `--k` means, and the exact-degree guarantee

`--k` is the **target average degree**, and the generators in
`../net_functions.py` hit it exactly: every graph satisfies
`2E == round(N·k)`.

`extract_lfc_csv.py` records `V`, `E` and `kbar = 2E/V` in each `_props.csv`,
so the realized degree is checkable without rescanning the `.gt` files. The
`nets/LFC/240/ws` CSVs are the one exception — they were extracted before
those columns were added, and were not rebuilt because nothing reads them
(no figure uses `V`/`E`/`kbar`). Re-running `submit_lfc_extract.sh 240`
regenerates them with the columns present.

Concretely, per model:

| generator | how ⟨k⟩ is pinned |
|---|---|
| `ws_network` | ring lattice of degree k, rewired — degree-preserving by construction |
| `mhk_network` | each grown node attaches with exactly k/2 edges; duplicates are rejected, not collapsed |
| `ke_network` | active set of size `m = k/2`, so each new node contributes m edges |

`ke_network` deactivates an active node with probability ∝ `1/(m + d_i)`,
the Klemm–Eguíluz rule. That rule is what produces the model's three
defining properties: a power-law degree distribution (tail exponent
≈ 3.3–3.5), emergent preferential attachment (Π(k) increasing, sublinear at
~k^0.36 because of the `m` offset), and a sharp decay of attachment rate
with node age.

## Output file naming

`generate_lfc.py` names its output `k<k>_p<p:.6f>.gt`. The name identifies
the graph completely within its `<k>_seed<seed>/` directory, so re-running
or backfilling a task **overwrites in place** rather than leaving a second
graph for the same parameter point.

The ws graphs in `../../nets/LFC/240/ws/` instead carry
`<ID>_<jobid>_<taskid>.gt` names, from a generation run that tagged each
file with its SLURM task. Both conventions are readable: **nothing reads a
`.gt` file by name**. `extract_lfc_csv.py` and `lfc_data_loader` glob
`*.gt` and key off the stored `probability` graph property, and the
notebook's `gt_path_for_p()` does the same. Never index a sorted `.gt`
listing positionally — only the `k<k>_p<p>` names sort into ascending-p
order.

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

## Large N (N=1000, N=5000)

Pass `NODES=`; the submit script picks appropriate memory and walltime:

```bash
NODES=1000 bash submit_lfc_generate.sh mhk 1 1 16
NODES=5000 bash submit_lfc_generate.sh mhk 1 1 16
bash submit_lfc_extract.sh 5000 16
```

One code path covers every N because `net_functions.gain_all_nodes`
computes H² for every node from a single eigendecomposition: an N=5000
graph takes ~9.5 min, against the ~22.6 h a per-node dense complex solve
would need. CSVs follow the same `<k>_seed<seed>_props.csv` naming at every
N.

## Seed values used for the released data

- **ws & mhk, k=8 and k=16**: seeds **1-100** (`8_seed1`...`8_seed100`,
  `16_seed1`...`16_seed100`), i.e. `submit_lfc_generate.sh {ws,mhk} 1 100 8 16`
  — 100 realizations each, pooled for the banded figures.
- **ws & mhk, all other k-values** (2, 4, 6, 10, 12, 14, 18, 20, 22, 24,
  26, 28, 30, 32): **seed 1 only** (`<k>_seed1`), i.e.
  `submit_lfc_generate.sh {ws,mhk} 1 1 <k>` — a single realization each.
- **ke**: **k=8 and k=16 only, seeds 1-4**, in the same
  `nets/LFC/240/ke/<k>_seed<n>/` layout as ws/mhk, i.e.
  `bash submit_lfc_generate.sh ke 1 4 8 16`.

  This is deliberately narrower than the ws/mhk sweeps, because ke has no
  `p` parameter: `ke_network(n, k, seed)` depends only on node count, degree
  and seed. Storing one `.gt` per p-value would give redundant draws of a
  single ensemble rather than a sweep. Independent realizations are the
  meaningful axis, so `--seed` indexes them and `p` is recorded only for
  schema consistency with ws/mhk.

  Consequence for figures: the ke panel of Figure S16 shows four
  **realizations** rather than four p-values. They are genuinely distinct —
  ke's growth is strongly stochastic, and across seeds 1-4 at k=16 the mean
  shortest path spans 1.97-2.88 and the spectral gap 0.41-1.36.
