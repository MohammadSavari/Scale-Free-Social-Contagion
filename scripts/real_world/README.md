# real_world

Real-world small-world and scale-free networks, processed through the same
LFC property pipeline as the synthetic mhk/ws/ke sweep in `../LFC/`
(`local_clustering`, `transitivity`, `shortest_path`, `eig_laplacian`,
`gains_top5`, `gains_bot5`), so they can be analyzed and compared directly.
All are pulled from the [Netzschleuder](https://networks.skewed.de)
repository via `graph_tool.collection.ns`.

Data lives at `../../nets/real_world/` (alongside `../../nets/LFC/` and
`../../nets/LTM/`), not next to these scripts - all scripts here default their
`--root`/output paths accordingly.

## Networks

| network | category | N | E | source |
|---|---|---|---|---|
| jazz_collab | small_world | 198 | 2742 | Gleiser & Danon (2003) |
| celegansneural | small_world | 297 | 2359 | White et al. (1986), C. elegans neurons |
| dolphins | small_world | 62 | 159 | Lusseau et al. (2003) |
| football | small_world | 115 | 613 | Girvan & Newman (2002) |
| celegans_metabolic | scale_free | 453 | 2025 | Jeong et al. (2000) / Duch & Arenas (2005) |
| uni_email | scale_free | 1133 | 10902 | Guimerà et al. (2003) |
| collins_yeast | scale_free | 1004 | 8319 | Collins et al. (2007) |
| polblogs | scale_free | 1222 | 19021 | Adamic & Glance (2005) |
| faa_routes | scale_free | 1226 | 2613 | US FAA (2010) |
| interactome_yeast | scale_free | 1458 | 1948 | Coulomb et al. (2005) |

N/E are post-cleanup (largest connected component, undirected, parallel
edges/self-loops removed) - `fetch_real_world_graph()` in
`generate_real_world.py` does this defensively for every network, not just
the ones that needed it.

`jazz_collab` and `celegans_metabolic` were generated first, interactively.
The other 8 were submitted as a batch SLURM job (`submit_real_world_generate.sh`).

## Known limitations

- **`power` (Western US power grid, N~4941) is deliberately excluded.**
  Gain computation cost scales roughly as N⁴ (selected nodes proportional
  to N, dense linear solve cost proportional to N³, done once per
  frequency); extrapolating from `celegans_metabolic`'s measured runtime
  (N=453, ~2.3 min for both `get_gain` calls) puts `power` at roughly 540
  hours - impractical at any reasonable HPC walltime limit. If you want to
  add it back, either grant a much longer walltime (the raw graph fetch
  itself is cheap and can still be cached via `--only power`) or change the
  algorithm (`get_gain`'s dense `linalg.solve` per node per frequency is the
  O(N⁴) bottleneck at this N - an iterative/sparse solver, or fewer
  frequencies, would be needed to make it practical).
- The raw, unannotated Netzschleuder downloads cached under
  `.ns_raw_cache/` have no `ID`/`probability`/`category` graph properties,
  so a naive `**/*.gt` scan would crash on them.
  `extract_real_world_csv.py` and `generate_real_world_figure.py` therefore
  skip any `.gt` file living under a dot-directory relative to the scan
  root.

## Directory layout

```
real_world/                              # this folder - scripts only
├── generate_real_world.py               # step 1: fetch + compute properties
├── extract_real_world_csv.py            # step 2: .gt -> CSVs
├── generate_real_world_figure.py         # step 3: CSVs/.gt -> summary PDF
├── submit_real_world_generate.sh         # SLURM wrapper for step 1
├── submit_real_world_post.sh             # SLURM wrapper for steps 2+3
└── README.md                             # this file

../../nets/real_world/                      # data (populated by these scripts, or by ../../reassemble.sh)
├── <category>/<name>/<name>.gt          # one network per subdirectory
├── <category>/<name>_props.csv          # from extract_real_world_csv.py
├── <category>/<name>_{top,bot}_5_corr_gains_degree.csv
├── figs/LFC_real_world_summary.pdf      # from generate_real_world_figure.py
└── .ns_raw_cache/<ns_name>.gt           # raw download cache, see below
```

## Environment

Same as the rest of the project - always run, in this order, before any
script below (interactively or inside an sbatch script):

```bash
source ~/jupy/bin/activate
module load StdEnv/2020 gcc/9.3.0 graph-tool/2.56 python/3.10
```

## Internet access (compute nodes typically have none)

SLURM compute nodes on many HPC clusters cannot reach the network - only
the login node can. `fetch_raw_graph()` in `generate_real_world.py`
therefore caches each raw download under `../../nets/real_world/.ns_raw_cache/`
the first time it's fetched, and `fetch_real_world_graph()` always reads
from that cache before ever calling `graph_tool.collection.ns[...]`.
**Consequence:** any network not yet in `.ns_raw_cache/` must be fetched
once on the login node before `submit_real_world_generate.sh` is submitted -
otherwise the SLURM job fails with a connection timeout. The cache shipped
with this package already has all 10 in-scope networks warmed.

## Scripts (run in this order)

1. **`generate_real_world.py`** - fetches (or loads from cache) each
   registered network and saves it as a `.gt` file under
   `../../nets/real_world/<category>/<name>/<name>.gt`.

   ```bash
   python generate_real_world.py                      # every registered network
   python generate_real_world.py --only jazz_collab    # just one
   ```

   For a full batch, `submit_real_world_generate.sh` submits this as a
   SLURM array (one task per network, run from `real_world/`):

   ```bash
   sbatch submit_real_world_generate.sh
   ```

2. **`extract_real_world_csv.py`** - thin wrapper around
   `../LFC/extract_lfc_csv.py` (no logic duplicated) that defaults `--root`
   to `../../nets/real_world`. Regenerates `<network>_props.csv` and
   `<network>_{top,bot}_5_corr_gains_degree.csv` from the `.gt` files.

   ```bash
   python extract_real_world_csv.py
   ```

3. **`generate_real_world_figure.py`** - builds a Figure-2/S15-style summary
   PDF at `../../nets/real_world/figs/LFC_real_world_summary.pdf`, one column
   per network found under `../../nets/real_world/**/*.gt`:
   - top row: collective frequency response H²(ω) for the top-5%/bottom-5%
     degree nodes (same random-walk-normalized-Laplacian gain
     `generate_lfc.py` already computed, unchanged), each curve rescaled
     to its own max of 1 for easy cross-network comparison.
   - bottom row: density histogram of the *symmetric* normalized Laplacian
     eigenvalue spectrum (`graph_tool`'s `laplacian(norm=True)`, computed
     fresh here - a different quantity from the *unnormalized*
     `eig_laplacian` property saved on the `.gt` file).

   ```bash
   python generate_real_world_figure.py
   ```

`submit_real_world_post.sh` runs steps 2 and 3 together; submit it with
`--dependency=afterany:<submit_real_world_generate.sh jobid>` so it waits
for the generation array to finish.

Figures 9, S19, and S20 (the celegans-comparison and 8-network-grid
figures built from this same data) are generated by the corresponding
cells in `figures/Figure_generator_nets_manual.ipynb`.

## Adding another real-world network

1. Add an entry to `REAL_WORLD_NETWORKS` in `generate_real_world.py`
   (Netzschleuder name + `category`: `small_world` or `scale_free`).
2. **On the login node** (not via `sbatch`):
   `python generate_real_world.py --only <name>` once, to warm
   `../../nets/real_world/.ns_raw_cache/` before submitting
   `submit_real_world_generate.sh`.
3. Check the gain-computation cost estimate before committing to a full
   run - `power`'s exclusion (see "Known limitations" above) is a reminder
   that not every candidate network is practical at this O(N⁴) cost.

`extract_real_world_csv.py` and `generate_real_world_figure.py` both
discover networks by globbing `../../nets/real_world/**/*.gt`, so no changes
are needed there.
