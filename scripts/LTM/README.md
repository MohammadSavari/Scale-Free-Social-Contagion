# LTM

Generates the LTM (linear threshold model) cascade realizations sweep at
N=1000 — ws/mhk/ke network types, k=8 and k=16, 100 seed realizations
each, plus the polarization-speed cascade simulation and eigenvalue
precompute that feed the LTM figures.

## p-values are matched on clustering, not on p

`p` means different things in the two models — a rewiring probability in ws,
a triad-formation probability in mhk — so generating both at the same
numeric `p` is not a controlled comparison. The p-values here are instead
chosen so the two models are compared at **matched clustering**:

- four target CC values spaced evenly across mhk's reachable CC range,
- the ws `p` realizing each target solved from ws's own CC(p) curve,
- plus, for ws only, an unmatched fifth value at `p = 0` (the pure ring
  lattice), whose clustering sits above mhk's ceiling and so has no mhk
  counterpart.

Realized agreement between the paired ws and mhk clustering values is
≤ 0.0032 across 100 seeds.

The chosen values are checked in as **`mhk_cc_matched_probs.json`**, a
fixed constant of the study — `generate_ltm.py` and `precompute_eig_norm.py`
both read their p-values from it, so the two stages cannot drift apart and
reproducing this data needs no calibration run.

## ke coverage

ke is generated for **k ∈ {8, 16} only**, and gets a single graph per
(k, seed) rather than one per p-value: `ke_network` has no p-dependence, so
p-labelled duplicates would be repeated draws of one ensemble rather than a
sweep. Independent realizations are indexed by `--seed`.

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
   arrays for the 18 (network, k, p) combos the eigenspectrum figure panels
   need (via `precompute_eig_norm.py`): per k ∈ {16, 8}, five ws (the four
   CC-matched p plus the unmatched `p = 0` ring lattice) and four mhk.
   Independent of step 3 — can run any time after step 1:
   ```bash
   sbatch --account="$SLURM_ACCOUNT" submit_eig_norm.sh
   python scripts/LTM/precompute_eig_norm.py --n-combos   # confirm the array size
   ```
   The `p = 0` ws entry is not optional: the eigenvalue panel loads one
   `.npy` for *every* p present in the pooled props, so omitting it makes
   Figure 1 fail on a missing file rather than quietly drop a curve.

`../pooling/lfc_data_loader.py` is *not* used by anything in this folder
(`precompute_eig_norm.py` reimplements its own pooling inline) — it's
only consumed by `../LFC/precompute_eig_cache.py`.

## Seeding

`--seed` on `generate_ltm.py` is the realization index and seeds network
generation; each graph within a realization is offset by its **p index**:

```python
graph_seed = args.seed * 1000 + p_index
```

The p **index** is used rather than p itself because indices are dense in
`[0, len(probs))`, so each realization's block of 1000 seeds stays disjoint
from its neighbours' and no two graphs in the sweep can collide. Deriving
the offset from p directly (e.g. `int(round(p * 1000))`) would map seed *s*
at `p = 1.0` onto seed *s+1* at `p = 0.0`.

`run_ltm_cascade.py` seeds nothing and needs nothing — with `top:`/`bot:`
selectors the cascade is deterministic (nodes are sorted by degree and an
explicit `seed_nodes` list is passed), so all cascade randomness enters at
graph-generation time. Its `--seed` selects which realization directory to
read, nothing more.
