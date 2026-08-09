# scripts/

The generation pipeline that produces `../nets/` and, downstream, the
figures in `../figures/`. Each subfolder is one independent pipeline
stage/dataset; read each subfolder's own README for exact commands and
order.

| Folder | Produces | Docs |
|---|---|---|
| `LFC/` | LFC spectral-gain sweep, any N (ws/mhk/ke) | [LFC/README.md](LFC/README.md) |
| `LTM/` | LTM cascade realizations, N=1000 (ws/mhk/ke) | [LTM/README.md](LTM/README.md) |
| `real_world/` | Real-world reference networks (Netzschleuder) | [real_world/README.md](real_world/README.md) |
| `pooling/` | Canonical `lfc_data_loader.py` utility (no own pipeline) | — |
| `figures/` | Notebook that turns `../nets/` into `../figures/` PDFs | — |

`net_functions.py` at this level is the **single source of truth** for the
three network generators (`ws_network`, `mhk_network`, `ke_network`), the
networkx→graph-tool conversion, the gain computation, and the
graph-property helpers. Every pipeline stage and the figure notebook import
from it, so the figures plot the same models the data was generated with.

`pooling/lfc_data_loader.py` is likewise the only copy — the figure
notebook imports it from there rather than carrying its own.

## Exact average degree

`--k` is the **target average degree** for all three models, and the
generators hit it exactly: `2E == round(N·k)` for every graph produced.
`extract_lfc_csv.py` records `V`, `E` and `kbar = 2E/V` in each `_props.csv`
so this is checkable straight from the released CSVs (see
[LFC/README.md](LFC/README.md) for the one subtree that predates those
columns). `smoke_test.py` asserts the same property directly on freshly
generated graphs.

## Gain computation

`net_functions.gain_all_nodes` recovers the frequency-domain gain H² for
**every** node from a single eigendecomposition, rather than a dense complex
solve per (node, frequency). The top-5% and bottom-5% leader sets are then
slices of that one result. It agrees with the direct per-node reference
implementation (`net_functions.get_gain`, kept for verification) to ~1e-14:

| N | `get_gain`, 5% of nodes | `gain_all_nodes`, all nodes |
|---|---|---|
| 240 | ~2.5 s | 0.09 s |
| 1000 | ~3.8 min | 5.2 s |
| 5000 | ~22.6 h | 9.5 min |

This is what lets one code path (`LFC/`, via `--nodes`) cover N=240, 1000
and 5000 alike.

## Common prerequisites (all of LFC/, LTM/, real_world/)

- Run every `.sh`/`.py` from the **repo root**
  (`temp_Scale-Free-Social-Contagion_clone/`), not from inside a subfolder
  — all output paths (`nets/...`, `logs/...`) are relative to the
  submission/invocation cwd.
- `logs/` must exist at the repo root (`mkdir -p logs`).
- No `--account` is hardcoded in any `.sh` — set
  `export SLURM_ACCOUNT=def-yourpi` and pass
  `--account="$SLURM_ACCOUNT"` on the `sbatch` command line.
- Standard environment before running anything (interactively or inside
  an sbatch script):
  ```bash
  source ~/jupy/bin/activate
  module load StdEnv/2020 gcc/9.3.0 graph-tool/2.56 python/3.10
  ```
- Never run graph_tool/networkx compute directly on the login node —
  always via `sbatch` or an interactive allocation.

## Sanity check

`smoke_test.py` asserts, for every k it is given, that the generated graph is
connected, has exactly N nodes, has no self-loops, and satisfies
`2E == round(N·k)` — i.e. that the realized average degree really is k:

```bash
python scripts/smoke_test.py
```

## Regenerating the figures

`figures/Figure_generator_nets_manual.ipynb` turns a populated `../nets/`
into the PDFs in `../figures/`. To run it non-interactively, cell by cell,
skipping any figure whose input data is absent:

```bash
cd scripts/figures && python run_figures.py
```
