# scripts/

The generation pipeline that produces `../nets/` and, downstream, the
figures in `../figures/`. Each subfolder is one independent pipeline
stage/dataset; run each subfolder's own README for exact commands and
order.

| Folder | Produces | Docs |
|---|---|---|
| `LFC/` | LFC spectral-gain sweep, N=240 (ws/mhk/ke) | [LFC/README.md](LFC/README.md) |
| `LFC_largeN/` | LFC sweep at N=1000/5000 (mhk only) | [LFC_largeN/README.md](LFC_largeN/README.md) |
| `LTM/` | LTM cascade realizations, N=1000 (ws/mhk/ke) | [LTM/README.md](LTM/README.md) |
| `real_world/` | Real-world reference networks (Netzschleuder) | [real_world/README.md](real_world/README.md) |
| `pooling/` | Shared `lfc_data_loader.py` utility (no own pipeline) | — |
| `figures/` | Notebooks that turn `../nets/` into `../figures/` PDFs | — |

## Common prerequisites (all of LFC/, LFC_largeN/, LTM/, real_world/)

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
  always via `sbatch` or the persistent Jupyter kernel.
