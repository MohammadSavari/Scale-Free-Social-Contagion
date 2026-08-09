# `data/` - chunked network archives

`../nets/` holds tens of thousands of small per-seed `.gt` network files
across the LFC and LTM model sweeps plus the real-world reference networks
— several GB in total. Packaged whole, some of these subtrees compress to
hundreds of MB each, above the 10MB per-file cap of this package's external
release target. Each subtree is therefore tarred and then split into
`<=10MB` chunks.

## Archives

Nine archives, each holding one subtree of `../nets/`:

| Archive | Extracts to | Contents |
|---|---|---|
| `LFC_240_ws` | `nets/LFC/240/ws` | 21400 `.gt` — k=8/16 at 100 seeds, 14 other k at seed 1 |
| `LFC_240_mhk` | `nets/LFC/240/mhk` | 21400 `.gt`, same layout |
| `LFC_240_ke` | `nets/LFC/240/ke` | 8 `.gt` — k ∈ {8,16}, seeds 1-4 (see "ke coverage") |
| `LFC_1000_mhk` | `nets/LFC/1000/mhk` | 1600 `.gt` — 16 k values × 100 p, seed 1 |
| `LFC_5000_mhk` | `nets/LFC/5000/mhk` | 1600 `.gt`, same layout |
| `LTM_1000_ws` | `nets/LTM/1000/ws` | 1000 `.gt` (100 seeds) + pooled eigenvalue `.npy` |
| `LTM_1000_mhk` | `nets/LTM/1000/mhk` | 800 `.gt` (100 seeds) + pooled eigenvalue `.npy` |
| `LTM_1000_ke` | `nets/LTM/1000/ke` | 8 `.gt` — k ∈ {8,16}, seeds 1-4 |
| `real_world` | `nets/real_world` | 18 `.gt` + 30 CSVs, 10 Netzschleuder networks |

Each archive also carries the `_props.csv` and gains CSVs derived from its
`.gt` files, so the tables the figures read are present without re-running
the extraction stage.

### ke coverage

The `ke` archives are much smaller than their ws/mhk counterparts by
design. `ke_network(n, k, seed)` has no `p` parameter — it depends only on
node count, degree and seed — so storing one graph per p-value would give
redundant draws of a single ensemble rather than a sweep. Independent
**realizations** are the meaningful axis instead, and there are four per k.
Figure S16's ke panel accordingly shows four realizations, not four
p-values.

## Files in this directory

- `<archive>.tar.gz.part000`, `part001`, … — the `<=10MB` chunks (the only
  files tracked; the whole `.tar.gz` exists only transiently during
  packaging and unpacking)
- `checksums_full.sha256` — sha256 of each whole (reassembled) archive
- `checksums_parts.sha256` — sha256 of each individual chunk, to verify the
  parts themselves transferred correctly
- `REASSEMBLE.md` — standalone unpacking instructions, for reading the
  archives without the rest of the repo

## Unpacking

Use `../reassemble.sh` from the repo root. It concatenates the parts,
verifies the result against `checksums_full.sha256`, and extracts it into
`../nets/` — the location the scripts and the figure notebook read from.

```bash
bash reassemble.sh LFC_240_ws     # one archive
bash reassemble.sh all            # every archive
```

## Repackaging

`../package_data.sh` is the inverse: it tars each subtree of `../nets/`,
splits it, and regenerates both checksum files and `REASSEMBLE.md`. It
walks tens of thousands of small files, so run it as a batch job rather
than interactively:

```bash
export SLURM_ACCOUNT=def-yourpi
sbatch --account="$SLURM_ACCOUNT" submit_package_data.sh
```
