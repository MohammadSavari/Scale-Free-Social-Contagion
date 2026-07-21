# Scale-Free-Social-Contagion

Data-availability package for ``How social contagions shape collective consensus in the presence of scale-free networks'' paper by Savari, M et al.
Bundles the generated network files (`nets/`) for external release, alongside the scripts that
produced and packaged them.

## Pipeline

`nets/` is produced by the generation pipeline in `scripts/` (network
generation → CSV extraction → eigenvalue precompute, per model/dataset) -
see [scripts/README.md](scripts/README.md) for the full stage-by-stage
breakdown and run order.

## Data

`nets/` is packaged into `<=10MB` chunks under `data/` for upload to a
release target with a per-file size limit - see
[data/README.md](data/README.md) for the full archive list and why it's
chunked.

To uncompress/reassemble an archive, use `reassemble.sh` from this
directory:

```bash
bash reassemble.sh LFC_240_ws     # one archive
bash reassemble.sh all            # every archive
```

It concatenates the parts, verifies the result against
`data/checksums_full.sha256`, and extracts it into `data/`.