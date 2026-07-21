# `data/` - chunked network archives

`nets/` holds thousands of small per-seed `.gt` network files across the
LFC and LTM model sweeps (240/1000/5000-node scans x ws/mhk/ke variants)
plus the real-world reference networks. Packaged whole, several of these
subtrees compress to hundreds of MB each - too large for a single file
upload to this package's external release target, which caps individual
files at 10MB. So each subtree is tarred and then split into `<=10MB`
chunks instead of uploaded as one archive.

`../package_data.sh` does the tarring/splitting; run it via
`../submit_package_data.sh` as a SLURM job rather than directly - tar/gzip
over this many small files is slow enough on this cluster's filesystem
(lustre metadata lookups) that it shouldn't run on the login node.

It produces 9 archives:

| Archive | Source (`nets/...`) |
|---|---|
| `LTM_1000_ws` | `LTM/1000/ws` |
| `LTM_1000_mhk` | `LTM/1000/mhk` |
| `LTM_1000_ke` | `LTM/1000/ke` |
| `LFC_240_ws` | `LFC/240/ws` |
| `LFC_240_mhk` | `LFC/240/mhk` |
| `LFC_240_ke` | `LFC/240/ke` |
| `LFC_1000_mhk` | `LFC/1000/mhk` |
| `LFC_5000_mhk` | `LFC/5000/mhk` |
| `real_world` | `real_world` |

## Files in this directory

- `<archive>.tar.gz.part000`, `part001`, ... - the `<=10MB` chunks (the
  only files meant to be uploaded/tracked; the whole `.tar.gz` is deleted
  right after splitting)
- `checksums_full.sha256` - sha256 of each whole (reassembled) archive
- `checksums_parts.sha256` - sha256 of each individual chunk, to verify
  the parts themselves transferred correctly

## Uncompressing an archive

Use `../reassemble.sh` from the repo root - it concatenates the parts,
verifies the result against `checksums_full.sha256`, and extracts it here:

```bash
bash reassemble.sh LFC_240_ws     # one archive
bash reassemble.sh all            # every archive
```
