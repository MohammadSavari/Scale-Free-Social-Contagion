#!/bin/bash
# Compresses this package's data directories into data/*.tar.gz.part### chunks
# (<=10MB each) + checksums, for external upload to a target with a per-file
# size limit. Run this from the package root (data_availability/) after
# generating/unpacking the data so that nets/{LFC,LTM,real_world}/ exist
# alongside this script.
#
# Meant to run as a SLURM job (see submit_package_data.sh) - tar/gzip over
# nets/ walks many thousands of small per-seed files, and lustre metadata
# lookups for that are slow enough that this should not run on the login
# node interactively.
#
# Usage:
#   bash package_data.sh
#
# Re-runnable: safe to call again any time the underlying data changes
# (e.g. after adding more seeds) - it always overwrites data/ and
# regenerates the checksums from the current archive contents.

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"

mkdir -p data
rm -f data/checksums_full.sha256 data/checksums_parts.sha256

# name:relative-path-under-nets
ARCHIVES=(
  "LTM_1000_ws:LTM/1000/ws"
  "LTM_1000_mhk:LTM/1000/mhk"
  "LTM_1000_ke:LTM/1000/ke"
  "LFC_240_ws:LFC/240/ws"
  "LFC_240_mhk:LFC/240/mhk"
  "LFC_240_ke:LFC/240/ke"
  "LFC_1000_mhk:LFC/1000/mhk"
  "LFC_5000_mhk:LFC/5000/mhk"
  "real_world:real_world"
)

for entry in "${ARCHIVES[@]}"; do
  name="${entry%%:*}"
  relpath="${entry#*:}"

  echo "Packaging nets/${relpath} -> data/${name}.tar.gz"
  tar -czf "data/${name}.tar.gz" -C nets "${relpath}"

  sha256sum "data/${name}.tar.gz" >> data/checksums_full.sha256

  echo "Splitting data/${name}.tar.gz into <=10MB parts"
  split -b 10M -d -a 3 "data/${name}.tar.gz" "data/${name}.tar.gz.part"
  rm "data/${name}.tar.gz"
done

echo "Computing checksums of chunks"
cd data
sha256sum *.tar.gz.part* > checksums_parts.sha256
cd "$ROOT"

cat > data/REASSEMBLE.md <<'EOF'
# Reassembling the chunked archives

Each archive was split into <=10MB parts named `<name>.tar.gz.part000`,
`part001`, etc. To reassemble and extract one:

```bash
cat <name>.tar.gz.part* > <name>.tar.gz
sha256sum <name>.tar.gz   # compare against checksums_full.sha256
tar xzf <name>.tar.gz
```

Archives: LTM_1000_ws, LTM_1000_mhk, LTM_1000_ke, LFC_240_ws, LFC_240_mhk,
LFC_240_ke, LFC_1000_mhk, LFC_5000_mhk, real_world.

- `checksums_full.sha256` - sha256 of each whole (reassembled) tar.gz.
- `checksums_parts.sha256` - sha256 of each individual chunk, for
  verifying transfer integrity of the parts themselves.
EOF

echo "Done. Chunks:"
ls -lh data/*.tar.gz.part* | head -20
echo "..."
echo
echo "Full-archive checksums (data/checksums_full.sha256):"
cat data/checksums_full.sha256
