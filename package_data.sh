#!/bin/bash
# Compresses this package's data directories into data/*.tar.gz.part### chunks
# (<=10MB each) + checksums, for external upload to a target with a per-file
# size limit. Run this from the repo root after generating nets/{LFC,LTM,
# real_world}/ so that they exist alongside this script.
#
# Meant to run as a SLURM job (see submit_package_data.sh) - tar/gzip over
# nets/ walks many thousands of small per-seed files, and lustre metadata
# lookups for that are slow enough that this should not run on the login
# node interactively.
#
# Usage:
#   bash package_data.sh                      # every archive
#   bash package_data.sh LFC_5000_mhk         # just these, leaving the rest
#
# Naming one or more archives repackages only those: their old chunks are
# replaced and both checksum files are rebuilt to cover everything then
# present in data/, so a partial run leaves the directory self-consistent.
#
# Optional env vars, for packaging a tree that lives outside the repo:
#   NETS_ROOT   directory holding LFC/, LTM/, real_world/   (default: nets)
#   DATA_DIR    directory to write the chunks into          (default: data)
#
# Re-runnable: safe to call again any time the underlying data changes
# (e.g. after adding more seeds).

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"

NETS_ROOT="${NETS_ROOT:-nets}"
DATA_DIR="${DATA_DIR:-data}"

# Editor and rerun leftovers that can accumulate next to the real data. They
# are excluded rather than deleted so packaging never mutates nets/.
#
# '<k>_eig_cache.npz' is excluded for a different reason: it is a derived
# performance cache that pooled_eig_laplacians() reads INSTEAD of the .gt
# files when present, so a cache left over from an earlier run would
# silently serve eigenvalues for graphs that are not in the release. It
# costs nothing to omit - regenerate it after unpacking with
# scripts/LFC/submit_eig_cache.sh, which rebuilds it from the shipped .gt
# files. The '<k>_p<p>_eig_norm.npy' arrays the LTM figures read are NOT
# caches and DO ship.
#
# real_world/.ns_raw_cache/ is likewise NOT excluded - those cached
# Netzschleuder downloads ship deliberately, since compute nodes have no
# internet (see scripts/real_world/README.md).
TAR_EXCLUDES=(
  --exclude='*.bak'
  --exclude='*~'
  --exclude='*.tmp'
  --exclude='__pycache__'
  --exclude='.ipynb_checkpoints'
  --exclude='*_eig_cache.npz'
)

# name:relative-path-under-NETS_ROOT
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

# No arguments -> every archive. Otherwise, validate each name against the
# table above so a typo fails loudly instead of silently packaging nothing.
if [[ $# -eq 0 ]]; then
  SELECTED=()
  for entry in "${ARCHIVES[@]}"; do SELECTED+=("${entry%%:*}"); done
else
  SELECTED=("$@")
  for want in "${SELECTED[@]}"; do
    found=0
    for entry in "${ARCHIVES[@]}"; do
      [[ "${entry%%:*}" == "$want" ]] && found=1 && break
    done
    if [[ $found -eq 0 ]]; then
      echo "Unknown archive '${want}'. Known:" >&2
      for entry in "${ARCHIVES[@]}"; do echo "  ${entry%%:*}" >&2; done
      exit 1
    fi
  done
fi

mkdir -p "$DATA_DIR"

for entry in "${ARCHIVES[@]}"; do
  name="${entry%%:*}"
  relpath="${entry#*:}"

  skip=1
  for want in "${SELECTED[@]}"; do [[ "$want" == "$name" ]] && skip=0 && break; done
  [[ $skip -eq 1 ]] && continue

  if [[ ! -d "${NETS_ROOT}/${relpath}" ]]; then
    echo "Missing ${NETS_ROOT}/${relpath} - nothing to package for ${name}" >&2
    exit 1
  fi

  # Drop any chunks from a previous run of THIS archive, so a shrinking
  # dataset cannot leave orphaned high-numbered parts behind that would
  # corrupt the reassembled tarball.
  rm -f "${DATA_DIR}/${name}".tar.gz.part*

  echo "Packaging ${NETS_ROOT}/${relpath} -> ${DATA_DIR}/${name}.tar.gz"
  tar -czf "${DATA_DIR}/${name}.tar.gz" "${TAR_EXCLUDES[@]}" -C "$NETS_ROOT" "${relpath}"

  echo "Splitting ${DATA_DIR}/${name}.tar.gz into <=10MB parts"
  split -b 10M -d -a 3 "${DATA_DIR}/${name}.tar.gz" "${DATA_DIR}/${name}.tar.gz.part"

  # Recorded as 'data/<name>.tar.gz' regardless of DATA_DIR, because that is
  # the path reassemble.sh reconstructs and checks against on the
  # downloading side.
  new_sum="$(sha256sum "${DATA_DIR}/${name}.tar.gz" | cut -d' ' -f1)"
  rm "${DATA_DIR}/${name}.tar.gz"

  touch "$DATA_DIR/checksums_full.sha256"
  grep -v "  data/${name}\.tar\.gz\$" "$DATA_DIR/checksums_full.sha256" \
      > "$DATA_DIR/checksums_full.sha256.tmp" || true
  echo "${new_sum}  data/${name}.tar.gz" >> "$DATA_DIR/checksums_full.sha256.tmp"
  mv "$DATA_DIR/checksums_full.sha256.tmp" "$DATA_DIR/checksums_full.sha256"
done

# Rewrite checksums_full in the canonical ARCHIVES order, so the file reads
# the same whether it was built in one run or several.
: > "$DATA_DIR/checksums_full.sha256.tmp"
for entry in "${ARCHIVES[@]}"; do
  name="${entry%%:*}"
  grep "  data/${name}\.tar\.gz\$" "$DATA_DIR/checksums_full.sha256" \
      >> "$DATA_DIR/checksums_full.sha256.tmp" || true
done
mv "$DATA_DIR/checksums_full.sha256.tmp" "$DATA_DIR/checksums_full.sha256"

# Always rebuilt from every chunk actually present, so it stays correct
# after a partial repackage.
echo "Computing checksums of chunks"
(cd "$DATA_DIR" && sha256sum *.tar.gz.part* > checksums_parts.sha256)

cat > "$DATA_DIR/REASSEMBLE.md" <<'EOF'
# Reassembling the chunked archives

Each archive was split into <=10MB parts named `<name>.tar.gz.part000`,
`part001`, etc. The archive members are paths relative to `nets/`, so an
archive must be extracted into `nets/`:

```bash
cat <name>.tar.gz.part* > <name>.tar.gz
sha256sum <name>.tar.gz   # compare against checksums_full.sha256
mkdir -p ../nets && tar xzf <name>.tar.gz -C ../nets
```

`../reassemble.sh` does all of this for you, including the checksum check.

Archives: LTM_1000_ws, LTM_1000_mhk, LTM_1000_ke, LFC_240_ws, LFC_240_mhk,
LFC_240_ke, LFC_1000_mhk, LFC_5000_mhk, real_world.

- `checksums_full.sha256` - sha256 of each whole (reassembled) tar.gz.
- `checksums_parts.sha256` - sha256 of each individual chunk, for
  verifying transfer integrity of the parts themselves.
EOF

echo
echo "Done. Packaged: ${SELECTED[*]}"
echo "Chunks now in ${DATA_DIR}: $(ls "$DATA_DIR"/*.tar.gz.part* | wc -l)"
echo
echo "Full-archive checksums (${DATA_DIR}/checksums_full.sha256):"
cat "$DATA_DIR/checksums_full.sha256"
