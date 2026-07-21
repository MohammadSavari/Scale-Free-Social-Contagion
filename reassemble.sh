#!/bin/bash
# Reassembles one (or all) of package_data.sh's chunked archives in data/
# back into a whole tar.gz, verifies it against data/checksums_full.sha256,
# and extracts it into data/.
#
# Usage:
#   bash reassemble.sh <archive-name>   # e.g. bash reassemble.sh LFC_240_ws
#   bash reassemble.sh all              # reassemble + extract every archive
#
# Archives: LTM_1000_ws, LTM_1000_mhk, LTM_1000_ke, LFC_240_ws, LFC_240_mhk,
# LFC_240_ke, LFC_1000_mhk, LFC_5000_mhk, real_world

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"

ARCHIVES=(
  LTM_1000_ws LTM_1000_mhk LTM_1000_ke
  LFC_240_ws LFC_240_mhk LFC_240_ke
  LFC_1000_mhk LFC_5000_mhk
  real_world
)

reassemble_one() {
  local name="$1"

  if ! compgen -G "data/${name}.tar.gz.part*" > /dev/null; then
    echo "No parts found for ${name} (expected data/${name}.tar.gz.part*)" >&2
    return 1
  fi

  echo "Reassembling data/${name}.tar.gz from parts"
  cat data/"${name}".tar.gz.part* > "data/${name}.tar.gz"

  echo "Verifying checksum"
  grep "data/${name}\.tar\.gz\$" data/checksums_full.sha256 | sha256sum -c -

  echo "Extracting data/${name}.tar.gz"
  tar xzf "data/${name}.tar.gz" -C data

  echo "Done: extracted ${name} into data/"
}

if [[ $# -ne 1 ]]; then
  echo "Usage: bash reassemble.sh <archive-name>|all" >&2
  echo "Archives: ${ARCHIVES[*]}" >&2
  exit 1
fi

if [[ "$1" == "all" ]]; then
  for name in "${ARCHIVES[@]}"; do
    reassemble_one "$name"
  done
else
  reassemble_one "$1"
fi
