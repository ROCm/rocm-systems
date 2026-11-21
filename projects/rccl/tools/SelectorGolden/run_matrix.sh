#!/usr/bin/env bash
# Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
#
# Record SelectorGolden over the runtime settings that change selection:
# the gfx1250 tuning table (SPX / DPX) and the CTA policy. Run it once on the
# baseline build and once on the candidate, then diff each pair:
#
#   ./run_matrix.sh out/develop  -- mpirun -np 8 --bind-to numa
#   RCCL_DIR=/path/to/new/build ./run_matrix.sh out/new -- mpirun -np 8 --bind-to numa
#   for f in out/develop/*.csv; do ./golden_diff.py "$f" "out/new/$(basename "$f")"; done
#
# Everything after "--" is the launcher prefix. Extra SelectorGolden options can
# go in GOLDEN_ARGS, for example GOLDEN_ARGS="-d float,bfloat16,fp8e4m3 -o sum,max".
set -euo pipefail

if [[ $# -lt 2 || "$2" != "--" ]]; then
  echo "usage: $0 OUTDIR -- LAUNCHER..." >&2
  exit 2
fi
out="$1"; shift 2
launcher=("$@")
here="$(cd "$(dirname "$0")" && pwd)"
bin="${GOLDEN_BIN:-$here/SelectorGolden}"
mkdir -p "$out"

partitions=(${GOLDEN_PARTITIONS:-SPX DPX})
policies=(${GOLDEN_CTA_POLICIES:-0 2})

for part in "${partitions[@]}"; do
  for pol in "${policies[@]}"; do
    f="$out/golden_${part}_cta${pol}.csv"
    echo "== $f"
    # -x passes the variable to every rank under Open MPI; other launchers
    # inherit the environment.
    RCCL_TUNING_PARTITION_MODE="$part" "${launcher[@]}" -x RCCL_TUNING_PARTITION_MODE \
      "$bin" -p "$pol" -f "$f" ${GOLDEN_ARGS:-}
  done
done
