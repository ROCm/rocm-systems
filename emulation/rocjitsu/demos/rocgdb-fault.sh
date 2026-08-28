#!/usr/bin/env bash
#
# Copyright (c) 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT
#
# Demo: catch a GPU memory-access fault (SIGSEGV) with ROCgdb, entirely in
# software with no AMD GPU. See rocgdb-fault.md. Regenerate the .cast with:
#   emulation/mirage/scripts/record_demo.sh emulation/rocjitsu/demos/rocgdb-fault.sh

set -euo pipefail

here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
kernel="$here/../tests/rocgdb/bad_access.hip"
mirage="${MIRAGE_BIN:-mirage}"

say() { printf '\n\033[1;36m# %s\033[0m\n' "$*"; }

say "A HIP kernel that stores through a wild, never-mapped device pointer:"
sed -n '/__global__/,/^}/p' "$kernel"

say "Build it for gfx950 (MI350X) with device debug info:"
tmp="$(mktemp -d)"
trap 'rm -rf "$tmp"' EXIT
app="$tmp/bad_access"
( set -x; hipcc --offload-arch=gfx950 -g -O0 -o "$app" "$kernel" )

say "Run it under ROCgdb — the emulator faults the wave on the wild store:"
( set -x
  "$mirage" run --profile mi350x --gdb \
    --gdb-ex 'run' \
    --gdb-ex 'backtrace' \
    --gdb-ex 'info registers pc' \
    -- "$app"
)

say "ROCgdb caught the GPU memory violation as SIGSEGV, stopped at the faulting store — no AMD GPU required."
