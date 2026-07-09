#!/usr/bin/env bash
#
# Copyright (c) 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT
#
# Demo: debug a real multi-wave GPU kernel with ROCgdb, entirely in software
# with no AMD GPU. See rocgdb-multiwave.md. Regenerate the .cast with:
#   emulation/mirage/scripts/record_demo.sh emulation/rocjitsu/demos/rocgdb-multiwave.sh

set -euo pipefail

here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
kernel="$here/../tests/rocgdb/multi_wave.hip"
mirage="${MIRAGE_BIN:-mirage}"

say() { printf '\n\033[1;36m# %s\033[0m\n' "$*"; }

say "A HIP kernel launched as one workgroup of 128 threads = two 64-lane waves:"
sed -n '/__global__/,/^}/p' "$kernel"

say "Build it for gfx950 (MI350X) with device debug info:"
tmp="$(mktemp -d)"
trap 'rm -rf "$tmp"' EXIT
app="$tmp/multi_wave"
( set -x; hipcc --offload-arch=gfx950 -g -O0 -o "$app" "$kernel" )

# Break on the store line so both waves have computed their per-thread `local`.
line="$(grep -nE 'data\[i\] = local' "$kernel" | head -1 | cut -d: -f1)"

say "Break inside the kernel: both waves of the workgroup stop together:"
( set -x
  "$mirage" run --profile mi350x --gdb \
    --gdb-ex "break multi_wave.hip:${line}" \
    --gdb-ex 'run' \
    --gdb-ex 'info threads' \
    --gdb-ex 'thread apply all -q print local' \
    --gdb-ex 'continue' \
    -- "$app"
)

say "Both waves correlated to workgroup (0,0,0) at positions /0 and /1, each reading its own private 'local' (3 and 451) — a real multi-wave kernel, no AMD GPU required."
