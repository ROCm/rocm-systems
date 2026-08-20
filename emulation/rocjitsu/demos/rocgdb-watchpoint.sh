#!/usr/bin/env bash
#
# Copyright (c) 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT
#
# Demo: catch which GPU wave writes a buffer with a hardware data watchpoint,
# entirely in software with no AMD GPU. See rocgdb-watchpoint.md. Regenerate the
# .cast with:
#   emulation/mirage/scripts/record_demo.sh emulation/rocjitsu/demos/rocgdb-watchpoint.sh

set -euo pipefail

here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
kernel="$here/../tests/rocgdb/add_one.hip"
mirage="${MIRAGE_BIN:-mirage}"

say() { printf '\n\033[1;36m# %s\033[0m\n' "$*"; }

say "A tiny HIP kernel that adds 1 to every element of a device buffer:"
sed -n '/__global__/,/^}/p' "$kernel"

say "Build it for gfx950 (MI350X) with device debug info:"
tmp="$(mktemp -d)"
trap 'rm -rf "$tmp"' EXIT
app="$tmp/add_one"
( set -x; hipcc --offload-arch=gfx950 -g -O0 -o "$app" "$kernel" )

# The launch line lets ROCgdb read the device pointer `d` before the kernel runs
# (no hard-coded GPU virtual address).
launch_line="$(grep -nE 'add_one<<<' "$kernel" | head -1 | cut -d: -f1)"

say "Set a hardware watchpoint on the device buffer and catch the GPU store:"
( set -x
  "$mirage" run --profile mi350x --gdb \
    --gdb-ex "break add_one.hip:${launch_line}" \
    --gdb-ex 'break add_one' \
    --gdb-ex 'run' \
    --gdb-ex 'set $waddr = (unsigned long)d' \
    --gdb-ex 'continue' \
    --gdb-ex 'watch *(int*)$waddr' \
    --gdb-ex 'continue' \
    --gdb-ex 'continue' \
    -- "$app"
)

say "ROCgdb trapped the exact wave whose 'data[i] += 1' wrote the watched address (Old value = 0, New value = 1)."
