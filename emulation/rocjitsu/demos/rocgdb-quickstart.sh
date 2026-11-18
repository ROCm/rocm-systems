#!/usr/bin/env bash
#
# Copyright (c) 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT
#
# Demo: debug a real GPU kernel with one command — `mirage run --gdb`.
# See rocgdb-quickstart.md. Regenerate the .cast with:
#   emulation/mirage/scripts/record_demo.sh emulation/rocjitsu/demos/rocgdb-quickstart.sh

set -euo pipefail

here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
kernel="$here/../tests/rocgdb/add_one.hip"
mirage="${MIRAGE_BIN:-mirage}"

say() { printf '\n\033[1;36m# %s\033[0m\n' "$*"; }

say "A tiny HIP kernel that adds 1 to every element:"
sed -n '/__global__/,/^}/p' "$kernel"

say "Build it for gfx950 (MI350X) with device debug info:"
tmp="$(mktemp -d)"
trap 'rm -rf "$tmp"' EXIT
app="$tmp/add_one"
( set -x; hipcc --offload-arch=gfx950 -g -O0 -o "$app" "$kernel" )

say "Debug the GPU kernel with one command — no GPU required:"
( set -x
  "$mirage" run --profile mi350x --gdb \
    --gdb-ex 'break add_one' \
    --gdb-ex 'run' \
    --gdb-ex 'info args' \
    --gdb-ex 'print n' \
    --gdb-ex 'print data[0]' \
    --gdb-ex 'continue' \
    -- "$app"
)

say "ROCgdb stopped a real emulated wave, read its source-level arguments, and resumed it."
