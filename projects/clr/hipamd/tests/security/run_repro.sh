#!/usr/bin/env bash
# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
#
# Turnkey build + run for the ROCM-26752 / SEC-00548 fat binary OOB reproducer.
#
# Usage:
#   ./run_repro.sh            # build with AddressSanitizer and run
#   HIPCC=/opt/rocm/bin/hipcc ./run_repro.sh
#   NO_ASAN=1 ./run_repro.sh  # build without ASan (relies on SIGSEGV to show OOB)
#
# Expected:
#   * Against an UNPATCHED HIP runtime -> ASan heap-buffer-overflow / SIGSEGV in
#     the fatbin parser (this is the reproduction of the vulnerability).
#   * Against a PATCHED HIP runtime    -> "ALL CASES PASSED" and exit code 0.
#
# For the cleanest ASan report, point this at a HIP runtime that was itself built
# with -fsanitize=address (e.g. via LD_LIBRARY_PATH / ROCM_PATH), otherwise the
# unpatched runtime typically surfaces the bug as a SIGSEGV instead.

set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
HIPCC="${HIPCC:-hipcc}"
SRC="${HERE}/hip_fatbin_oob_repro.cpp"
BIN="${HERE}/hip_fatbin_oob_repro"

FLAGS=(-O0 -g)
if [[ -z "${NO_ASAN:-}" ]]; then
  FLAGS+=(-fsanitize=address -fno-omit-frame-pointer)
fi

echo "== Building ${SRC}"
echo "   ${HIPCC} ${FLAGS[*]} -> ${BIN}"
"${HIPCC}" "${FLAGS[@]}" "${SRC}" -o "${BIN}"

echo "== Running reproducer"
export ASAN_OPTIONS="${ASAN_OPTIONS:-abort_on_error=1:halt_on_error=1:detect_leaks=0}"
"${BIN}"
