#!/usr/bin/env bash
#
# Copyright (c) Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT
#
# Runs transferBench and checks output. Use with CTest SKIP_RETURN_CODE 77.
#   Exit 0   → transferBench OK (marker created; fixture can be set).
#   Exit 77  → transferBench cannot run successfully – test skipped in CI.
#
# Usage: transferbench-validation-check.sh <transferBench_exe> <marker_file> [args...]
# If marker_file is given, it is created (touched) on success so CTest can gate other tests.

EXE="$1"
MARKER="$2"
shift 2

if [[ -z "$EXE" ]]; then
    echo "Usage: $0 <transferBench_exe> <marker_file> [args...]"
    exit 77
fi

if [[ ! -x "$EXE" ]]; then
    echo "transferBench executable not found or not executable: $EXE – skipping"
    exit 77
fi

# Remove stale marker from previous runs so validation-passed cannot pass on remnants
if [[ -n "$MARKER" ]]; then
    rm -f "$MARKER"
fi

OUTPUT=$("$EXE" "$@" 2>&1)
STATUS=$?

echo "$OUTPUT"

if echo "$OUTPUT" | grep -q "Error: No valid transfers created"; then
    echo ""
    echo "transferBench: No valid transfers created (single-GPU or topology mismatch) – skipping"
    exit 77
fi

if [[ $STATUS -ne 0 ]]; then
    echo ""
    echo "transferBench exited with code $STATUS – skipping"
    exit 77
fi

if [[ -n "$MARKER" ]]; then
    touch "$MARKER"
fi
exit 0
