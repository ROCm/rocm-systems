#!/usr/bin/env bash
#
# Copyright (c) Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT
#
# Runs transferBench and checks output. Use with CTest SKIP_RETURN_CODE 77.
#   Exit 0   → transferBench OK (fixture transferbench_available is set).
#   Exit 77  → No valid transfers (e.g. single-GPU system) – test skipped in CI.
#   Other    → Unexpected failure.
#
# Usage: transferbench_validation_check.sh <transferBench_exe> [args...]

EXE="$1"
shift

if [[ -z "$EXE" ]]; then
    echo "Usage: $0 <transferBench_exe> [args...]"
    exit 77
fi

OUTPUT=$("$EXE" "$@" 2>&1)
STATUS=$?

echo "$OUTPUT"

if echo "$OUTPUT" | grep -q "Error: No valid transfers created"; then
    echo ""
    echo "transferBench: No valid transfers created (single-GPU or topology mismatch) – skipping"
    exit 77
fi

exit $STATUS
