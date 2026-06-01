#!/usr/bin/env bash
# Copyright (c) 2025-2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT
#
# Scalar-vs-SIMD A/B equivalence check. force_scalar() is process-wide and
# immutable, so a single run exercises one execute mode. This driver runs the
# *SimdCorrectness* suites twice -- once forcing scalar, once SIMD -- with each
# case dumping its per-lane results, then diffs the two dumps. The SIMD fast
# paths are required to be bit-identical to the scalar generated bodies, so any
# difference is a real divergence and fails the test.
set -uo pipefail

BIN="${1:?usage: simd_ab_diff.sh <rocjitsu_tests binary>}"
FILTER='*SimdCorrectness*'
TMP="$(mktemp -d)"
trap 'rm -rf "$TMP"' EXIT

RJ_FORCE_SCALAR=1 RJ_SIMD_DUMP="$TMP/scalar.txt" "$BIN" --gtest_filter="$FILTER" >/dev/null 2>&1
rc_scalar=$?
RJ_FORCE_SCALAR=0 RJ_SIMD_DUMP="$TMP/simd.txt" "$BIN" --gtest_filter="$FILTER" >/dev/null 2>&1
rc_simd=$?

# A non-zero gtest exit means an in-process invariant (e.g. inactive-lane
# preservation) failed; surface it but still attempt the diff for context.
if [ "$rc_scalar" -ne 0 ] || [ "$rc_simd" -ne 0 ]; then
    echo "WARNING: gtest exited non-zero (scalar=$rc_scalar simd=$rc_simd) -- see the per-test results"
fi

if [ ! -s "$TMP/scalar.txt" ] || [ ! -s "$TMP/simd.txt" ]; then
    echo "FAIL: result dump empty -- filter '$FILTER' matched no recording tests"
    exit 1
fi

sort "$TMP/scalar.txt" -o "$TMP/scalar.txt"
sort "$TMP/simd.txt" -o "$TMP/simd.txt"

if ! diff -u "$TMP/scalar.txt" "$TMP/simd.txt"; then
    echo "FAIL: SIMD execute path diverged from scalar (see unified diff above)"
    exit 1
fi

echo "PASS: scalar == SIMD across $(wc -l <"$TMP/scalar.txt") recorded cases"
[ "$rc_scalar" -eq 0 ] && [ "$rc_simd" -eq 0 ]
