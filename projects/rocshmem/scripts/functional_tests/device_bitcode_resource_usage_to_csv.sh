#!/usr/bin/env bash
###############################################################################
# Copyright (c) Advanced Micro Devices, Inc. All rights reserved.
#
# SPDX-License-Identifier: MIT
###############################################################################
# Extract per-kernel GPU resource usage from the device-bitcode JIT artifact
# (librocshmem_device_<arch>.bc, produced by DeviceBitcode.cmake's
# `rocshmem_device_bitcode` target -- built by default as part of `cmake
# --build .` since that target is declared ALL).
#
# This artifact's `opt -O3` (over the whole llvm-linked module, no per-TU
# boundary, no LTO cost-model gate) is a materially different optimization
# regime from the production library's cost-gated LTO inliner (see
# DeviceBitcode.cmake's comments) -- a source change validated as
# safe/beneficial via the production-library resource-usage numbers alone has
# NOT been checked against this artifact until this is also run. Neither
# resource_usage_compare.sh/measure.sh measured this artifact at all before
# this script existed.
#
# Backend-compiling the already-opted .bc directly with
# -Rpass-analysis=kernel-resource-usage reuses the exact codegen decisions
# that would apply when this .bc is later JIT-linked and dispatched
# (rocshmem_hipmodule_init) -- it is not a re-optimization, just running the
# AMDGPU backend to get its resource-usage remarks.
#
# The resulting remarks have no real source location (the .bc has been
# merged across many TUs by llvm-link and re-optimized by opt -O3 with no
# -g), so source_file/line in the output CSV are the placeholder
# "<bitcode>"/0 for every row -- only mangled_name (and the resource
# columns) are meaningful for this artifact, which is sufficient since
# resource_usage_diff.py keys/compares by (arch, build_config, mangled_name).
#
# Usage:
#   device_bitcode_resource_usage_to_csv.sh <bc_file> <arch> <build_config> <commit> <out_csv>
###############################################################################
set -euo pipefail

BC_FILE="${1:?usage: device_bitcode_resource_usage_to_csv.sh <bc_file> <arch> <build_config> <commit> <out_csv>}"
ARCH="${2:?missing arch}"
BUILD_CONFIG="${3:?missing build_config}"
COMMIT="${4:?missing commit}"
OUT_CSV="${5:?missing out_csv}"

SCRIPT_DIR="$(cd "$(dirname "$(realpath "$0")")" && pwd)"
CLANGXX="/opt/rocm/llvm/bin/clang++"
command -v "$CLANGXX" >/dev/null 2>&1 || CLANGXX="clang++"

[[ -f "$BC_FILE" ]] || { echo "error: no such file: $BC_FILE" >&2; exit 1; }

WORKDIR="$(mktemp -d /tmp/rocshmem-device-bitcode-XXXXXX)"
trap 'rm -rf "$WORKDIR"' EXIT

RAW_LOG="$WORKDIR/raw.log"
SUMMARY_LOG="$WORKDIR/resource_usage_summary.log"

echo "  [device-bitcode] backend-compiling $BC_FILE (mcpu=$ARCH) for resource-usage remarks..." >&2
"$CLANGXX" -target amdgcn-amd-amdhsa -mcpu="$ARCH" \
  -Rpass-analysis=kernel-resource-usage \
  -c "$BC_FILE" -o "$WORKDIR/out.o" 2>"$RAW_LOG" || {
    echo "error: backend compile of $BC_FILE failed:" >&2
    cat "$RAW_LOG" >&2
    exit 1
  }

# Normalize `remark: <unknown>:0:0: Function Name: X [-Rpass-analysis=...]`
# (this direct backend-only invocation has no frontend source-location
# metadata to attach, unlike the production library's per-TU compiles) into the
# `<file>:<line>:<col>: Key: value` shape resource_usage_to_csv.py expects
# -- strip the "remark: " severity prefix and the trailing bracketed flag
# name, and substitute a stable placeholder location.
sed -E \
  -e 's/^remark: <unknown>:0:0:/<bitcode>:0:0:/' \
  -e 's/ \[-Rpass-analysis=kernel-resource-usage\]$//' \
  "$RAW_LOG" | grep -E '<bitcode>:0:0:' > "$SUMMARY_LOG" || true

if [[ ! -s "$SUMMARY_LOG" ]]; then
  echo "error: no kernel-resource-usage remarks found compiling $BC_FILE" >&2
  echo "--- raw compiler output ---" >&2
  cat "$RAW_LOG" >&2
  exit 1
fi

python3 "$SCRIPT_DIR/resource_usage_to_csv.py" \
  --log "$SUMMARY_LOG" \
  --arch "$ARCH" --build-config "${BUILD_CONFIG}-bitcode" --commit "$COMMIT" \
  --out "$OUT_CSV" --top 0 >&2

echo "$OUT_CSV"
