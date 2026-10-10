#!/usr/bin/env bash

# Copyright (c) 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT
#
# From the pinned runtime CTS corpus checkout:
#   bash run-runtime-cts.sh build
#   bash run-runtime-cts.sh run [pytest options]
# Optional: ROCJITSU_SOURCE_DIR, ROCJITSU_BUILD_DIR,
#           ROCJITSU_RUNTIME_CTS_CORPUS_DIR (defaults to the working directory).
set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source_dir="$(realpath "${ROCJITSU_SOURCE_DIR:-${script_dir}/../..}")"
mode="${1:-run}"
if (( $# )); then shift; fi
case "${mode}" in
  build) if (( $# )); then echo "build takes no arguments" >&2; exit 2; fi ;;
  run) ;;
  *) echo "Usage: $0 build|run [pytest options]" >&2; exit 2 ;;
esac

# Build and run must use the same architectures, configs, and suites.
targets=(
  'gfx1250 gfx1250_mi455x.json'
  'gfx950 gfx950_mi355x_kmd.json'
  'gfx942 gfx942_cdna3_kmd.json'
  'gfx1201 gfx1201_r9700.json'
  'gfx1100 gfx1100_w7900.json'
)
suites=(aql)

if [[ "${mode}" == run ]]; then
  launcher="${ROCJITSU_BUILD_DIR:-${source_dir}/build}/tools/rocjitsu/rocjitsu"
  if [[ ! -x "${launcher}" ]]; then
    echo "Missing executable Rocjitsu launcher: ${launcher}. Set ROCJITSU_BUILD_DIR." >&2
    exit 1
  fi
  launcher="$(realpath "${launcher}")"
fi
cd "${ROCJITSU_RUNTIME_CTS_CORPUS_DIR:-${PWD}}"

if [[ "${mode}" == build ]]; then
  architectures=()
  for entry in "${targets[@]}"; do
    read -r target config <<< "${entry}"
    architectures+=("${target}")
  done
  cmake -S corpus/runtime-cts -B build/runtime-cts -G Ninja \
    -DCMAKE_BUILD_TYPE=Release \
    "-DCTS_SUITES=$(IFS=';'; echo "${suites[*]}")" \
    "-DCTS_ARCHS=$(IFS=';'; echo "${architectures[*]}")" \
    "-DAMDGPU_LLVM_BIN=$(rocm-sdk path --root)/lib/llvm/bin"
  cmake --build build/runtime-cts --parallel "$(nproc)"
  exit 0
fi

mkdir -p .pytest-artifacts/runtime-cts/junit
status=0
for suite in "${suites[@]}"; do
  for entry in "${targets[@]}"; do
    read -r target config <<< "${entry}"
    printf -v wrapper '%q ' "${launcher}" --config "${source_dir}/configs/${config}" --
    python3 -m pytest tests/test_corpus.py --suite "${suite}" --target "${target}" \
      --binary-dir build/runtime-cts --run-wrapper "${wrapper}" \
      --artifact-directory .pytest-artifacts/runtime-cts \
      --junitxml ".pytest-artifacts/runtime-cts/junit/${suite}-${target}.xml" \
      -v -ra "$@" || status=1
  done
done
exit "${status}"
