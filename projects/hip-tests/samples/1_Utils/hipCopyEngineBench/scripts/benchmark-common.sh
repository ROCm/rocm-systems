#!/usr/bin/env bash
#
# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
#
# SPDX-License-Identifier: MIT

set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
example_dir="$(cd "${script_dir}/.." && pwd)"

BUILD_DIR="${BUILD_DIR:-${example_dir}/build}"
SRC_GPU="${SRC_GPU:-0}"
WARMUP="${WARMUP:-3}"
ITERATIONS="${ITERATIONS:-50}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${PWD}}"

# The runtime only publishes kernel-side copy rings when these are set.
export DEBUG_CLR_ENABLE_ABCE="${DEBUG_CLR_ENABLE_ABCE:-1}"
export GPU_MAX_SDMA_DEVICE_QUEUES="${GPU_MAX_SDMA_DEVICE_QUEUES:-16}"

# require_executable <path> <variable that overrides it>
require_executable() {
  if [[ ! -x "$1" ]]; then
    echo "ERROR: benchmark executable not found: $1" >&2
    echo "Build it, or set $2=/path/to/executable (or BUILD_DIR)." >&2
    exit 1
  fi
}

# require_positive <name> <value>: a sweep that doubles its variable never leaves 0.
require_positive() {
  if ((${2} < 1)); then
    echo "ERROR: $1 must be at least 1" >&2
    exit 1
  fi
}

create_output_dir() {
  local prefix="$1"
  local timestamp

  timestamp="$(date '+%Y-%m-%d_%Hh%Mm%Ss')"
  OUTPUT_DIR="${OUTPUT_ROOT}/${prefix}_${timestamp}"
  SUMMARY_FILE="${OUTPUT_DIR}/summary.csv"
  LOG_FILE="${OUTPUT_DIR}/benchmark.log"
  mkdir -p "${OUTPUT_DIR}"
  : >"${SUMMARY_FILE}"
  : >"${LOG_FILE}"
}

# run_benchmark <executable> <result csv> [benchmark arguments...]
run_benchmark() {
  local executable="$1"
  local result_csv="$2"
  shift 2
  local command=("${executable}"
    --srcGpu "${SRC_GPU}"
    --warmup "${WARMUP}"
    --iterations "${ITERATIONS}"
    --outputFile "${OUTPUT_DIR}/${result_csv}"
    "$@")

  printf 'Command:' >>"${LOG_FILE}"
  printf ' %q' "${command[@]}" >>"${LOG_FILE}"
  printf '\n' >>"${LOG_FILE}"
  if ! "${command[@]}" >>"${LOG_FILE}" 2>&1; then
    echo "ERROR: ${executable##*/} failed; see ${LOG_FILE}" >&2
    exit 1
  fi
}

append_summary() {
  local result_csv="$1"

  if [[ ! -s "${SUMMARY_FILE}" ]]; then
    cat "${OUTPUT_DIR}/${result_csv}" >>"${SUMMARY_FILE}"
  else
    tail -n +2 "${OUTPUT_DIR}/${result_csv}" >>"${SUMMARY_FILE}"
  fi
}

print_results() {
  echo "Results: ${OUTPUT_DIR}"
  echo "Summary: ${SUMMARY_FILE}"
  echo "Log:     ${LOG_FILE}"
}
