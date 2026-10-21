#!/usr/bin/env bash
#
# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
#
# SPDX-License-Identifier: MIT

set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WARMUP="${WARMUP:-3}"
ITERATIONS="${ITERATIONS:-100}"
# shellcheck source=benchmark-common.sh
source "${script_dir}/benchmark-common.sh"

LATENCY_BENCHMARK="${LATENCY_BENCHMARK:-${BUILD_DIR}/copy-engine-latency}"
DST_GPU="${DST_GPU:-1}"
MIN_COPY_SIZE="${MIN_COPY_SIZE:-256}"
MAX_COPY_SIZE="${MAX_COPY_SIZE:-1048576}"
NUM_COPY_COMMANDS="${NUM_COPY_COMMANDS:-1}"

require_executable "${LATENCY_BENCHMARK}" LATENCY_BENCHMARK
require_positive MIN_COPY_SIZE "${MIN_COPY_SIZE}"
create_output_dir "results_latency"

echo "Running latency sweep from ${MIN_COPY_SIZE} to ${MAX_COPY_SIZE} bytes"
for ((size = MIN_COPY_SIZE; size <= MAX_COPY_SIZE; size *= 2)); do
  result_csv="latency_${size}.csv"
  echo "  Copy size: ${size}"
  run_benchmark "${LATENCY_BENCHMARK}" "${result_csv}" \
    --dstGpu "${DST_GPU}" \
    --minCopySize "${size}" \
    --maxCopySize "${size}" \
    --numCopyCommands "${NUM_COPY_COMMANDS}"
  append_summary "${result_csv}"
done

print_results
