#!/usr/bin/env bash
#
# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
#
# SPDX-License-Identifier: MIT

set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=benchmark-common.sh
source "${script_dir}/benchmark-common.sh"

BENCHMARK="${BENCHMARK:-${BUILD_DIR}/copy-engine-bw}"
MIN_COPY_SIZE="${MIN_COPY_SIZE:-$((1 << 10))}"
MAX_COPY_SIZE="${MAX_COPY_SIZE:-$((1 << 30))}"
MIN_DESTINATIONS="${MIN_DESTINATIONS:-1}"
MAX_DESTINATIONS="${MAX_DESTINATIONS:-7}"

require_executable "${BENCHMARK}" BENCHMARK
create_output_dir "results_bandwidth_single"

echo "Running single-producer bandwidth sweeps from ${MIN_COPY_SIZE} to " \
  "${MAX_COPY_SIZE} bytes"
for ((destinations = MIN_DESTINATIONS;
      destinations <= MAX_DESTINATIONS;
      ++destinations)); do
  result_csv="bandwidth_${destinations}dst.csv"
  echo "  Destinations: ${destinations}"
  run_benchmark "${BENCHMARK}" "${result_csv}" \
    --minCopySize "${MIN_COPY_SIZE}" \
    --maxCopySize "${MAX_COPY_SIZE}" \
    --numCopyCommands 1 \
    --numDestinations "${destinations}"
  append_summary "${result_csv}"
done

print_results
