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
MIN_COPY_SIZE="${MIN_COPY_SIZE:-$((1 << 14))}"
MAX_COPY_SIZE="${MAX_COPY_SIZE:-$((1 << 20))}"
NUM_COPY_COMMANDS="${NUM_COPY_COMMANDS:-100}"
NUM_DESTINATIONS="${NUM_DESTINATIONS:-1}"
WARPS_PER_WORKGROUP="${WARPS_PER_WORKGROUP:-1}"
WORKGROUP_COUNTS="${WORKGROUP_COUNTS:-16 32 64 128 256 304}"

read -r -a workgroup_counts <<<"${WORKGROUP_COUNTS}"

require_executable "${BENCHMARK}" BENCHMARK
create_output_dir "results_contention"

echo "Running contention sweeps from ${MIN_COPY_SIZE} to " \
  "${MAX_COPY_SIZE} bytes"
for workgroups in "${workgroup_counts[@]}"; do
  result_csv="contention_${workgroups}wgs_"\
"${WARPS_PER_WORKGROUP}waves_${NUM_COPY_COMMANDS}copies.csv"
  echo "  Workgroups/destination: ${workgroups}"
  run_benchmark "${BENCHMARK}" "${result_csv}" \
    --minCopySize "${MIN_COPY_SIZE}" \
    --maxCopySize "${MAX_COPY_SIZE}" \
    --numCopyCommands "${NUM_COPY_COMMANDS}" \
    --numDestinations "${NUM_DESTINATIONS}" \
    --wgsPerDestination "${workgroups}" \
    --warpsPerWG "${WARPS_PER_WORKGROUP}"
  append_summary "${result_csv}"
done

print_results
