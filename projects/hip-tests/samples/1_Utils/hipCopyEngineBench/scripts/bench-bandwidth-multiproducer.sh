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
MAX_COPY_SIZE="${MAX_COPY_SIZE:-$((1 << 20))}"
NUM_COPY_COMMANDS="${NUM_COPY_COMMANDS:-100}"
NUM_DESTINATIONS="${NUM_DESTINATIONS:-7}"
MIN_WORKGROUPS_PER_DESTINATION="${MIN_WORKGROUPS_PER_DESTINATION:-2}"
MAX_WORKGROUPS_PER_DESTINATION="${MAX_WORKGROUPS_PER_DESTINATION:-32}"
WARPS_PER_WORKGROUP="${WARPS_PER_WORKGROUP:-1}"

require_executable "${BENCHMARK}" BENCHMARK
require_positive MIN_WORKGROUPS_PER_DESTINATION "${MIN_WORKGROUPS_PER_DESTINATION}"
create_output_dir "results_bandwidth_multiproducer"

echo "Running multi-producer bandwidth sweeps from ${MIN_COPY_SIZE} to " \
  "${MAX_COPY_SIZE} bytes"
for ((workgroups = MIN_WORKGROUPS_PER_DESTINATION;
      workgroups <= MAX_WORKGROUPS_PER_DESTINATION;
      workgroups *= 2)); do
  result_csv="bandwidth_${NUM_DESTINATIONS}dst_${workgroups}wgs_"\
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
