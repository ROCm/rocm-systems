#!/usr/bin/env bash
#
# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
#
# SPDX-License-Identifier: MIT

set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WARMUP="${WARMUP:-5}"
ITERATIONS="${ITERATIONS:-20}"
# shellcheck source=benchmark-common.sh
source "${script_dir}/benchmark-common.sh"

# USE_MPI=1 runs the MPI build, one rank per GPU, under MPIRUN (default mpirun).
USE_MPI="${USE_MPI:-0}"
MPIRUN="${MPIRUN:-mpirun}"
if [[ "${USE_MPI}" == 1 ]]; then
  ALLGATHER_BENCHMARK="${ALLGATHER_BENCHMARK:-${BUILD_DIR}/copy-engine-allgather-mpi}"
else
  ALLGATHER_BENCHMARK="${ALLGATHER_BENCHMARK:-${BUILD_DIR}/copy-engine-allgather}"
fi
MIN_GPUS="${MIN_GPUS:-2}"
MAX_GPUS="${MAX_GPUS:-8}"
MIN_BYTES="${MIN_BYTES:-4096}"
MAX_BYTES="${MAX_BYTES:-268435456}"

require_executable "${ALLGATHER_BENCHMARK}" ALLGATHER_BENCHMARK
require_positive MIN_GPUS "${MIN_GPUS}"
create_output_dir "results_allgather"

echo "Running allgather sweeps from ${MIN_BYTES} to ${MAX_BYTES} bytes"
for ((gpus = MIN_GPUS; gpus <= MAX_GPUS; gpus *= 2)); do
  result_csv="allgather_${gpus}gpus.csv"
  echo "  GPUs: ${gpus}"
  launcher=()
  gpu_args=(--numGpus "${gpus}")
  if [[ "${USE_MPI}" == 1 ]]; then
    read -ra launcher <<<"${MPIRUN}"
    launcher+=(-x DEBUG_CLR_ENABLE_ABCE -x GPU_MAX_SDMA_DEVICE_QUEUES -n "${gpus}")
    gpu_args=()
  fi
  command=("${launcher[@]}" "${ALLGATHER_BENCHMARK}" "${gpu_args[@]}"
    --minBytes "${MIN_BYTES}"
    --maxBytes "${MAX_BYTES}"
    --warmup "${WARMUP}"
    --iterations "${ITERATIONS}"
    --outputFile "${OUTPUT_DIR}/${result_csv}")
  printf 'Command:' >>"${LOG_FILE}"
  printf ' %q' "${command[@]}" >>"${LOG_FILE}"
  printf '\n' >>"${LOG_FILE}"
  if ! "${command[@]}" >>"${LOG_FILE}" 2>&1; then
    echo "ERROR: ${ALLGATHER_BENCHMARK##*/} failed; see ${LOG_FILE}" >&2
    exit 1
  fi
  append_summary "${result_csv}"
done

print_results
