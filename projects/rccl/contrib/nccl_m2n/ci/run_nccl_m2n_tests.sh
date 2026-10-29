#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0

# Run host geometry checks and the no-MPI basic API matrix within an existing
# Slurm allocation. The API binary creates one local pthread per GPU, so srun
# launches one process only.

set -euo pipefail

: "${NCCL_HOME:?NCCL_HOME must be set}"
numGpus="${NCCL_M2N_TEST_GPUS:-${NGPUS:-}}"

if [[ -z "${numGpus}" ]]; then
  numGpus="$(nvidia-smi -L 2>/dev/null | wc -l)"
fi
if ! [[ "${numGpus}" =~ ^[1-9][0-9]*$ ]]; then
  echo "NCCL_M2N_TEST_GPUS/NGPUS must name at least one GPU; got '${numGpus}'." >&2
  exit 2
fi

export LD_LIBRARY_PATH="${NCCL_HOME}/lib:${LD_LIBRARY_PATH:-}"
if [[ -n "${CUDA_HOME:-}" ]]; then
  export LD_LIBRARY_PATH="${CUDA_HOME}/lib64:${CUDA_HOME}/lib:${LD_LIBRARY_PATH}"
fi

if [[ "${1:-}" != "--inside-srun" && -n "${SLURM_JOB_ID:-}" ]]; then
  exec srun -N 1 -n 1 --exclusive --cpu-bind=none --export=ALL \
    bash "$0" --inside-srun
fi

binary="${NCCL_HOME}/bin/basic_api_test_local"
if [[ ! -x "${binary}" ]]; then
  echo "Missing executable basic API test: ${binary}" >&2
  exit 1
fi

log="$(mktemp)"
trap 'rm -f "${log}"' EXIT

geometryBinary="${NCCL_HOME}/bin/multi_dim_geometry_test"
if [[ ! -x "${geometryBinary}" ]]; then
  echo "Missing executable geometry test: ${geometryBinary}" >&2
  exit 1
fi
"${geometryBinary}" 2>&1 | tee "${log}"
if ! grep -Fxq 'MULTI_DIM_GEOMETRY_PASS pairs=109872' "${log}"; then
  echo "multi_dim_geometry_test did not complete all 109872 geometry pairs." >&2
  exit 1
fi

"${binary}" -N "${numGpus}" 2>&1 | tee "${log}"

if (( numGpus >= 8 )); then
  pipeLog="$(mktemp)"
  trap 'rm -f "${log}" "${pipeLog}"' EXIT
  if ! "${binary}" -N 8 --filter pipe_lsa_fanout_reuse --algorithm ring --api window --copy-algorithm pipe \
    >"${pipeLog}" 2>&1; then
    cat "${pipeLog}"
    exit 1
  fi
  cat "${pipeLog}"
  if ! grep -Eq '^worldSize=8, devices=[1-9][0-9]*,' "${pipeLog}"; then
    echo "PIPE fanout reuse run did not execute with 8 local ranks." >&2
    exit 1
  fi
  if ! grep -Eq '^\[       OK \] Matrix/BasicApiLocalTest.Reshard/.*pipe_lsa_fanout_reuse' "${pipeLog}"; then
    echo "PIPE fanout reuse run did not complete its matrix case." >&2
    exit 1
  fi
fi

if ! grep -Eq "^worldSize=${numGpus}, devices=[1-9][0-9]*," "${log}"; then
  echo "basic_api_test_local did not run with ${numGpus} local ranks." >&2
  exit 1
fi

if ! grep -Eq '^\[  PASSED  \] [1-9][0-9]* tests?\.' "${log}"; then
  echo "basic_api_test_local completed without any passing test cases." >&2
  exit 1
fi
