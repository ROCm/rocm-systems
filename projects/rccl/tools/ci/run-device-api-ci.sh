#!/usr/bin/env bash
# Run the RCCL device-API benchmark suite (symmetric memory / LSA device
# kernels / GIN proxy) against a freshly built rccl-tests tree.
#
# Mirrors the legacy rocJenkins "device-api" testCommand. Consumes the trees the
# earlier sbatch stages produced:
#   $ROCM_PATH                              Cached ROCm tree (from fetch-rocm.sh)
#   $MPI_HOME                               OpenMPI install   (from build-ompi.sh)
#   $WORKDIR/projects/rccl/build/release    librccl.so (in-tree RCCL build)
#   $WORKDIR/rccl-tests/build               rccl-tests perf binaries
# where $WORKDIR == the rocm-systems checkout root.
#
# ROCM_PATH / MPI_HOME are normally already in the environment (device-api.sbatch
# exports them). When run standalone they are read from the env fragments the
# build stages wrote ($WORKDIR/.ci-out/{rocm,ompi}.env).
#
# Each benchmark is wrapped in `timeout` so a hung mpirun / driver (e.g. GIN
# dmabuf failures wedging ranks in unkillable HSA waits) cannot wedge the
# whole CI job. Failures are collected into FAILED_RUNS and surfaced at the
# end; the script exits non-zero iff any run failed.
#
# Environment overrides:
#   ROCM_PATH           ROCm tree (else read from .ci-out/rocm.env).
#   MPI_HOME            OpenMPI install (else read from .ci-out/ompi.env).
#   NP                  MPI ranks per run         (default: 8)
#   BENCH_ARGS          Common bench args         (default: "-b 8 -e 1G -f 2 -g 1")
#   BENCH_TIMEOUT       Per-bench wall-clock cap  (default: 600s)
#   BENCH_KILL_AFTER    SIGKILL grace after BENCH_TIMEOUT (default: 30s)

set -euo pipefail

NP="${NP:-8}"
BENCH_ARGS="${BENCH_ARGS:--b 8 -e 1G -f 2 -g 1}"
BENCH_TIMEOUT="${BENCH_TIMEOUT:-600s}"
BENCH_KILL_AFTER="${BENCH_KILL_AFTER:-30s}"

script_dir="$(cd "$(dirname "$0")" && pwd)"
RCCL_DIR="$(cd "${script_dir}/../.." && pwd)"
WORKDIR="$(cd "${RCCL_DIR}/../.." && pwd)"

RCCL_LIB_DIR="${WORKDIR}/projects/rccl/build/release"
RCCL_TESTS_DIR="${WORKDIR}/rccl-tests"

# The env fragments the build stages wrote are the authoritative record of what
# was actually built; prefer them over any ambient ROCM_PATH / MPI_HOME so a
# stray value from the node environment cannot shadow the trees under test.
# shellcheck source=/dev/null  # runtime fragment written by fetch-rocm.sh
[[ -f "${WORKDIR}/.ci-out/rocm.env" ]] && source "${WORKDIR}/.ci-out/rocm.env"
# shellcheck source=/dev/null  # runtime fragment written by build-ompi.sh
[[ -f "${WORKDIR}/.ci-out/ompi.env" ]] && source "${WORKDIR}/.ci-out/ompi.env"

: "${ROCM_PATH:?run-device-api-ci.sh: ROCM_PATH unset (run fetch-rocm.sh / via sbatch)}"
: "${MPI_HOME:?run-device-api-ci.sh: MPI_HOME unset (run build-ompi.sh / via sbatch)}"

if [[ ! -x "${ROCM_PATH}/bin/hipcc" ]]; then
  echo "ERROR: ROCM_PATH=${ROCM_PATH} does not look like a ROCm tree (no bin/hipcc)" >&2
  exit 1
fi
if [[ ! -f "${MPI_HOME}/lib/libmpi.so" ]]; then
  echo "ERROR: no libmpi.so at ${MPI_HOME}/lib (did build-ompi.sh run?)" >&2
  exit 1
fi

echo "==> ROCM_PATH = ${ROCM_PATH}"
echo "==> MPI_HOME  = ${MPI_HOME}"
echo "==> RCCL libs = ${RCCL_LIB_DIR}"

if [[ ! -f "${RCCL_LIB_DIR}/librccl.so" && ! -f "${RCCL_LIB_DIR}/librccl.so.1" ]]; then
  echo "librccl.so not found under ${RCCL_LIB_DIR}"
  ls -la "${RCCL_LIB_DIR}" 2>/dev/null || echo "(directory missing)"
  exit 1
fi

cd "${RCCL_TESTS_DIR}"

export PATH="${MPI_HOME}/bin:${ROCM_PATH}/bin:${PATH}"
export LD_LIBRARY_PATH="${RCCL_LIB_DIR}:${MPI_HOME}/lib:${ROCM_PATH}/lib:${LD_LIBRARY_PATH:-}"

PERF_DIR=build
if [[ ! -d "${PERF_DIR}" || ! -f "${PERF_DIR}/all_gather_perf" ]]; then
  echo "rccl-tests perf binaries not found under ${PERF_DIR}"
  ls -la
  exit 1
fi

# Supported bins per device-api feature. Extend as new device-kernel impls land.
SYMMETRIC_BINS="broadcast_perf reduce_perf sendrecv_perf scatter_perf gather_perf"
LSA_BINS="all_reduce_perf"
GIN_BINS="alltoall_perf"

BASE_ENV="-x NCCL_CUMEM_ENABLE=1 -x HSA_FORCE_FINE_GRAIN_PCIE=1"
GIN_ENV="${BASE_ENV} -x NCCL_DMABUF_ENABLE=1 -x NCCL_GIN_TYPE=2 -x HSA_NO_SCRATCH_RECLAIM=1 -x NCCL_ENV_PLUGIN=none"

FAILED_RUNS=()

# Word-splitting on $env_flags / $BENCH_ARGS / $extra_args is intentional so
# the mpirun flag list can grow without per-call array gymnastics.
# shellcheck disable=SC2086
run_bench() {
  local mode="$1"
  local bin="$2"
  local env_flags="$3"
  shift 3
  local extra_args="$*"
  echo "=== ${mode}: ${bin} ${extra_args} ==="
  set +e
  timeout --kill-after="${BENCH_KILL_AFTER}" "${BENCH_TIMEOUT}" \
    mpirun -np "${NP}" ${env_flags} -x LD_LIBRARY_PATH \
      "./${PERF_DIR}/${bin}" ${BENCH_ARGS} ${extra_args}
  local rc=$?
  set -e
  if [[ ${rc} -ne 0 ]]; then
    # timeout(1): rc=124 = SIGTERM after BENCH_TIMEOUT, rc=137 = SIGKILL after BENCH_KILL_AFTER.
    if [[ ${rc} -eq 124 || ${rc} -eq 137 ]]; then
      FAILED_RUNS+=("${mode}:${bin} (TIMEOUT >${BENCH_TIMEOUT}, rc=${rc})")
    else
      FAILED_RUNS+=("${mode}:${bin} (rc=${rc})")
    fi
  fi
}

run_collective_suite() {
  local bins="$1"
  local mode="$2"
  local env_flags="$3"
  shift 3
  for bin in ${bins}; do
    run_bench "${mode}" "${bin}" "${env_flags}" "$@"
  done
}

echo "=== 1) Symmetric memory ==="
run_collective_suite "${SYMMETRIC_BINS}" symmetric "${BASE_ENV}" -R 2

echo "=== 2) LSA device kernels ==="
for d in 1 2; do
  run_collective_suite "${LSA_BINS}" "lsa-d${d}" "${BASE_ENV}" -R 2 -D "${d}"
done

echo "=== 3) GIN proxy ==="
for d in 3 4; do
  run_collective_suite "${GIN_BINS}" "gin-d${d}" "${GIN_ENV}" -R 2 -D "${d}"
done

if [[ ${#FAILED_RUNS[@]} -ne 0 ]]; then
  echo "=== FAILED RUNS (${#FAILED_RUNS[@]}) ==="
  printf '  %s\n' "${FAILED_RUNS[@]}"
  exit 1
fi

echo "All device-api benchmark runs succeeded."
