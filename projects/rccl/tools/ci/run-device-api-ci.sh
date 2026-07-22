#!/usr/bin/env bash
# Run the RCCL device-API benchmark suite (symmetric memory / LSA device kernels
# / GIN proxy) against the freshly built rccl-tests tree. Mirrors the legacy
# rocJenkins "device-api" testCommand.
#
# Consumes ROCM_PATH (rocm.env), MPI_HOME (ompi.env), the in-tree RCCL build at
# projects/rccl/build/release, and rccl-tests/build. ROCM_PATH / MPI_HOME come
# from the environment (device-api.sbatch exports them) or, when run standalone,
# from $WORKDIR/.ci-out/{rocm,ompi}.env.
#
# Each bench is wrapped in `timeout` so a hung mpirun/driver can't wedge the job;
# failures are collected and surfaced at the end (exit non-zero iff any failed).
#
# Test matrix lives in lib/device-api-tests.json; RCCL_CI_DEBUG=1 adds its
# debug_env to every run.
#
# Environment overrides:
#   ROCM_PATH / MPI_HOME   Else read from .ci-out/{rocm,ompi}.env
#   NP                     MPI ranks per run         (default: 8)
#   BENCH_ARGS             Common bench args         (default: from JSON bench_args)
#   BENCH_TIMEOUT          Per-bench wall-clock cap  (default: 600s)
#   BENCH_KILL_AFTER       SIGKILL grace after BENCH_TIMEOUT (default: 30s)
#   CONFIG                 Test-matrix JSON          (default: lib/device-api-tests.json)
#   RCCL_CI_DEBUG          Set to 1 to add debug_env to every run
#   RCCL_CI_DEBUG_DIR      Dir for NCCL_DEBUG_FILE output when RCCL_CI_DEBUG=1
#                          (default: ${SLURM_SUBMIT_DIR:-$PWD}/nccl-debug)

set -euo pipefail

NP="${NP:-8}"
BENCH_TIMEOUT="${BENCH_TIMEOUT:-600s}"
BENCH_KILL_AFTER="${BENCH_KILL_AFTER:-30s}"

script_dir="$(cd "$(dirname "$0")" && pwd)"
RCCL_DIR="$(cd "${script_dir}/../.." && pwd)"
WORKDIR="$(cd "${RCCL_DIR}/../.." && pwd)"

RCCL_LIB_DIR="${WORKDIR}/projects/rccl/build/release"
RCCL_TESTS_DIR="${WORKDIR}/projects/rccl-tests"
CONFIG="${CONFIG:-${script_dir}/lib/device-api-tests.json}"

# Prefer the build stages' env fragments over any ambient ROCM_PATH / MPI_HOME.
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
if [[ ! -d "${PERF_DIR}" || ! -f "${PERF_DIR}/all_reduce_perf" ]]; then
  echo "rccl-tests perf binaries not found under ${PERF_DIR}"
  ls -la
  exit 1
fi

PARSER="${script_dir}/lib/parse_device_api_config.py"
[[ -f "${CONFIG}" ]] || { echo "ERROR: test-matrix config not found: ${CONFIG}" >&2; exit 1; }
[[ -f "${PARSER}" ]] || { echo "ERROR: config parser not found: ${PARSER}" >&2; exit 1; }
echo "==> test matrix = ${CONFIG}"

# Capture to a var (not `mapfile < <(...)`) so a parser failure aborts the job.
CONFIG_TSV="$(python3 "${PARSER}" "${CONFIG}")" || {
  echo "ERROR: failed to parse test matrix ${CONFIG}" >&2
  exit 1
}
mapfile -t CONFIG_RECORDS <<< "${CONFIG_TSV}"

CFG_BENCH_ARGS=""
DEBUG_ENV=""
SUITE_NAMES=() SUITE_ENVS=() SUITE_ARGS=() SUITE_BINS=()
for rec in "${CONFIG_RECORDS[@]}"; do
  IFS=$'\t' read -r kind f1 f2 f3 f4 <<< "${rec}"
  case "${kind}" in
    bench_args) CFG_BENCH_ARGS="${f1}" ;;
    debug_env)  DEBUG_ENV="${f1}" ;;
    suite)      SUITE_NAMES+=("${f1}"); SUITE_ENVS+=("${f2}"); SUITE_ARGS+=("${f3}"); SUITE_BINS+=("${f4}") ;;
  esac
done

if [[ ${#SUITE_NAMES[@]} -eq 0 ]]; then
  echo "ERROR: no suites parsed from ${CONFIG}; refusing to report success" >&2
  exit 1
fi
echo "==> ${#SUITE_NAMES[@]} suites to run: ${SUITE_NAMES[*]}"

# In debug mode, expand {LOGDIR} in debug_env to a real per-job dir so
# NCCL_DEBUG_FILE writes per-rank logs there (keeping stdout readable).
if [[ -n "${RCCL_CI_DEBUG:-}" && -n "${DEBUG_ENV}" ]]; then
  RCCL_CI_DEBUG_DIR="${RCCL_CI_DEBUG_DIR:-${SLURM_SUBMIT_DIR:-$(pwd)}/nccl-debug}"
  mkdir -p "${RCCL_CI_DEBUG_DIR}"
  DEBUG_ENV="${DEBUG_ENV//\{LOGDIR\}/${RCCL_CI_DEBUG_DIR}}"
  echo "==> RCCL_CI_DEBUG=1: NCCL debug logs -> ${RCCL_CI_DEBUG_DIR}/nccl-debug.<host>.<pid>.log"
fi

BENCH_ARGS="${BENCH_ARGS:-${CFG_BENCH_ARGS}}"

FAILED_RUNS=()

# Word-splitting on the flag vars below is intentional.
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

for i in "${!SUITE_NAMES[@]}"; do
  mode="${SUITE_NAMES[$i]}"
  env_flags="${SUITE_ENVS[$i]}"
  extra_args="${SUITE_ARGS[$i]}"
  if [[ -n "${RCCL_CI_DEBUG:-}" && -n "${DEBUG_ENV}" ]]; then
    env_flags="${env_flags} ${DEBUG_ENV}"
  fi
  echo "=== suite: ${mode} ==="
  # shellcheck disable=SC2086  # intentional word-splitting
  for bin in ${SUITE_BINS[$i]}; do
    run_bench "${mode}" "${bin}" "${env_flags}" ${extra_args}
  done
done

if [[ ${#FAILED_RUNS[@]} -ne 0 ]]; then
  echo "=== FAILED RUNS (${#FAILED_RUNS[@]}) ==="
  printf '  %s\n' "${FAILED_RUNS[@]}"
  exit 1
fi

echo "All device-api benchmark runs succeeded."
