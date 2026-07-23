#!/usr/bin/env bash
# Run the RCCL/rocSHMEM GIN test suite (baremetal translation of
# docker_build_test.bash) against the freshly built rocSHMEM + RCCL + rccl-tests.
#
# Consumes the .ci-out/*.env fragments written by the build stages (ROCM_PATH,
# MPI_HOME, ROCSHMEM_INSTALL_DIR, RCCL_INSTALL_PREFIX, RCCL_TESTS_BIN_DIR) or the
# same values from the environment (gin.sbatch exports them).
#
# Each run is timed and summarized; exits non-zero if any failed. gin.sbatch keeps
# test failures non-gating (separate red check). Jira AICOMRCCL-1478: Enable GIN test gating.
#
# Environment overrides:
#   NP             MPI ranks per run         (default: 8)
#   MSG_SIZE       Per-rank message size     (default: 33554432 = 32 MiB)
#   BENCH_TIMEOUT  Per-test wall-clock cap   (default: 600s)
#   BENCH_KILL_AFTER  SIGKILL grace          (default: 30s)
#   CONFIG         Test-matrix JSON          (default: lib/gin-tests.json)
#   RCCL_CI_DEBUG  Set to 1 to add the config's debug_env to every run
#   RCCL_CI_DEBUG_DIR  Dir for NCCL_DEBUG_FILE output when RCCL_CI_DEBUG=1
#                  (default: ${SLURM_SUBMIT_DIR:-$PWD}/nccl-debug)

set -euo pipefail

NP="${NP:-8}"
MSG_SIZE="${MSG_SIZE:-33554432}"
BENCH_TIMEOUT="${BENCH_TIMEOUT:-600s}"
BENCH_KILL_AFTER="${BENCH_KILL_AFTER:-30s}"

script_dir="$(cd "$(dirname "$0")" && pwd)"
WORKDIR="$(cd "${script_dir}/../../../.." && pwd)"
CONFIG="${CONFIG:-${script_dir}/lib/gin-tests.json}"
PARSER="${script_dir}/lib/parse_gin_config.py"

for frag in rocm ompi rocshmem rccl; do
  f="${WORKDIR}/.ci-out/${frag}.env"
  # shellcheck source=/dev/null
  [[ -f "${f}" ]] && source "${f}"
done

: "${ROCM_PATH:?run-gin-ci.sh: ROCM_PATH unset (run fetch-rocm.sh / via sbatch)}"
: "${MPI_HOME:?run-gin-ci.sh: MPI_HOME unset (run build-ompi.sh / via sbatch)}"
: "${ROCSHMEM_INSTALL_DIR:?run-gin-ci.sh: ROCSHMEM_INSTALL_DIR unset (run build-rocshmem.sh)}"
: "${RCCL_INSTALL_PREFIX:?run-gin-ci.sh: RCCL_INSTALL_PREFIX unset (set by gin.sbatch)}"
: "${RCCL_TESTS_BIN_DIR:?run-gin-ci.sh: RCCL_TESTS_BIN_DIR unset (set by gin.sbatch)}"
# build-rocshmem.sh records where the test binary landed (bin/ vs share/rocshmem/);
# fall back to bin/ for older env fragments.
ROCSHMEM_TESTS_BIN_DIR="${ROCSHMEM_TESTS_BIN_DIR:-${ROCSHMEM_INSTALL_DIR}/bin}"

[[ -f "${CONFIG}" ]] || { echo "ERROR: test-matrix config not found: ${CONFIG}" >&2; exit 1; }
[[ -f "${PARSER}" ]] || { echo "ERROR: config parser not found: ${PARSER}" >&2; exit 1; }

export PATH="${MPI_HOME}/bin:${ROCM_PATH}/bin:${ROCM_PATH}/llvm/bin:${PATH}"
LD_LIBRARY_PATH="${ROCSHMEM_INSTALL_DIR}/lib:${RCCL_INSTALL_PREFIX}/lib:${MPI_HOME}/lib:${ROCM_PATH}/lib:${LD_LIBRARY_PATH:-}"
# ROCm ships bundled sysdeps (librocm_sysdeps_numa.so.1, drm, ...) that rocSHMEM
# links against; the dir moves between layouts, so add whichever exists.
for _sysdeps in "${ROCM_PATH}/lib/rocm_sysdeps/lib" "${ROCM_PATH}/core/lib/rocm_sysdeps/lib"; do
  [[ -d "${_sysdeps}" ]] && LD_LIBRARY_PATH="${LD_LIBRARY_PATH}:${_sysdeps}"
done
export LD_LIBRARY_PATH
export OMPI_ALLOW_RUN_AS_ROOT=1 OMPI_ALLOW_RUN_AS_ROOT_CONFIRM=1

# -e end-size used by the rccl-tests runs and rocSHMEM -v: NP * MSG_SIZE bytes.
E=$(( NP * MSG_SIZE ))

echo "==> ROCM_PATH            = ${ROCM_PATH}"
echo "==> MPI_HOME             = ${MPI_HOME}"
echo "==> ROCSHMEM_INSTALL_DIR = ${ROCSHMEM_INSTALL_DIR}"
echo "==> RCCL_INSTALL_PREFIX  = ${RCCL_INSTALL_PREFIX}"
echo "==> RCCL_TESTS_BIN_DIR   = ${RCCL_TESTS_BIN_DIR}"
echo "==> NP=${NP} MSG_SIZE=${MSG_SIZE} (E=NP*MSG_SIZE=${E})"
echo "==> test matrix          = ${CONFIG}"

MCA=""
DEBUG_ENV=""
TEST_NAMES=() TEST_KINDS=() TEST_BINS=() TEST_ENVS=() TEST_ARGS=()
CONFIG_TSV="$(python3 "${PARSER}" "${CONFIG}")" || {
  echo "ERROR: failed to parse test matrix ${CONFIG}" >&2; exit 1; }
while IFS=$'\t' read -r kind f1 f2 f3 f4 f5; do
  case "${kind}" in
    mca)        MCA="${f1}" ;;
    debug_env)  DEBUG_ENV="${f1}" ;;
    test)       TEST_NAMES+=("${f1}"); TEST_KINDS+=("${f2}"); TEST_BINS+=("${f3}"); TEST_ENVS+=("${f4}"); TEST_ARGS+=("${f5}") ;;
  esac
done <<< "${CONFIG_TSV}"

if [[ ${#TEST_NAMES[@]} -eq 0 ]]; then
  echo "ERROR: no tests parsed from ${CONFIG}" >&2; exit 1
fi

# In debug mode, expand {LOGDIR} in debug_env to a real per-job dir so
# NCCL_DEBUG_FILE writes per-rank logs there (keeping stdout readable).
if [[ -n "${RCCL_CI_DEBUG:-}" && -n "${DEBUG_ENV}" ]]; then
  RCCL_CI_DEBUG_DIR="${RCCL_CI_DEBUG_DIR:-${SLURM_SUBMIT_DIR:-$(pwd)}/nccl-debug}"
  mkdir -p "${RCCL_CI_DEBUG_DIR}"
  DEBUG_ENV="${DEBUG_ENV//\{LOGDIR\}/${RCCL_CI_DEBUG_DIR}}"
  echo "==> RCCL_CI_DEBUG=1: NCCL debug logs -> ${RCCL_CI_DEBUG_DIR}/nccl-debug.<host>.<pid>.log"
fi
echo "==> ${#TEST_NAMES[@]} tests to run: ${TEST_NAMES[*]}"

FAILED_RUNS=()

# Word-splitting on flag/arg vars below is intentional.
# shellcheck disable=SC2086
run_test() {
  local name="$1" kind="$2" bin="$3" env_flags="$4" args="$5"
  local bin_path
  case "${kind}" in
    rocshmem)   bin_path="${ROCSHMEM_TESTS_BIN_DIR}/${bin}" ;;
    rccl-tests) bin_path="${RCCL_TESTS_BIN_DIR}/${bin}" ;;
    *) echo "  SKIP ${name}: unknown kind '${kind}'"; FAILED_RUNS+=("${name} (unknown kind)"); return ;;
  esac
  if [[ ! -x "${bin_path}" ]]; then
    echo "  SKIP ${name}: binary not found/executable: ${bin_path}"
    FAILED_RUNS+=("${name} (missing ${bin})")
    return
  fi
  args="${args//\{E\}/${E}}"
  if [[ -n "${RCCL_CI_DEBUG:-}" && -n "${DEBUG_ENV}" ]]; then
    env_flags="${env_flags} ${DEBUG_ENV}"
  fi
  echo "=== ${name}: ${bin} ${args} ==="
  set +e
  timeout --kill-after="${BENCH_KILL_AFTER}" "${BENCH_TIMEOUT}" \
    mpirun -np "${NP}" ${MCA} ${env_flags} -x LD_LIBRARY_PATH \
      "${bin_path}" ${args}
  local rc=$?
  set -e
  if [[ ${rc} -ne 0 ]]; then
    if [[ ${rc} -eq 124 || ${rc} -eq 137 ]]; then
      FAILED_RUNS+=("${name} (TIMEOUT >${BENCH_TIMEOUT}, rc=${rc})")
    else
      FAILED_RUNS+=("${name} (rc=${rc})")
    fi
  fi
}

for i in "${!TEST_NAMES[@]}"; do
  run_test "${TEST_NAMES[$i]}" "${TEST_KINDS[$i]}" "${TEST_BINS[$i]}" "${TEST_ENVS[$i]}" "${TEST_ARGS[$i]}"
done

if [[ ${#FAILED_RUNS[@]} -ne 0 ]]; then
  echo "=== FAILED / SKIPPED RUNS (${#FAILED_RUNS[@]} of ${#TEST_NAMES[@]}) ==="
  printf '  %s\n' "${FAILED_RUNS[@]}"
  # Exit non-zero on failure; gin.sbatch turns this into a non-gating red check.
  exit 1
fi

echo "All GIN test runs succeeded."
