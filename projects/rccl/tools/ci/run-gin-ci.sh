#!/usr/bin/env bash
# Run the RCCL/rocSHMEM GIN test suite (baremetal translation of
# docker_build_test.bash) against the freshly built rocSHMEM + RCCL + rccl-tests.
#
# Consumes the .ci-out/*.env fragments written by the build stages (ROCM_PATH,
# MPI_HOME, ROCSHMEM_INSTALL_DIR, RCCL_INSTALL_PREFIX, RCCL_TESTS_BIN_DIR) or the
# same values from the environment (gin.sbatch exports them).
#
# Tests are NON-GATING: every run is attempted and timed out individually, and a
# summary is printed, but the script still exits 0 (GIN is WIP). Set
# GIN_GATE_TESTS=1 to make any test failure fail the job.
#
# Environment overrides:
#   NP             MPI ranks per run         (default: 8)
#   MSG_SIZE       Per-rank message size     (default: 33554432 = 32 MiB)
#   BENCH_TIMEOUT  Per-test wall-clock cap   (default: 600s)
#   BENCH_KILL_AFTER  SIGKILL grace          (default: 30s)
#   CONFIG         Test-matrix JSON          (default: lib/gin-tests.json)
#   GIN_GATE_TESTS Set to 1 to fail the job on any test failure

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
# Some ROCm layouts ship bundled sysdeps here; include only if present.
[[ -d "${ROCM_PATH}/core/lib/rocm_sysdeps/lib" ]] && \
  LD_LIBRARY_PATH="${LD_LIBRARY_PATH}:${ROCM_PATH}/core/lib/rocm_sysdeps/lib"
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
TEST_NAMES=() TEST_KINDS=() TEST_BINS=() TEST_ENVS=() TEST_ARGS=()
CONFIG_TSV="$(python3 "${PARSER}" "${CONFIG}")" || {
  echo "ERROR: failed to parse test matrix ${CONFIG}" >&2; exit 1; }
while IFS=$'\t' read -r kind f1 f2 f3 f4 f5; do
  case "${kind}" in
    mca)  MCA="${f1}" ;;
    test) TEST_NAMES+=("${f1}"); TEST_KINDS+=("${f2}"); TEST_BINS+=("${f3}"); TEST_ENVS+=("${f4}"); TEST_ARGS+=("${f5}") ;;
  esac
done <<< "${CONFIG_TSV}"

if [[ ${#TEST_NAMES[@]} -eq 0 ]]; then
  echo "ERROR: no tests parsed from ${CONFIG}" >&2; exit 1
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
  if [[ "${GIN_GATE_TESTS:-}" == "1" ]]; then
    echo "GIN_GATE_TESTS=1: failing the job."
    exit 1
  fi
  echo "Tests are non-gating (GIN WIP); not failing the job."
  echo "Jira ID: AICOMRCCL-1478 Enable GIN test gating."
  exit 0
fi

echo "All GIN test runs succeeded."
