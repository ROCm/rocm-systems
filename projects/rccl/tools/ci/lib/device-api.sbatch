#!/usr/bin/env bash
#SBATCH --job-name=rccl-device-api
#SBATCH --nodes=1
#SBATCH --ntasks=8
#SBATCH --partition=interactive
#SBATCH --exclusive
#SBATCH --time=04:00:00
#SBATCH --output=%x-%j.out
#SBATCH --error=%x-%j.err
#SBATCH --propagate=NONE
#SBATCH --qos=vip_prio
#
# Compute-node orchestrator for the RCCL device-API CI job, invoked from
# .github/workflows/rccl-suite.yml (suite=device-api) via submit_slurm_job.py.
# On one 8-GPU node:
#   1. consume ROCm (provisioned on the head node) via .ci-out/rocm.env
#   2. lib/build-ompi.sh     build OpenMPI (--with-rocm) -> ompi.env
#   3. projects/rccl/install.sh   build RCCL (--rocshmem-gin). install.sh builds
#                                  the sibling rocSHMEM checkout into
#                                  projects/rccl/ext/rocshmem -> rocshmem.env
#   4. rccl-tests (install.sh)    ENABLE_DEVICE_API=1, no ENABLE_ROCSHMEM_GIN
#                                  (AR/AG/RS -R 2 + LSA all_reduce -D 1/-D 2)
#   5. rccl-tests (cmake)         ENABLE_DEVICE_API + ENABLE_ROCSHMEM_GIN
#                                  (GIN host-proxy -D 3/-D 4 alltoall)
#   6. run-device-api-ci.sh       non-GIN matrix, then GIN matrix
#
# Required env (from the workflow via `sbatch --export=ALL,...`):
#   RCCL_DEVICE_API_WORKDIR   GHA workspace (rocm-systems root)
# Optional env:
#   RCCL_AMDGPU_TARGETS       RCCL --amdgpu_targets (default: gfx950)
#   RCCL_DEVICE_API_CACHE     Cache root, shared with head node (default: /apps/rccl-ci)

set -euo pipefail

echo "=== sbatch device-api on $(hostname) ==="
date -Is
env | grep -E '^SLURM_' | sort || true
echo
echo "ROCM_RELEASE=${ROCM_RELEASE:-<unset: resolve latest nightly>}"
echo "ROCM_AMDGPU_FAMILY=${ROCM_AMDGPU_FAMILY:-<default: gfx950-dcgpu>}"
echo "RCCL_AMDGPU_TARGETS=${RCCL_AMDGPU_TARGETS:-<default: gfx950>}"
echo "RCCL_DEVICE_API_CACHE=${RCCL_DEVICE_API_CACHE:-<default: /apps/rccl-ci>}"
echo

workdir="${RCCL_DEVICE_API_WORKDIR:?RCCL_DEVICE_API_WORKDIR must be exported from the workflow}"
cd "${workdir}"

ci_dir="${workdir}/projects/rccl/tools/ci"
amdgpu_targets="${RCCL_AMDGPU_TARGETS:-gfx950}"

# Phase-split logs into build-%j and test-%j (SLURM_SUBMIT_DIR = the *-logs dir).
logs_dir="${SLURM_SUBMIT_DIR:-$(pwd)}"
jid="${SLURM_JOB_ID:-local}"
build_out="${logs_dir}/build-${jid}.out"; build_err="${logs_dir}/build-${jid}.err"
test_out="${logs_dir}/test-${jid}.out";   test_err="${logs_dir}/test-${jid}.err"

# Cap parallelism: this node reports nproc=384, and -j384 OOM/thrashes RCCL's
# heavy C++ compile. Default to <=32; export so build-ompi.sh inherits it too.
if [[ -z "${RCCL_BUILD_JOBS:-}" ]]; then
  nproc_count="$(nproc)"
  RCCL_BUILD_JOBS=$(( nproc_count < 32 ? nproc_count : 32 ))
fi
export RCCL_BUILD_JOBS
build_jobs="${RCCL_BUILD_JOBS}"

# --- build phase -> build-%j.{out,err}. Runs in a subshell so a hard `exit` or
# a failing final command is caught below and skips the test phase. `set -e` is
# ignored inside the subshell (it is the left operand of `||`), so each step
# relies on the explicit check that follows it. ---
build_rc=0
(
# 1) ROCm: provisioned on the head node from TheRock CI artifacts; consume the
# workspace tree via rocm.env.
echo "==> [1/6] use ROCm (provisioned from TheRock artifacts on head node)"
rocm_env="${workdir}/.ci-out/rocm.env"
if [[ ! -f "${rocm_env}" ]]; then
  echo "ERROR: ${rocm_env} not found. The workflow must run the 'Provision ROCm" >&2
  echo "       from TheRock artifacts' step on the head node before sbatch." >&2
  exit 1
fi
set -a
# shellcheck source=/dev/null  # runtime fragment written by the provision step
source "${rocm_env}"
set +a
if [[ ! -x "${ROCM_PATH:-}/bin/hipcc" ]]; then
  echo "ERROR: ROCM_PATH='${ROCM_PATH:-}' (from rocm.env) has no bin/hipcc" >&2
  exit 1
fi
echo "    ROCM_PATH=${ROCM_PATH}"
echo "    ROCM_RELEASE=${ROCM_RELEASE:-<unset>}"
echo "    hipcc: $("${ROCM_PATH}/bin/hipcc" --version 2>/dev/null | head -n1)"

# 2) OpenMPI: built here against ROCM_PATH into the per-job workspace cache
# (no shared cache), then ompi.env (MPI_HOME) is sourced below.
echo "==> [2/6] build OpenMPI"
"${ci_dir}/lib/build-ompi.sh"
set -a
# shellcheck source=/dev/null  # runtime fragment written by build-ompi.sh
source "${workdir}/.ci-out/ompi.env"
set +a

# Toolchain env for the in-tree RCCL / rccl-tests builds.
export HIP_PATH="${ROCM_PATH}"
export PATH="${ROCM_PATH}/bin:${MPI_HOME}/bin:${PATH}"
export LD_LIBRARY_PATH="${ROCM_PATH}/lib:${MPI_HOME}/lib:${LD_LIBRARY_PATH:-}"
export CMAKE_PREFIX_PATH="${ROCM_PATH}/lib/cmake:${ROCM_PATH}${CMAKE_PREFIX_PATH:+:${CMAKE_PREFIX_PATH}}"

# Confirm the hipcc on PATH is the provisioned one (nothing shadows it).
resolved_hipcc="$(command -v hipcc || true)"
if [[ "${resolved_hipcc}" != "${ROCM_PATH}/bin/hipcc" ]]; then
  echo "ERROR: hipcc on PATH is '${resolved_hipcc}', expected '${ROCM_PATH}/bin/hipcc'" >&2
  exit 1
fi
echo "==> verified hipcc = ${resolved_hipcc}"

# Ensure `import yaml` works for the build; must be sourced so the venv persists.
# shellcheck source=/dev/null
source "${ci_dir}/lib/ensure-python-yaml.sh"

# 3) RCCL with GIN / rocSHMEM. Leave ROCSHMEM_INSTALL_DIR unset so install.sh
# builds the sibling projects/rocshmem checkout (gda + USE_SDMA) and installs
# it at projects/rccl/ext/rocshmem. A pre-set install dir skips that build.
# --prefix is required: without an install tree, lib/cmake/rccl/*config.cmake is
# never written and the rccl_cfg check below fails the job before any test.
echo "==> [3/6] build RCCL (GIN, targets=${amdgpu_targets})"
unset ROCSHMEM_INSTALL_DIR
rccl_prefix="${workdir}/.ci-out/rccl-install"
rm -rf "${rccl_prefix}"
(
  cd "${workdir}/projects/rccl" \
  && MPI_PATH="${MPI_HOME}" ./install.sh -lj "${build_jobs}" \
       --amdgpu_targets="${amdgpu_targets}" \
       --prefix="${rccl_prefix}" \
       --rocshmem-gin --no_clean
)
rocshmem_prefix="${workdir}/projects/rccl/ext/rocshmem"
if [[ ! -f "${rocshmem_prefix}/lib/librocshmem.a" ]]; then
  echo "ERROR: --rocshmem-gin did not install librocshmem.a under ${rocshmem_prefix}/lib" >&2
  ls -la "${rocshmem_prefix}/lib" 2>/dev/null >&2 || true
  exit 1
fi
ROCSHMEM_INSTALL_DIR="${rocshmem_prefix}"
export ROCSHMEM_INSTALL_DIR
{
  printf 'export ROCSHMEM_INSTALL_DIR=%q\n' "${ROCSHMEM_INSTALL_DIR}"
} > "${workdir}/.ci-out/rocshmem.env"
echo "    rocSHMEM install: ${ROCSHMEM_INSTALL_DIR}"
if [[ ! -f "${rccl_prefix}/lib/librccl.so" && ! -f "${rccl_prefix}/lib/librccl.so.1" ]]; then
  echo "ERROR: no librccl.so under ${rccl_prefix}/lib" >&2
  ls -la "${rccl_prefix}/lib" 2>/dev/null >&2 || true
  exit 1
fi
rccl_cfg="$(ls "${rccl_prefix}"/lib*/cmake/rccl/*[Cc]onfig.cmake 2>/dev/null | head -n1 || true)"
if [[ -z "${rccl_cfg}" ]]; then
  echo "ERROR: no RCCL cmake config under ${rccl_prefix}/lib*/cmake/rccl" >&2
  ls -laR "${rccl_prefix}"/lib*/cmake 2>/dev/null >&2 || true
  exit 1
fi
rccl_cmake_dir="$(dirname "${rccl_cfg}")"
echo "    in-tree RCCL cmake config: ${rccl_cmake_dir}"

# 4) rccl-tests via install.sh with ENABLE_DEVICE_API only. This is the
# device-API-without-rocSHMEM build. It runs AllReduce/AllGather/ReduceScatter
# at -R 2 and LSA all_reduce_perf (-D 1/-D 2). install.sh always writes into
# projects/rccl-tests/build.
echo "==> [4/6] build rccl-tests (ENABLE_DEVICE_API via install.sh)"
rccl_tests_src="${workdir}/projects/rccl-tests"
rccl_tests_nongin_build="${rccl_tests_src}/build"
(
  cd "${rccl_tests_src}" \
  && ./install.sh --mpi --enable-device-api \
       --mpi_home="${MPI_HOME}" --rccl_home="${rccl_prefix}" \
       --rocm_home="${ROCM_PATH}" \
       --gpu_targets="${amdgpu_targets}"
)
# Symmetric + LSA suites in device-api-nongin-tests.json.
required_nongin_bins=(
  all_reduce_perf all_gather_perf reduce_scatter_perf
)
for perf_bin in "${required_nongin_bins[@]}"; do
  if [[ ! -x "${rccl_tests_nongin_build}/${perf_bin}" ]]; then
    echo "ERROR: ${perf_bin} not built under ${rccl_tests_nongin_build}" >&2
    ls -la "${rccl_tests_nongin_build}" 2>/dev/null >&2 || true
    exit 1
  fi
done

# 5) rccl-tests via cmake so the GIN matrix binary RDC-links rocshmem
# (install.sh/Makefile path does not enable ENABLE_ROCSHMEM_GIN). Needed because
# this job's librccl.so is --rocshmem-gin. RCCL_TESTS_BIN_DIR is consumed by
# run-device-api-ci.sh for the GIN matrix. ROCSHMEM_INSTALL_DIR is the
# install.sh rocSHMEM tree (projects/rccl/ext/rocshmem).
echo "==> [5/6] build rccl-tests (ENABLE_DEVICE_API + ENABLE_ROCSHMEM_GIN)"
rccl_tests_build="${workdir}/.ci-out/rccl-tests-build"
rm -rf "${rccl_tests_build}"; mkdir -p "${rccl_tests_build}"
(
  cd "${rccl_tests_build}"
  cmake -DCMAKE_BUILD_TYPE=Release -DUSE_MPI=ON \
    -DENABLE_DEVICE_API=ON -DENABLE_ROCSHMEM_GIN=ON \
    -DROCSHMEM_INSTALL_DIR="${ROCSHMEM_INSTALL_DIR}" \
    -DRCCL_DIR="${rccl_cmake_dir}" \
    -DCMAKE_PREFIX_PATH="${rccl_prefix};${MPI_HOME}" \
    -DGPU_TARGETS="${amdgpu_targets}" \
    "${rccl_tests_src}"
  # -k: all_reduce_perf still fails under ENABLE_ROCSHMEM_GIN (see gin.sbatch).
  # This configure also builds gin_rocshmem_gda_factory_test and the test/
  # host unit target. required_perf_bins below does not check those, so a
  # green job only proves the RDC-linked matrix binaries exist.
  make -j"${build_jobs}" -k || true
)
# Keep in sync with foreach(_gin_rdc_bin) in
# projects/rccl-tests/src/CMakeLists.txt. Because make runs with -k, this
# -x check is what fails the job when one of those binaries did not build.
# This job launches only alltoall_perf (gin-d3 / gin-d4, NCCL_GIN_TYPE=2).
# all_gather_perf and broadcast_perf are RDC-linked for gin-tests.json
# (Anvil SDMA, NCCL_GIN_TYPE=7).
required_perf_bins=(
  all_gather_perf alltoall_perf broadcast_perf
)
for perf_bin in "${required_perf_bins[@]}"; do
  if [[ ! -x "${rccl_tests_build}/${perf_bin}" ]]; then
    echo "ERROR: ${perf_bin} not built under ${rccl_tests_build}" >&2
    ls -la "${rccl_tests_build}" 2>/dev/null >&2 || true
    exit 1
  fi
done
# Persist for a standalone run-device-api-ci.sh (the in-process export above
# does not survive a new shell). Mirrors gin.sbatch's .ci-out/rccl.env.
{
  printf 'export RCCL_INSTALL_PREFIX=%q\n' "${rccl_prefix}"
  printf 'export RCCL_TESTS_BIN_DIR=%q\n' "${rccl_tests_build}"
  printf 'export RCCL_TESTS_NONGIN_BIN_DIR=%q\n' "${rccl_tests_nongin_build}"
  printf 'export RCCL_TESTS_DIR=%q\n' "${rccl_tests_src}"
} > "${workdir}/.ci-out/rccl.env"
) > "${build_out}" 2> "${build_err}" || build_rc=$?
echo "==> build phase logs: ${build_out} / ${build_err}"

if [[ "${build_rc}" -ne 0 ]]; then
  echo "ERROR: build phase failed (rc=${build_rc}); skipping device-api tests." >&2
  echo "----- last 60 lines of ${build_err} -----" >&2
  tail -n 60 "${build_err}" >&2 || true
  exit "${build_rc}"
fi

# --- test phase -> test-%j.{out,err} ---
# Source build fragments so ROCM_PATH / MPI_HOME / ROCSHMEM / RCCL paths are set
# even though the build ran in a subshell.
set -a
# shellcheck source=/dev/null
source "${workdir}/.ci-out/rocm.env"
# shellcheck source=/dev/null
source "${workdir}/.ci-out/ompi.env"
# shellcheck source=/dev/null
source "${workdir}/.ci-out/rocshmem.env"
# shellcheck source=/dev/null
source "${workdir}/.ci-out/rccl.env"
set +a

nongin_rc=0
gin_rc=0
{
# 6) Benchmarks: non-GIN matrix (AR/AG/RS -R 2 + LSA -D 1/-D 2) against the
# install.sh build, then the GIN matrix (host-proxy -D 3/-D 4 alltoall)
# against the cmake ENABLE_ROCSHMEM_GIN build. Skip the GIN
# AllToAllDevtime smoke on the non-GIN pass; the GIN pass runs it.
# Capture each matrix with its own || rc=$?: set -e is ignored on the left of ||
# inside a compound command, so a single trailing || on the brace group would
# keep only the second matrix's status.
echo "==> [6/6] run benchmarks"
ulimit -l unlimited 2>/dev/null || echo "WARN: could not raise RLIMIT_MEMLOCK (ulimit -l)"
echo "==> RLIMIT_MEMLOCK (ulimit -l) = $(ulimit -l)"

echo "==> non-GIN matrix (install.sh ENABLE_DEVICE_API build)"
CONFIG="${ci_dir}/lib/device-api-nongin-tests.json" \
RCCL_TESTS_BIN_DIR="${RCCL_TESTS_NONGIN_BIN_DIR}" \
RCCL_CI_SKIP_DEVTIME=1 \
  "${ci_dir}/run-device-api-ci.sh" || nongin_rc=$?

echo "==> GIN matrix (cmake ENABLE_ROCSHMEM_GIN build)"
CONFIG="${ci_dir}/lib/device-api-tests.json" \
RCCL_TESTS_BIN_DIR="${RCCL_TESTS_BIN_DIR}" \
  "${ci_dir}/run-device-api-ci.sh" || gin_rc=$?
} > "${test_out}" 2> "${test_err}"
echo "==> test phase logs: ${test_out} / ${test_err}"
if [[ "${nongin_rc}" -ne 0 || "${gin_rc}" -ne 0 ]]; then
  echo "ERROR: device-api tests failed (nongin_rc=${nongin_rc} gin_rc=${gin_rc})" >&2
  echo "----- last 60 lines of ${test_err} -----" >&2
  tail -n 60 "${test_err}" >&2 || true
  # run-device-api-ci.sh writes "=== FAILED RUNS" to stdout. The non-GIN matrix
  # is followed by the GIN matrix, so that summary can sit above the last 60 lines.
  echo "----- FAILED RUNS in ${test_out} -----" >&2
  grep -E -A 30 '^=== FAILED RUNS' "${test_out}" >&2 || true
  echo "----- last 60 lines of ${test_out} -----" >&2
  tail -n 60 "${test_out}" >&2 || true
  if [[ "${nongin_rc}" -ne 0 ]]; then
    exit "${nongin_rc}"
  fi
  exit "${gin_rc}"
fi
