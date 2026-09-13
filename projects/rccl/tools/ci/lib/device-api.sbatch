#!/usr/bin/env bash
#SBATCH --job-name=rccl-device-api
#SBATCH --nodes=1
#SBATCH --ntasks=8
#SBATCH --partition=interactive
#SBATCH --exclusive
#SBATCH --time=02:00:00
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
#   3. lib/build-rocshmem.sh   build rocSHMEM -> rocshmem.env
#   4. projects/rccl/install.sh   build RCCL (in-tree, release, --rocshmem-gin)
#   5. rccl-tests (cmake)         build rccl-tests (ENABLE_DEVICE_API + ENABLE_ROCSHMEM_GIN)
#   6. run-device-api-ci.sh       run the device-API benchmark suite
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

# --- build phase -> build-%j.{out,err} ---
{
# 1) ROCm: provisioned on the head node from TheRock CI artifacts; consume the
# workspace tree via rocm.env.
echo "==> [1/5] use ROCm (provisioned from TheRock artifacts on head node)"
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
echo "==> [2/5] build OpenMPI"
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

# 3) rocSHMEM (in-tree, IPC+SDMA) -> rocshmem.env (ROCSHMEM_INSTALL_DIR)
echo "==> [3/6] build rocSHMEM"
GPU_TARGETS="${amdgpu_targets}" "${ci_dir}/lib/build-rocshmem.sh"
set -a
# shellcheck source=/dev/null
source "${workdir}/.ci-out/rocshmem.env"
set +a

# 4) RCCL with GIN / rocSHMEM (device-API + GIN-SDMA -D 3 paths need rocshmem in the executable).
echo "==> [4/6] build RCCL (GIN, targets=${amdgpu_targets})"
(
  cd "${workdir}/projects/rccl" \
  && MPI_PATH="${MPI_HOME}" ./install.sh -lj "${build_jobs}" \
       --amdgpu_targets="${amdgpu_targets}" \
       --rocshmem-gin --no_clean \
       --cmake-options "-DROCSHMEM_INSTALL_DIR=${ROCSHMEM_INSTALL_DIR}"
)
rccl_prefix="${workdir}/projects/rccl/build/release"
rccl_cfg="$(ls "${rccl_prefix}"/lib*/cmake/rccl/*[Cc]onfig.cmake 2>/dev/null | head -n1)"
if [[ -z "${rccl_cfg}" ]]; then
  echo "ERROR: no RCCL cmake config under ${rccl_prefix}/lib*/cmake/rccl" >&2
  ls -laR "${rccl_prefix}"/lib*/cmake 2>/dev/null >&2 || true
  exit 1
fi
rccl_cmake_dir="$(dirname "${rccl_cfg}")"
echo "    in-tree RCCL cmake config: ${rccl_cmake_dir}"

# 5) rccl-tests via cmake so all_gather_perf -D 3 links rocshmem (install.sh/Makefile
# path does not enable ENABLE_ROCSHMEM_GIN). RCCL_TESTS_BIN_DIR is consumed by
# run-device-api-ci.sh.
echo "==> [5/6] build rccl-tests (ENABLE_DEVICE_API + ENABLE_ROCSHMEM_GIN)"
rccl_tests_src="${workdir}/projects/rccl-tests"
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
  make -j"${build_jobs}" -k all_gather_perf alltoall_perf all_reduce_perf || true
)
if [[ ! -x "${rccl_tests_build}/all_reduce_perf" ]]; then
  echo "ERROR: all_reduce_perf not built under ${rccl_tests_build}" >&2
  ls -la "${rccl_tests_build}" 2>/dev/null >&2 || true
  exit 1
fi
export RCCL_TESTS_BIN_DIR="${rccl_tests_build}"
} > "${build_out}" 2> "${build_err}"
echo "==> build phase logs: ${build_out} / ${build_err}"

# --- test phase -> test-%j.{out,err} ---
{
# 6) Benchmarks.
echo "==> [6/6] run benchmarks"
# GIN/RoCE QP creation pins memory; a low memlock limit makes ibv_create_qp fail with ENOMEM. Raise it for the mpirun ranks and log it.
ulimit -l unlimited 2>/dev/null || echo "WARN: could not raise RLIMIT_MEMLOCK (ulimit -l)"
echo "==> RLIMIT_MEMLOCK (ulimit -l) = $(ulimit -l)"
"${ci_dir}/run-device-api-ci.sh"
} > "${test_out}" 2> "${test_err}"
echo "==> test phase logs: ${test_out} / ${test_err}"
