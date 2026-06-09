#!/usr/bin/env bash
#SBATCH --job-name=rccl-device-api
#SBATCH --nodes=1
#SBATCH --ntasks=8
#SBATCH --partition=interactive
#SBATCH --exclusive
#SBATCH --time=02:00:00
#SBATCH --output=%x-%j.out
#SBATCH --error=%x-%j.err
#
# Compute-node entrypoint + orchestrator for the RCCL device-API CI job.
# Invoked from .github/workflows/rccl-device-api.yml via `sbatch --wait`. Runs
# the whole build + benchmark flow on a single 8-GPU node:
#
#   1. lib/fetch-rocm.sh     fetch ROCm (TheRock nightly) -> writes rocm.env
#   2. lib/build-ompi.sh     build OpenMPI (--with-rocm)   -> writes ompi.env
#   3. projects/rccl/install.sh           build RCCL (in-tree, release)
#   4. rccl-tests install.sh              build rccl-tests (ENABLE_DEVICE_API=1)
#   5. run-device-api-ci.sh  run the device-API benchmark suite
#
# RCCL and rccl-tests are built straight from their own install.sh scripts (no
# wrapper); only the cached ROCm + OpenMPI prerequisites have dedicated scripts.
#
# Required env (exported by the workflow via `sbatch --export=ALL,...`):
#   RCCL_DEVICE_API_WORKDIR   Path to the GHA workspace (the rocm-systems root).
#
# Optional env:
#   ROCM_RELEASE              ROCm nightly/dev version fetch-rocm.sh fetches
#                             (empty => latest nightly for the family).
#   ROCM_AMDGPU_FAMILY        TheRock artifact group (default: gfx950-dcgpu).
#   RCCL_AMDGPU_TARGETS       RCCL --amdgpu_targets (default: gfx950).
#   RCCL_DEVICE_API_CACHE     Persistent cache root (default: /apps/rccl-ci;
#                             must be on storage shared with the compute node).
#   RCCL_TESTS_REPO/REF       rccl-tests source (default: ROCm/rccl-tests @ develop).

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
build_jobs="${RCCL_BUILD_JOBS:-$(nproc)}"
rccl_tests_repo="${RCCL_TESTS_REPO:-https://github.com/ROCm/rccl-tests.git}"
rccl_tests_ref="${RCCL_TESTS_REF:-develop}"

# 1) ROCm (cached, version-keyed) -> .ci-out/rocm.env (ROCM_PATH, ROCM_RELEASE)
echo "==> [1/5] fetch ROCm"
"${ci_dir}/lib/fetch-rocm.sh"
set -a
# shellcheck source=/dev/null  # runtime fragment written by fetch-rocm.sh
source "${workdir}/.ci-out/rocm.env"
set +a

# 2) OpenMPI (cached, --with-rocm) -> .ci-out/ompi.env (MPI_HOME)
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

# 3) RCCL: built directly from its own install.sh (-l = local gpu arch only).
echo "==> [3/5] build RCCL (targets=${amdgpu_targets})"
( cd "${workdir}/projects/rccl" \
  && MPI_PATH="${MPI_HOME}" ./install.sh -lj "${build_jobs}" \
       --amdgpu_targets="${amdgpu_targets}" )
rccl_prefix="${workdir}/projects/rccl/build/release"

# 4) rccl-tests: cloned + built fresh each run (so RCCL breakage is caught here)
#    straight from its own install.sh with the device API enabled.
echo "==> [4/5] build rccl-tests (ENABLE_DEVICE_API=1)"
rccl_tests_dir="${workdir}/rccl-tests"
rm -rf "${rccl_tests_dir}"
git clone --depth 1 --branch "${rccl_tests_ref}" "${rccl_tests_repo}" "${rccl_tests_dir}"
( cd "${rccl_tests_dir}" \
  && ENABLE_DEVICE_API=1 ./install.sh \
       --mpi --mpi_home="${MPI_HOME}" --rccl_home="${rccl_prefix}" )

# 5) Benchmarks.
echo "==> [5/5] run benchmarks"
"${ci_dir}/run-device-api-ci.sh"
