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
#
# Compute-node orchestrator for the RCCL device-API CI job, invoked from
# .github/workflows/rccl-suite.yml (suite=device-api) via submit_slurm_job.py.
# On one 8-GPU node:
#   1. consume ROCm (provisioned on the head node) via .ci-out/rocm.env
#   2. lib/build-ompi.sh     build OpenMPI (--with-rocm) -> ompi.env
#   3. projects/rccl/install.sh   build RCCL (in-tree, release)
#   4. rccl-tests install.sh      build rccl-tests (ENABLE_DEVICE_API=1)
#   5. run-device-api-ci.sh       run the device-API benchmark suite
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
# 1) ROCm: provisioned on the head node; consume its shared tree via rocm.env.
echo "==> [1/5] use ROCm (provisioned on head node)"
rocm_env="${workdir}/.ci-out/rocm.env"
if [[ ! -f "${rocm_env}" ]]; then
  echo "ERROR: ${rocm_env} not found. The workflow must run the 'Provision ROCm'" >&2
  echo "       step (lib/fetch-rocm.sh on the head node) before sbatch." >&2
  exit 1
fi
set -a
# shellcheck source=/dev/null  # runtime fragment written by fetch-rocm.sh
source "${rocm_env}"
set +a
if [[ ! -x "${ROCM_PATH:-}/bin/hipcc" ]]; then
  echo "ERROR: ROCM_PATH='${ROCM_PATH:-}' (from rocm.env) has no bin/hipcc" >&2
  exit 1
fi
echo "    ROCM_PATH=${ROCM_PATH}"
echo "    ROCM_RELEASE=${ROCM_RELEASE:-<unset>}"

# Prove the build uses the ROCm we downloaded, not a system /opt/rocm. Fail loudly
# rather than silently compiling against the wrong toolchain.
rocm_cache="${RCCL_DEVICE_API_CACHE:-/apps/rccl-ci}/rocm"
case "${ROCM_PATH}" in
  "${rocm_cache}"/*) : ;;
  *)
    if [[ -z "${ROCM_PATH_OVERRIDE:-}" ]]; then
      echo "ERROR: ROCM_PATH=${ROCM_PATH} is not under the provisioned cache" >&2
      echo "       ${rocm_cache} (stale rocm.env or wrong ROCm?)." >&2
      exit 1
    fi
    echo "    (ROCM_PATH_OVERRIDE in effect; skipping cache-path check)" ;;
esac
if [[ -n "${ROCM_RELEASE:-}" ]] \
   && ! grep -qx "release=${ROCM_RELEASE}" "${ROCM_PATH}/.stamp" 2>/dev/null; then
  echo "ERROR: ${ROCM_PATH}/.stamp does not record release=${ROCM_RELEASE}" >&2
  cat "${ROCM_PATH}/.stamp" 2>/dev/null >&2 || echo "       (.stamp missing)" >&2
  exit 1
fi
echo "    hipcc: $("${ROCM_PATH}/bin/hipcc" --version 2>/dev/null | head -n1)"

# 2) OpenMPI: prebuilt by the OpenMPI provision job; this is a cache hit that
# just writes ompi.env (MPI_HOME). Falls back to building if run standalone.
echo "==> [2/5] use OpenMPI"
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

# 3) RCCL via its own install.sh (-l = local gpu arch only).
echo "==> [3/5] build RCCL (targets=${amdgpu_targets})"
( cd "${workdir}/projects/rccl" \
  && MPI_PATH="${MPI_HOME}" ./install.sh -lj "${build_jobs}" \
       --amdgpu_targets="${amdgpu_targets}" )
rccl_prefix="${workdir}/projects/rccl/build/release"

# 4) rccl-tests: built in-tree from projects/rccl-tests (must match
# RCCL_TESTS_DIR in run-device-api-ci.sh). install.sh wipes its own build/.
echo "==> [4/5] build rccl-tests (ENABLE_DEVICE_API=1)"
rccl_tests_dir="${workdir}/projects/rccl-tests"
( cd "${rccl_tests_dir}" \
  && ENABLE_DEVICE_API=1 ./install.sh \
       --mpi --mpi_home="${MPI_HOME}" --rccl_home="${rccl_prefix}" \
       --gpu_targets="${amdgpu_targets}" )
} > "${build_out}" 2> "${build_err}"
echo "==> build phase logs: ${build_out} / ${build_err}"

# --- test phase -> test-%j.{out,err} ---
{
# 5) Benchmarks.
echo "==> [5/5] run benchmarks"
# GIN/RoCE QP creation pins memory; a low memlock limit makes ibv_create_qp fail with ENOMEM. Raise it for the mpirun ranks and log it.
ulimit -l unlimited 2>/dev/null || echo "WARN: could not raise RLIMIT_MEMLOCK (ulimit -l)"
echo "==> RLIMIT_MEMLOCK (ulimit -l) = $(ulimit -l)"
"${ci_dir}/run-device-api-ci.sh"
} > "${test_out}" 2> "${test_err}"
echo "==> test phase logs: ${test_out} / ${test_err}"
