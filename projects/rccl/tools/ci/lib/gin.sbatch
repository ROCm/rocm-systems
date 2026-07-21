#!/usr/bin/env bash
#SBATCH --job-name=rccl-gin
#SBATCH --nodes=1
#SBATCH --ntasks=8
#SBATCH --partition=interactive
#SBATCH --exclusive
#SBATCH --time=04:00:00
#SBATCH --output=%x-%j.out
#SBATCH --error=%x-%j.err
#
# Compute-node orchestrator for the RCCL/rocSHMEM GIN CI job (baremetal
# translation of docker_build_test.bash + dockerfile-gin-fast-path), invoked
# from .github/workflows/rccl-suite.yml (suite=gin) via submit_slurm_job.py.
# On one 8-GPU node:
#   1. consume ROCm (provisioned on the head node) via .ci-out/rocm.env
#   2. lib/build-ompi.sh      build OpenMPI (--with-rocm) -> ompi.env
#   3. lib/build-rocshmem.sh  build rocSHMEM (IPC+SDMA)   -> rocshmem.env
#   4. projects/rccl  install.sh  build RCCL (GIN/rocSHMEM) -> rccl.env
#   5. projects/rccl-tests        build rccl-tests (DEVICE_API + ROCSHMEM)
#   6. run-gin-ci.sh              run the GIN/rocSHMEM tests (non-gating)
#
# Required env (from the workflow via `sbatch --export=ALL,...`):
#   RCCL_DEVICE_API_WORKDIR   GHA workspace (rocm-systems root)
# Optional env:
#   RCCL_AMDGPU_TARGETS       GPU arch (default: gfx950)
#   RCCL_DEVICE_API_CACHE     Cache root, shared with head node (default: /apps/rccl-ci)

set -euo pipefail

echo "=== sbatch gin on $(hostname) ==="
date -Is
env | grep -E '^SLURM_' | sort || true
echo

workdir="${RCCL_DEVICE_API_WORKDIR:?RCCL_DEVICE_API_WORKDIR must be exported from the workflow}"
cd "${workdir}"

ci_dir="${workdir}/projects/rccl/tools/ci"
gpu_targets="${RCCL_AMDGPU_TARGETS:-gfx950}"

# Phase-split logs into build-%j and test-%j (SLURM_SUBMIT_DIR = the *-logs dir).
logs_dir="${SLURM_SUBMIT_DIR:-$(pwd)}"
jid="${SLURM_JOB_ID:-local}"
build_out="${logs_dir}/build-${jid}.out"; build_err="${logs_dir}/build-${jid}.err"
test_out="${logs_dir}/test-${jid}.out";   test_err="${logs_dir}/test-${jid}.err"

# Cap parallelism (nodes report nproc=384; -j384 thrashes the C++ compile).
if [[ -z "${RCCL_BUILD_JOBS:-}" ]]; then
  nproc_count="$(nproc)"
  RCCL_BUILD_JOBS=$(( nproc_count < 32 ? nproc_count : 32 ))
fi
export RCCL_BUILD_JOBS
build_jobs="${RCCL_BUILD_JOBS}"

# --- build phase -> build-%j.{out,err}. Runs in a subshell so any failure (a
# hard `exit` or a set -e abort) is caught below and skips the test phase. ---
build_rc=0
(
# 1) ROCm: provisioned on the head node; consume its shared tree via rocm.env.
echo "==> [1/6] use ROCm (provisioned on head node)"
rocm_env="${workdir}/.ci-out/rocm.env"
if [[ ! -f "${rocm_env}" ]]; then
  echo "ERROR: ${rocm_env} not found. The workflow must run 'Provision ROCm'" >&2
  echo "       (lib/fetch-rocm.sh on the head node) before sbatch." >&2
  exit 1
fi
set -a
# shellcheck source=/dev/null
source "${rocm_env}"
set +a
if [[ ! -x "${ROCM_PATH:-}/bin/hipcc" ]]; then
  echo "ERROR: ROCM_PATH='${ROCM_PATH:-}' (from rocm.env) has no bin/hipcc" >&2
  exit 1
fi
echo "    ROCM_PATH=${ROCM_PATH} ROCM_RELEASE=${ROCM_RELEASE:-<unset>}"

rocm_cache="${RCCL_DEVICE_API_CACHE:-/apps/rccl-ci}/rocm"
case "${ROCM_PATH}" in
  "${rocm_cache}"/*) : ;;
  *)
    if [[ -z "${ROCM_PATH_OVERRIDE:-}" ]]; then
      echo "ERROR: ROCM_PATH=${ROCM_PATH} is not under ${rocm_cache}" >&2
      exit 1
    fi ;;
esac
echo "    hipcc: $("${ROCM_PATH}/bin/hipcc" --version 2>/dev/null | head -n1)"

export HIP_PATH="${ROCM_PATH}"
export PATH="${ROCM_PATH}/bin:${ROCM_PATH}/llvm/bin:${PATH}"
export LD_LIBRARY_PATH="${ROCM_PATH}/lib:${LD_LIBRARY_PATH:-}"
export CMAKE_PREFIX_PATH="${ROCM_PATH}/lib/cmake:${ROCM_PATH}${CMAKE_PREFIX_PATH:+:${CMAKE_PREFIX_PATH}}"

# 2) OpenMPI: prebuilt by the OpenMPI provision job; this is a cache hit that
# just writes ompi.env (MPI_HOME). Falls back to building if run standalone.
echo "==> [2/6] use OpenMPI"
"${ci_dir}/lib/build-ompi.sh"
set -a
# shellcheck source=/dev/null
source "${workdir}/.ci-out/ompi.env"
set +a
export PATH="${MPI_HOME}/bin:${PATH}"

# Ensure `import yaml` works for the build; must be sourced so the venv persists.
# shellcheck source=/dev/null
source "${ci_dir}/lib/ensure-python-yaml.sh"

# 3) rocSHMEM (in-tree, IPC+SDMA) -> rocshmem.env (ROCSHMEM_INSTALL_DIR)
echo "==> [3/6] build rocSHMEM"
GPU_TARGETS="${gpu_targets}" "${ci_dir}/lib/build-rocshmem.sh"
set -a
# shellcheck source=/dev/null
source "${workdir}/.ci-out/rocshmem.env"
set +a

# 4) RCCL with GIN / rocSHMEM. Mirrors the Dockerfile: patch the rocSHMEM
# threshold guard, then install.sh with ENABLE_ROCSHMEM_GIN.
echo "==> [4/6] build RCCL (GIN, targets=${gpu_targets})"
rccl_src="${workdir}/projects/rccl"
rccl_prefix="${workdir}/.ci-out/rccl-install"
rm -rf "${rccl_prefix}"
wrap="${rccl_src}/src/rccl_wrap.cc"
if grep -q 'comm->enableRocshmem && comm->nNodes > 1' "${wrap}" 2>/dev/null; then
  sed -i \
    's/if (comm->enableRocshmem && comm->nNodes > 1 && (comm->nRanks\/comm->nNodes == 8) && comm->rocshmemThreshold <= 1048576)/if (comm->enableRocshmem)/' \
    "${wrap}"
  echo "    patched rccl_wrap.cc rocSHMEM guard"
else
  echo "    (rccl_wrap.cc guard pattern not found; assuming already patched)"
fi
(
  cd "${rccl_src}"
  MPI_PATH="${MPI_HOME}" ./install.sh -j "${build_jobs}" \
    --amdgpu_targets="${gpu_targets}" \
    --prefix="${rccl_prefix}" \
    --no-device-linker --no_clean \
    --cmake-options "-DENABLE_ROCSHMEM_GIN=ON -DROCSHMEM_INSTALL_DIR=${ROCSHMEM_INSTALL_DIR}"
)
# Workaround: hipify renames core.h->core_tmp.h but gin headers still include ../core.h
rccl_build="${rccl_src}/build/release"
ln -sf core_tmp.h "${rccl_build}/include/nccl_device/core.h"
ln -sf ../hipify/src/include/nccl_tuner.h "${rccl_build}/include/nccl_tuner.h"
cp -r "${rccl_build}/include/." "${rccl_prefix}/include/"
if [[ ! -f "${rccl_prefix}/lib/librccl.so" && ! -f "${rccl_prefix}/lib/librccl.so.1" ]]; then
  echo "ERROR: no librccl.so under ${rccl_prefix}/lib" >&2
  ls -la "${rccl_prefix}/lib" 2>/dev/null >&2 || true
  exit 1
fi

# 5) rccl-tests (DEVICE_API + ROCSHMEM). allreduce fails to build; -k keeps going
# so alltoall_perf still lands (matches the Dockerfile's `make -k || true`).
echo "==> [5/6] build rccl-tests (ENABLE_DEVICE_API + ENABLE_ROCSHMEM)"
rccl_tests_src="${workdir}/projects/rccl-tests"
rccl_tests_build="${workdir}/.ci-out/rccl-tests-build"

# --whole-archive rocSHMEM into alltoall_perf so -rdynamic exports its host symbols
# (e.g. rocshmem::envvar::log_flags) for librccl.so, which is built ENABLE_ROCSHMEM_GIN
# and leaves them undefined. CI-time patch to avoid editing tracked rccl-tests files.
tlib="${rccl_tests_src}/src/CMakeLists.txt"
if grep -q 'target_link_libraries(alltoall_perf PRIVATE ${ROCSHMEM_LIBRARY}' "${tlib}" 2>/dev/null; then
  sed -i \
    's|target_link_libraries(alltoall_perf PRIVATE ${ROCSHMEM_LIBRARY}|target_link_libraries(alltoall_perf PRIVATE -Wl,--whole-archive ${ROCSHMEM_LIBRARY} -Wl,--no-whole-archive|' \
    "${tlib}"
  echo "    patched alltoall_perf link: --whole-archive rocSHMEM"
else
  echo "    (alltoall_perf whole-archive pattern not found; skipping)"
fi

rm -rf "${rccl_tests_build}"; mkdir -p "${rccl_tests_build}"
(
  cd "${rccl_tests_build}"
  cmake -DCMAKE_BUILD_TYPE=Release -DUSE_MPI=ON \
    -DENABLE_DEVICE_API=ON -DENABLE_ROCSHMEM=ON \
    -DROCSHMEM_INSTALL_DIR="${ROCSHMEM_INSTALL_DIR}" \
    -DCMAKE_PREFIX_PATH="${rccl_prefix};${MPI_HOME}" \
    -DGPU_TARGETS="${gpu_targets}" \
    "${rccl_tests_src}"
  make -j"${build_jobs}" -k || true
)
if [[ ! -x "${rccl_tests_build}/alltoall_perf" ]]; then
  echo "ERROR: alltoall_perf not built under ${rccl_tests_build}" >&2
  ls -la "${rccl_tests_build}" 2>/dev/null >&2 || true
  exit 1
fi

# Diag: confirm alltoall_perf exports the rocSHMEM host symbol librccl.so resolves
# at runtime (log_flags); a miss here => runtime 'undefined symbol'.
sym='rocshmem.*envvar.*log_flags'
rocshmem_a="${ROCSHMEM_INSTALL_DIR}/lib/librocshmem.a"
librccl="$(ls "${rccl_prefix}"/lib/librccl.so* 2>/dev/null | head -n1)"
echo "== log_flags symbol diag =="
nm -C "${rocshmem_a}" 2>/dev/null | grep -E " [TDBRtdbr] .*${sym}" | head || echo "  not defined in librocshmem.a"
nm -CD "${rccl_tests_build}/alltoall_perf" 2>/dev/null | grep -E "${sym}" || echo "  NOT exported by alltoall_perf"
nm -CD "${librccl}" 2>/dev/null | grep -E " U .*${sym}" || echo "  not undefined in librccl.so"

# Hand the install/build locations to the run stage.
{
  printf 'export RCCL_INSTALL_PREFIX=%q\n' "${rccl_prefix}"
  printf 'export RCCL_TESTS_BIN_DIR=%q\n' "${rccl_tests_build}"
} > "${workdir}/.ci-out/rccl.env"
) > "${build_out}" 2> "${build_err}" || build_rc=$?
echo "==> build phase logs: ${build_out} / ${build_err}"

# Skip the test phase if the build failed (testing a broken build is pointless).
if [[ "${build_rc}" -ne 0 ]]; then
  echo "ERROR: build phase failed (rc=${build_rc}); skipping GIN tests." >&2
  echo "----- last 60 lines of ${build_err} -----" >&2
  tail -n 60 "${build_err}" >&2 || true
  exit "${build_rc}"
fi

# --- test phase -> test-%j.{out,err}. Don't fail the sbatch on test failures
# (that would look like a build/infra failure); drop a `tests-failed` sentinel
# instead, which the workflow turns into a red (non-gating for gin) result. ---
test_rc=0
{
echo "==> [6/6] run GIN tests"
"${ci_dir}/run-gin-ci.sh"
} > "${test_out}" 2> "${test_err}" || test_rc=$?
echo "==> test phase logs: ${test_out} / ${test_err}"
if [[ "${test_rc}" -ne 0 ]]; then
  echo "GIN tests failed (rc=${test_rc}); non-gating (sentinel written)." >&2
  # Capture the "FAILED / SKIPPED RUNS" summary into the sentinel so the workflow
  # can show exactly which tests failed, not just a red check.
  sed -n '/=== FAILED \/ SKIPPED RUNS/,$p' "${test_out}" > "${logs_dir}/tests-failed" 2>/dev/null || true
  [[ -s "${logs_dir}/tests-failed" ]] \
    || echo "GIN tests failed (rc=${test_rc}); see the test log group." > "${logs_dir}/tests-failed"
fi
