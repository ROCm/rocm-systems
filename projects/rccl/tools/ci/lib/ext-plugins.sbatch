#!/usr/bin/env bash
#SBATCH --job-name=rccl-ext-plugins
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --partition=interactive
#SBATCH --exclusive
#SBATCH --time=02:00:00
#SBATCH --output=%x-%j.out
#SBATCH --error=%x-%j.err
#
# Compute-node orchestrator for the RCCL ext-plugins CI suite, invoked from
# .github/workflows/rccl-suite.yml (suite=ext-plugins) via submit_slurm_job.py.
# On one multi-GPU node:
#   1. consume ROCm (provisioned on the head node) via .ci-out/rocm.env
#   2. projects/rccl  install.sh              build RCCL from PR sources
#   3. projects/rccl-tests                    build all_reduce_perf (no MPI)
#   4. test/ext-plugins                       run pytest -m ext_mixed
#
# Scope: only the mixed-plugin tests (the pytest port of
# plugins/mixed/example/test.sh plus the new runtime check). They build their
# own libnccl-mixed.so from in-tree headers and drive all_reduce_perf in a
# single multi-GPU process, so they need no OpenMPI and no prebuilt example
# plugins. The tuner/profiler/inspector suites need a full OpenMPI + every
# *_perf binary (and cuda.h for the profiler/inspector examples); wiring those
# in is a separate, heavier follow-up.
#
# Required env (from the workflow via `sbatch --export=ALL,...`):
#   RCCL_DEVICE_API_WORKDIR   GHA workspace (rocm-systems root)
# Optional env:
#   RCCL_AMDGPU_TARGETS       GPU arch (default: gfx950)

set -euo pipefail

echo "=== sbatch ext-plugins on $(hostname) ==="
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

# ROCm floor (provisioned on the head node from TheRock artifacts). Sourced at
# top level so BOTH the build and test phases inherit ROCM_PATH/PATH/lib paths;
# the runtime pytest (all_reduce_perf) needs the ROCm runtime libraries too.
rocm_env="${workdir}/.ci-out/rocm.env"
if [[ ! -f "${rocm_env}" ]]; then
  echo "ERROR: ${rocm_env} not found. The workflow must run 'Provision ROCm" >&2
  echo "       from TheRock artifacts' on the head node before sbatch." >&2
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
export HIP_PATH="${ROCM_PATH}"
export PATH="${ROCM_PATH}/bin:${ROCM_PATH}/llvm/bin:${PATH}"
export LD_LIBRARY_PATH="${ROCM_PATH}/lib:${LD_LIBRARY_PATH:-}"
export CMAKE_PREFIX_PATH="${ROCM_PATH}/lib/cmake:${ROCM_PATH}${CMAKE_PREFIX_PATH:+:${CMAKE_PREFIX_PATH}}"

rccl_prefix="${workdir}/.ci-out/rccl-install"
# conftest.py reads RCCL_INSTALL_DIR both as the librccl.so dir (LD_LIBRARY_PATH)
# and as the plugin root (RCCL_INSTALL_DIR/plugins/...), so it points at lib/.
rccl_install_dir="${rccl_prefix}/lib"
rccl_tests_src="${workdir}/projects/rccl-tests"

# --- build phase -> build-%j.{out,err}. Runs in a subshell so any failure (a
# hard `exit` or a set -e abort) is caught below and skips the test phase. ---
build_rc=0
(
# Ensure `import yaml` works for the RCCL build (some cmake steps need it).
# shellcheck source=/dev/null
source "${ci_dir}/lib/ensure-python-yaml.sh"

# 1) RCCL from the PR sources.
echo "==> [1/3] build RCCL (targets=${gpu_targets})"
rccl_src="${workdir}/projects/rccl"
rm -rf "${rccl_prefix}"
(
  cd "${rccl_src}"
  ./install.sh -j "${build_jobs}" \
    --amdgpu_targets="${gpu_targets}" \
    --prefix="${rccl_prefix}" \
    --no_clean
)
if [[ ! -f "${rccl_install_dir}/librccl.so" && ! -f "${rccl_install_dir}/librccl.so.1" ]]; then
  echo "ERROR: no librccl.so under ${rccl_install_dir}" >&2
  ls -la "${rccl_install_dir}" 2>/dev/null >&2 || true
  exit 1
fi

# Locate the in-tree RCCL CMake package config so rccl-tests links THIS build
# rather than any RCCL under the ROCm floor (same pin the gin suite uses).
rccl_cfg="$(ls "${rccl_prefix}"/lib*/cmake/rccl/*[Cc]onfig.cmake 2>/dev/null | head -n1)"
if [[ -z "${rccl_cfg}" ]]; then
  echo "ERROR: no RCCL cmake config under ${rccl_prefix}/lib*/cmake/rccl" >&2
  exit 1
fi
rccl_cmake_dir="$(dirname "${rccl_cfg}")"
echo "    in-tree RCCL cmake config: ${rccl_cmake_dir}"

# 2) rccl-tests: only all_reduce_perf is needed, run as a single multi-GPU
# process (no MPI launcher), so build without MPI to keep it light.
echo "==> [2/3] build rccl-tests (all_reduce_perf, no MPI)"
rccl_tests_build="${rccl_tests_src}/build"
rm -rf "${rccl_tests_build}"; mkdir -p "${rccl_tests_build}"
(
  cd "${rccl_tests_build}"
  cmake -DCMAKE_BUILD_TYPE=Release -DUSE_MPI=OFF \
    -DRCCL_DIR="${rccl_cmake_dir}" \
    -DCMAKE_PREFIX_PATH="${rccl_prefix}" \
    -DGPU_TARGETS="${gpu_targets}" \
    ..
  make -j"${build_jobs}" all_reduce_perf
)
if [[ ! -x "${rccl_tests_build}/all_reduce_perf" ]]; then
  echo "ERROR: all_reduce_perf not built under ${rccl_tests_build}" >&2
  ls -la "${rccl_tests_build}" 2>/dev/null >&2 || true
  exit 1
fi

# 3) Hand the install/test locations to the run stage. RCCL_INSTALL_DIR is used
# by the mixed runtime test for LD_LIBRARY_PATH (librccl.so); RCCL_TESTS_DIR is
# where it finds build/all_reduce_perf.
{
  printf 'export RCCL_INSTALL_DIR=%q\n' "${rccl_install_dir}"
  printf 'export RCCL_TESTS_DIR=%q\n'   "${rccl_tests_src}"
} > "${workdir}/.ci-out/ext-plugins.env"
) > "${build_out}" 2> "${build_err}" || build_rc=$?
echo "==> build phase logs: ${build_out} / ${build_err}"

# Skip the test phase if the build failed (testing a broken build is pointless).
if [[ "${build_rc}" -ne 0 ]]; then
  echo "ERROR: build phase failed (rc=${build_rc}); skipping ext-plugins tests." >&2
  echo "----- last 60 lines of ${build_err} -----" >&2
  tail -n 60 "${build_err}" >&2 || true
  exit "${build_rc}"
fi

# --- test phase -> test-%j.{out,err}. Don't fail the sbatch on test failures
# (that would look like a build/infra failure); drop a `tests-failed` sentinel
# instead, which the workflow turns into a red (non-gating) result. ---
test_rc=0
{
echo "==> [3/3] run ext-plugins pytest (-m ext_mixed)"
set -a
# shellcheck source=/dev/null
source "${workdir}/.ci-out/ext-plugins.env"
set +a
echo "    RCCL_INSTALL_DIR=${RCCL_INSTALL_DIR}"
echo "    RCCL_TESTS_DIR=${RCCL_TESTS_DIR}"

ext_dir="${workdir}/projects/rccl/test/ext-plugins"
venv_dir="${workdir}/.ci-out/ext-plugins-venv"
python3 -m venv --system-site-packages "${venv_dir}"
# shellcheck source=/dev/null
source "${venv_dir}/bin/activate"
python3 -m pip install --quiet --disable-pip-version-check -r "${ext_dir}/requirements.txt"

# conftest.py resolves assets/logs from the current working directory.
cd "${ext_dir}"
python3 -m pytest -v -rs --color=no -m ext_mixed
} > "${test_out}" 2> "${test_err}" || test_rc=$?
echo "==> test phase logs: ${test_out} / ${test_err}"
if [[ "${test_rc}" -ne 0 ]]; then
  echo "ext-plugins tests failed (rc=${test_rc}); non-gating (sentinel written)." >&2
  # Capture pytest's short summary into the sentinel so the workflow can show
  # exactly which tests failed, not just a red check.
  sed -n '/=* short test summary info =*/,$p' "${test_out}" > "${logs_dir}/tests-failed" 2>/dev/null || true
  [[ -s "${logs_dir}/tests-failed" ]] \
    || echo "ext-plugins tests failed (rc=${test_rc}); see the test log group." > "${logs_dir}/tests-failed"
fi
