#!/usr/bin/env bash
#SBATCH --job-name=rccl-ext-plugins
#SBATCH --nodes=1
#SBATCH --ntasks=8
#SBATCH --partition=interactive
#SBATCH --exclusive
#SBATCH --time=04:00:00
#SBATCH --output=%x-%j.out
#SBATCH --error=%x-%j.err
#
# Compute-node orchestrator for the RCCL ext-plugins CI suite, invoked from
# .github/workflows/rccl-suite.yml (suite=ext-plugins) via submit_slurm_job.py.
# On one multi-GPU node:
#   1. consume ROCm (provisioned on the head node) via .ci-out/rocm.env
#   2. projects/rccl  install.sh              build RCCL from PR sources
#   3. lib/build-ompi-ucx.sh                  build a UCX-enabled OpenMPI (isolated)
#   4. projects/rccl-tests                    build the *_perf binaries (with MPI)
#   5. plugins/tuner/example                  build + install libnccl-tuner-example.so
#   6. test/ext-plugins                       run pytest -m "ext_mixed or ext_tuner"
#
# Scope: the mixed-plugin tests (the pytest port of plugins/mixed/example/test.sh
# plus the new runtime check) and the CSV tuner tests. The tuner tests launch
# mpirun with `--mca pml ucx`, so this suite uses its OWN UCX-enabled OpenMPI via
# build-ompi-ucx.sh, kept fully isolated from the gin/device-api OpenMPI. The
# profiler/inspector suites additionally need cuda.h for their example .so and
# are left as a follow-up.
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

# Per-phase wall-clock marks. Embedded in the build log so the CI job can report
# exactly how long each build step took (independent of when logs are dumped).
_bt0=$(date +%s)
_pt(){ echo "PHASE_TS $1 epoch=$(date +%s) iso=$(date -Is) elapsed=$(( $(date +%s) - _bt0 ))s"; }

# 1) RCCL from the PR sources.
_pt "build:rccl:start"
echo "==> [1/5] build RCCL (targets=${gpu_targets})"
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

# 2) UCX-enabled OpenMPI, isolated from the gin/device-api OpenMPI. The tuner
# tests launch `mpirun --mca pml ucx`, which the shared build lacks.
_pt "build:ompi-ucx:start"
echo "==> [2/5] build UCX-enabled OpenMPI (isolated)"
"${ci_dir}/lib/build-ompi-ucx.sh"
set -a
# shellcheck source=/dev/null
source "${workdir}/.ci-out/ompi-ucx.env"
set +a
export PATH="${MPI_HOME}/bin:${PATH}"
if [[ ! -x "${MPI_HOME}/bin/mpirun" ]]; then
  echo "ERROR: no mpirun under ${MPI_HOME}/bin" >&2
  exit 1
fi

# 3) rccl-tests WITH MPI: the tuner tests drive several collectives via mpirun,
# so build all the *_perf binaries. Use -k so one failing target does not sink
# the rest; the required binaries are checked explicitly below.
_pt "build:rccl-tests:start"
echo "==> [3/5] build rccl-tests (*_perf, USE_MPI=ON)"
rccl_tests_build="${rccl_tests_src}/build"
rm -rf "${rccl_tests_build}"; mkdir -p "${rccl_tests_build}"
(
  cd "${rccl_tests_build}"
  cmake -DCMAKE_BUILD_TYPE=Release -DUSE_MPI=ON \
    -DRCCL_DIR="${rccl_cmake_dir}" \
    -DCMAKE_PREFIX_PATH="${rccl_prefix};${MPI_HOME}" \
    -DGPU_TARGETS="${gpu_targets}" \
    ..
  make -j"${build_jobs}" -k || true
)
missing=""
for b in all_reduce_perf all_gather_perf broadcast_perf reduce_perf reduce_scatter_perf; do
  [[ -x "${rccl_tests_build}/${b}" ]] || missing="${missing} ${b}"
done
if [[ -n "${missing}" ]]; then
  echo "ERROR: rccl-tests missing perf binaries:${missing}" >&2
  ls -la "${rccl_tests_build}" 2>/dev/null >&2 || true
  exit 1
fi

# 4) CSV tuner example plugin. Installed where conftest.py expects it
# (RCCL_INSTALL_DIR/plugins/tuner/example); its presence is also what makes
# conftest run (not skip) the ext_tuner tests.
_pt "build:tuner-so:start"
echo "==> [4/5] build + install libnccl-tuner-example.so"
tuner_src="${rccl_src}/plugins/tuner/example"
tuner_dst="${rccl_install_dir}/plugins/tuner/example"
make -C "${tuner_src}" -j"${build_jobs}" default
tuner_so="$(find "${tuner_src}" -name libnccl-tuner-example.so -type f 2>/dev/null | head -n1)"
if [[ -z "${tuner_so}" ]]; then
  echo "ERROR: libnccl-tuner-example.so not produced under ${tuner_src}" >&2
  exit 1
fi
mkdir -p "${tuner_dst}"
cp -f "${tuner_so}" "${tuner_dst}/libnccl-tuner-example.so"
echo "    installed -> ${tuner_dst}/libnccl-tuner-example.so"

# 5) Hand the install/test locations to the run stage. RCCL_INSTALL_DIR is the
# librccl.so dir and the plugin root; OMPI_INSTALL_DIR/UCX come from ompi-ucx.env
# and are re-sourced there so the runtime LD_LIBRARY_PATH resolves UCX.
{
  printf 'export RCCL_INSTALL_DIR=%q\n' "${rccl_install_dir}"
  printf 'export RCCL_TESTS_DIR=%q\n'   "${rccl_tests_src}"
  printf 'export OMPI_INSTALL_DIR=%q\n' "${MPI_HOME}"
} > "${workdir}/.ci-out/ext-plugins.env"
_pt "build:end"
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
_tt0=$(date +%s)
_pt(){ echo "PHASE_TS $1 epoch=$(date +%s) iso=$(date -Is) elapsed=$(( $(date +%s) - _tt0 ))s"; }
_pt "test:start"
echo "==> [5/5] run ext-plugins pytest (-m 'ext_mixed or ext_tuner')"
# UCX mpirun registers RDMA memory via ibv_reg_mr; SLURM jobs submitted from the
# GHA runner often inherit RLIMIT_MEMLOCK=8192 kB. Raise it for pytest/mpirun.
ulimit -l unlimited 2>/dev/null || echo "WARN: could not raise RLIMIT_MEMLOCK (ulimit -l)"
echo "==> RLIMIT_MEMLOCK (ulimit -l) = $(ulimit -l)"
set -a
# shellcheck source=/dev/null
source "${workdir}/.ci-out/ompi-ucx.env"
# shellcheck source=/dev/null
source "${workdir}/.ci-out/ext-plugins.env"
set +a
echo "    RCCL_INSTALL_DIR=${RCCL_INSTALL_DIR}"
echo "    RCCL_TESTS_DIR=${RCCL_TESTS_DIR}"
echo "    OMPI_INSTALL_DIR=${OMPI_INSTALL_DIR}"

ext_dir="${workdir}/projects/rccl/test/ext-plugins"
venv_dir="${workdir}/.ci-out/ext-plugins-venv"
python3 -m venv --system-site-packages "${venv_dir}"
# shellcheck source=/dev/null
source "${venv_dir}/bin/activate"
python3 -m pip install --quiet --disable-pip-version-check -r "${ext_dir}/requirements.txt"

# conftest.py resolves assets/logs from the current working directory.
cd "${ext_dir}"

# Each tuner test captures the whole rccl-tests stdout, RCCL debug logging
# included. ${workdir} lives on the shared NFS mount the GHA runner uses, and
# writing those logs there is what dominates the suite runtime: the same
# broadcast run takes 5.9s with its 57 MB log on node-local disk and 865s with it
# on that NFS. Keep the logs node-local (world-writable /tmp, so this works for
# any user the job runs as) and copy them back afterwards for artifact upload.
export RCCL_EXT_PLUGINS_LOGDIR="${SLURM_TMPDIR:-/tmp}/rccl-ext-plugins-logs-${jid}"
rm -rf "${RCCL_EXT_PLUGINS_LOGDIR}"
mkdir -p "${RCCL_EXT_PLUGINS_LOGDIR}"
echo "    RCCL_EXT_PLUGINS_LOGDIR=${RCCL_EXT_PLUGINS_LOGDIR}"

# Safety net: cap any single test at 10 min (pytest-timeout, signal method so it
# interrupts a blocked subprocess). A hung test then fails fast instead of eating
# the whole SLURM allocation, and the remaining tests still run.
pytest_rc=0
python3 -m pytest -v -rs --color=no --timeout=600 --timeout-method=signal \
  -m "ext_mixed or ext_tuner" || pytest_rc=$?

mkdir -p "${ext_dir}/logs"
cp -r "${RCCL_EXT_PLUGINS_LOGDIR}/." "${ext_dir}/logs/" 2>/dev/null || true
echo "==> test logs copied to ${ext_dir}/logs ($(du -sh "${ext_dir}/logs" 2>/dev/null | cut -f1))"
_pt "test:end"
# Make pytest's status the status of this block so the sentinel logic below runs.
(exit "${pytest_rc}")
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
