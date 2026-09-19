#!/bin/bash
# Batch payload for the RCCL AINIC Crusoe gfx950 runner.
#
# Submitted with `sbatch` by scripts/crusoe_slurm_launch.sh and executed on the
# GPU compute node(s). SPUR forwards the launch step's environment via
# `sbatch --export=ALL`, so this script reads its inputs (GITHUB_WORKSPACE,
# TEST_RUNNER_DIR, RUNNER_TEMP, SCOPE, RUN_RCCL_BUILD_DIR, GTEST_BIN_DIR,
# PERF_BIN_DIR, ...) straight from the environment.
#
# It runs two phases inside the single allocation (build then test -- they share
# one allocation because each sbatch submission is a fresh, separately-queued and
# separately-preemptible allocation on this oversubscribed cluster):
#   Phase 0  build RCCL fresh with --enable-mpi-tests on the compute node
#   Phase 1  run the selected AINIC suites against that fresh build
# The phase that ran and its exit code are written to
# ${RUNNER_TEMP}/crusoe_result.txt (phase=build|test, rc=N) so the caller and the
# job summary can report exactly what failed without digging through the log.
set -o pipefail
ulimit -l unlimited 2>/dev/null || true

RESULT_FILE="${RUNNER_TEMP}/crusoe_result.txt"
# Record per-phase exit codes so the workflow can surface a Build vs Test result
# as separate checklist steps: build_rc is always written; test_rc only once the
# test phase runs (i.e. build passed). phase=/rc= are kept for the legacy summary
# step: they point at the phase that determined the outcome (build if it failed,
# otherwise test).
record_result() {
  # $1 = build_rc, $2 = test_rc (empty if the test phase did not run)
  local build_rc="$1" test_rc="${2:-}"
  {
    printf 'build_rc=%s\n' "${build_rc}"
    if [ -n "${test_rc}" ]; then
      printf 'test_rc=%s\n' "${test_rc}"
    fi
    if [ "${build_rc}" != 0 ] || [ -z "${test_rc}" ]; then
      printf 'phase=build\nrc=%s\n' "${build_rc}"
    else
      printf 'phase=test\nrc=%s\n' "${test_rc}"
    fi
  } > "${RESULT_FILE}" 2>/dev/null || true
}

# SPUR runs the batch script on every allocated node. Elect one launcher; extra
# copies wait on an NFS sentinel so the job can leave COMPLETING.
_done="${RUNNER_TEMP}/ainic_launcher_done_${SLURM_JOB_ID:-$$}"
_first=$(PYTHONPATH="${GITHUB_WORKSPACE}/${TEST_RUNNER_DIR}${PYTHONPATH:+:${PYTHONPATH}}" python3 -c 'from lib.test_executor import expand_slurm_hostlist; import os; h=expand_slurm_hostlist(os.environ.get("SLURM_JOB_NODELIST") or os.environ.get("SLURM_NODELIST") or ""); print(h[0] if h else "")')
if [ -n "${_first}" ] && [ "$(hostname -s)" != "${_first}" ]; then
  echo "Parking extra SPUR copy on $(hostname -s); launcher is ${_first}"
  while [ ! -f "${_done}" ]; do sleep 5; done
  exit 0
fi
trap 'rm -f "${PARK_SENTINEL:-}"; touch "${_done}"' EXIT

# $HOME is NFS without O_TMPFILE; clang needs a local tmpdir or the gfx950 probe
# fails and the library builds with no device code.
export TMPDIR="/dev/shm/rccl-ci-${SLURM_JOB_ID:-$$}"
mkdir -p "${TMPDIR}"
export PATH="/opt/openmpi/bin:/opt/rocm/bin:${PATH}"
export LD_LIBRARY_PATH="/opt/ucx/lib:/opt/openmpi/lib:/opt/rocm/lib:${LD_LIBRARY_PATH:-}"
unset HIP_VISIBLE_DEVICES ROCR_VISIBLE_DEVICES CUDA_VISIBLE_DEVICES
# Allocation is 1 task/node; RCCL tests need 8 ranks/node. OpenMPI's slurm RAS
# reads this; test_runner also passes --host n:8.
if [ -n "${SLURM_NNODES:-}" ]; then
  export SLURM_TASKS_PER_NODE="8(x${SLURM_NNODES})"
fi
# SPUR srun is not Slurm srun. Point OpenMPI plm slurm at the in-tree shim.
SHIMDIR="/dev/shm/rccl-srun-${SLURM_JOB_ID:-$$}"
mkdir -p "${SHIMDIR}"
cp "${GITHUB_WORKSPACE}/${TEST_RUNNER_DIR}/scripts/spur_srun.sh" "${SHIMDIR}/srun"
chmod +x "${SHIMDIR}/srun"
export PATH="${SHIMDIR}:${PATH}"
export OMPI=/opt/openmpi
export PARK_SENTINEL="/dev/shm/crusoe-mpirun.active"
touch "${PARK_SENTINEL}"
cd "${GITHUB_WORKSPACE}/${TEST_RUNNER_DIR}" || exit 1
HOSTFILE="${RUNNER_TEMP}/mpi_hostfile"
python3 - "${HOSTFILE}" <<'PY'
import os, sys
from lib.test_executor import expand_slurm_hostlist
path = sys.argv[1]
nl = os.environ.get("SLURM_JOB_NODELIST") or os.environ.get("SLURM_NODELIST") or ""
hosts = expand_slurm_hostlist(nl)
open(path, "w", encoding="utf-8").write("\n".join(hosts) + ("\n" if hosts else ""))
print("MPI hosts:", ",".join(hosts))
PY
export RCCL_TEST_MPI_HOSTFILE="${HOSTFILE}"

# Phase 0: build RCCL with --enable-mpi-tests on THIS compute node rather than
# the submit node. RCCL_BUILD_DIR is intentionally not in the environment here
# (only RUN_RCCL_BUILD_DIR is), so test_runner does a real build instead of
# treating it as a prebuilt library: test_runner.py invokes install.sh, which
# defaults -j to $(nproc) and therefore uses ALL of the compute node's cores.
# WORKDIR points at this run's own checkout so install.sh and the build tree
# resolve there (on NFS, shared with every node in the allocation). The build is
# always required: gtest suites need the fresh rccl-UnitTestsMPI binary, and the
# perf suites load the freshly built librccl.so via LD_LIBRARY_PATH.
echo "================================================================"
echo "Phase 0: build RCCL (--enable-mpi-tests) on $(hostname) [nproc=$(nproc)]"
echo "================================================================"
WORKDIR="${GITHUB_WORKSPACE}/projects/rccl" \
python3 test_runner.py \
  --config configs/mi355x_ainic_crusoe_roce.json \
  --system ainic \
  --skip-tests \
  2>&1 | tee "${RUNNER_TEMP}/crusoe_build.log"
build_rc=$?
if [ "${build_rc}" -ne 0 ]; then
  echo "ERROR: RCCL build failed (rc=${build_rc}); aborting batch job."
  record_result "${build_rc}"
  exit "${build_rc}"
fi
echo "== rccl-UnitTestsMPI =="
ls -la "${GITHUB_WORKSPACE}/projects/rccl/build/debug/test/rccl-UnitTestsMPI" 2>&1 || true

# Phase 1: single test run. One test_runner invocation executes every selected
# suite. Each suite resolves its own binary directory from the config
# (GTEST_BIN_DIR for gtest suites, PERF_BIN_DIR for rccl-tests suites), so no
# group split is needed here. --scope selects nightly (all enabled) vs smoke
# (only suites marked smoke).
echo "================================================================"
echo "Phase 1: run AINIC suites (--scope ${SCOPE})"
echo "================================================================"
RCCL_BUILD_DIR="${RUN_RCCL_BUILD_DIR}" \
GTEST_BIN_DIR="${GTEST_BIN_DIR}" \
PERF_BIN_DIR="${PERF_BIN_DIR}" \
python3 test_runner.py \
  --config configs/mi355x_ainic_crusoe_roce.json \
  --system ainic \
  --no-build \
  --scope "${SCOPE}" \
  --report-suffix ainic \
  2>&1 | tee "${RUNNER_TEMP}/crusoe_run.log"
test_rc=$?
record_result "${build_rc}" "${test_rc}"
exit "${test_rc}"
