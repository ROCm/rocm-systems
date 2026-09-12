#!/bin/bash
# "Probe & pin GPU nodes" workflow step for the RCCL AINIC Crusoe gfx950 runner.
#
# Runs on the SPUR submit node (the self-hosted runner) BEFORE the submit/run
# step, so the GitHub Actions checklist shows node selection as its own pass/fail
# instead of being buried inside the run step's log. It probes idle GPU nodes and
# pins a symmetric pair (see scripts/crusoe_probe_lib.sh), then hands the pinned
# nodes to the run step via GITHUB_OUTPUT (`pin=...`) and RUNNER_TEMP/crusoe_pin.txt.
#
# Behaviour on a busy cluster (deliberate): a saturated cluster is NOT a failure.
# This step waits patiently, re-probing until a healthy symmetric pair frees up
# (bounded only by the job's timeout-minutes), matching the previous in-loop
# behaviour -- it only goes red on a genuine probe error or the overall timeout,
# never merely because the cluster is currently full.
#
# Inputs (from the workflow step env): SALLOC_* , GITHUB_WORKSPACE,
# TEST_RUNNER_DIR, RUNNER_TEMP, SPUR_CONTROLLER_ADDR.
set -o pipefail
export PATH="/usr/local/bin:${PATH}"

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=crusoe_probe_lib.sh
. "${HERE}/crusoe_probe_lib.sh"

export PYTHONPATH="${GITHUB_WORKSPACE}/${TEST_RUNNER_DIR}${PYTHONPATH:+:${PYTHONPATH}}"
probe_lib_init_args

echo "Probing for ${SALLOC_NODES} symmetric idle GPU nodes (partition=${SALLOC_PARTITION}, excluded=${SALLOC_EXCLUDE:-none})..."
# A busy cluster is not a failure -- keep waiting. But a pair that is idle and
# symmetric yet cannot bootstrap mpirun (mpi_smoke_pair) is blacklisted, and if
# that keeps happening the OOB interface / fabric is broken cluster-wide, not a
# stray bad node: surface that instead of silently waiting out the job timeout.
PROBE_MAX_SMOKE_REJECTS="${PROBE_MAX_SMOKE_REJECTS:-10}"
# Fresh blacklist for this run (RUNNER_TEMP is reused across runs).
reset_bad_nodes
PIN=""
while true; do
  rebuild_exclude_arg
  PIN=$(probe_symmetric_pair "${SALLOC_NODES}" || true)
  if [ -n "${PIN}" ] && [ "$(printf '%s\n' "${PIN}" | tr ',' '\n' | sed '/^$/d' | wc -l)" -ge "${SALLOC_NODES}" ]; then
    break
  fi
  # In this step BAD_NODES only grows from mpi_smoke_pair rejections (no
  # allocation faults here), so its size is the count of idle-but-unusable
  # nodes proven this run. probe_symmetric_pair runs in a $() subshell, so pull
  # the rejections it recorded back from the file before counting them.
  sync_bad_nodes
  nrej=$(printf '%s\n' "${BAD_NODES}" | tr ',' '\n' | sed '/^$/d' | wc -l)
  if [ "${nrej}" -ge "${PROBE_MAX_SMOKE_REJECTS}" ]; then
    echo "::error::MPI bootstrap smoke failed on ${nrej} idle nodes over ${PROBE_OOB_IF:-ens3} (e.g. \"an ORTE daemon has unexpectedly failed ... no route found between them\"). This is not a busy cluster: OpenMPI cannot bring up daemons across nodes on the pinned OOB interface, so every suite would die at launch. Check the ${PROBE_OOB_IF:-ens3} routing/fabric or the oob_tcp_if_include/btl_tcp_if_include in the ainic mpi_args. Blacklisted: ${BAD_NODES}"
    exit 1
  fi
  echo "No usable symmetric pair yet (cluster busy, or ${nrej} nodes failed the MPI smoke); waiting before another probe..."
  sleep 30
done

echo "Pinned symmetric pair: ${PIN}"
printf '%s\n' "${PIN}" > "${RUNNER_TEMP}/crusoe_pin.txt"
if [ -n "${GITHUB_OUTPUT:-}" ]; then
  echo "pin=${PIN}" >> "${GITHUB_OUTPUT}"
fi
