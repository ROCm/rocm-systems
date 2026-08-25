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
PIN=""
while true; do
  rebuild_exclude_arg
  PIN=$(probe_symmetric_pair "${SALLOC_NODES}" || true)
  if [ -n "${PIN}" ] && [ "$(printf '%s' "${PIN}" | tr ',' '\n' | sed '/^$/d' | wc -l)" -ge "${SALLOC_NODES}" ]; then
    break
  fi
  echo "No idle symmetric pair yet (cluster busy); waiting before another probe..."
  sleep 30
done

echo "Pinned symmetric pair: ${PIN}"
printf '%s\n' "${PIN}" > "${RUNNER_TEMP}/crusoe_pin.txt"
if [ -n "${GITHUB_OUTPUT:-}" ]; then
  echo "pin=${PIN}" >> "${GITHUB_OUTPUT}"
fi
