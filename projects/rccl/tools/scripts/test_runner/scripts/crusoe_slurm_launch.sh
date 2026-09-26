#!/bin/bash
# Orchestration for the RCCL AINIC Crusoe gfx950 runner, executed on the SPUR
# submit node (the self-hosted GitHub runner). It:
#   1. takes the symmetric pair pinned by the preceding "Probe & pin GPU nodes"
#      step (INITIAL_PIN / RUNNER_TEMP/crusoe_pin.txt); if that pair is gone by
#      submit time it re-probes with scripts/crusoe_probe_lib.sh (self-healing),
#   2. submits scripts/crusoe_run_batch.sh with sbatch and waits until RUNNING,
#   3. streams the allocation's output and watches its terminal state,
#   4. self-heals around genuinely faulty hardware while treating preemption on
#      this oversubscribed cluster as a benign retry (no blacklisting).
#
# Inputs come from the environment (set by the workflow step): SALLOC_* ,
# INITIAL_PIN, GITHUB_WORKSPACE, TEST_RUNNER_DIR, RUNNER_TEMP, SCOPE,
# RUN_RCCL_BUILD_DIR, GTEST_BIN_DIR, PERF_BIN_DIR, WORKDIR, MPI_PATH, ROCM_PATH,
# SPUR_CONTROLLER_ADDR. Exits with the workload's exit code (0 = suites passed).
set -o pipefail
export PATH="/usr/local/bin:${PATH}"

# Probe / node-selection helpers (spur_cmd, probe_symmetric_pair, the auto-
# blacklist machinery, and the *_ARG builders) are shared with the probe step.
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=crusoe_probe_lib.sh
. "${HERE}/crusoe_probe_lib.sh"
probe_lib_init_args

# sbatch --export=ALL (default) forwards these to the batch job. Note
# RUN_RCCL_BUILD_DIR (not RCCL_BUILD_DIR): the bare name is only set inline on
# the test run inside the batch so it never leaks into Phase 0's build.
export RUN_RCCL_BUILD_DIR GTEST_BIN_DIR PERF_BIN_DIR SCOPE \
       GITHUB_WORKSPACE TEST_RUNNER_DIR RUNNER_TEMP \
       WORKDIR MPI_PATH ROCM_PATH SPUR_CONTROLLER_ADDR
export PYTHONPATH="${GITHUB_WORKSPACE}/${TEST_RUNNER_DIR}${PYTHONPATH:+:${PYTHONPATH}}"

# The batch payload is a committed script (no temp heredoc): sbatch runs it
# directly off NFS on the compute node(s).
BATCH_SCRIPT="${GITHUB_WORKSPACE}/${TEST_RUNNER_DIR}/scripts/crusoe_run_batch.sh"
SLURM_OUT="${RUNNER_TEMP}/crusoe_slurm.out"
: > "${SLURM_OUT}"
# Stale result from a previous attempt must not be mistaken for this one's.
rm -f "${RUNNER_TEMP}/crusoe_result.txt"

# The pair pinned by the probe step. Prefer the explicit env var; fall back to
# the file it wrote. Used on the first launch attempt; retries re-probe.
INITIAL_PIN="${INITIAL_PIN:-}"
if [ -z "${INITIAL_PIN}" ] && [ -f "${RUNNER_TEMP}/crusoe_pin.txt" ]; then
  INITIAL_PIN="$(tr -d '[:space:]' < "${RUNNER_TEMP}/crusoe_pin.txt")"
fi

job_state_of() {
  spur_cmd squeue -j "$1" -h -o '%T' 2>/dev/null | tr -d ' ' || true
}
job_reason_of() {
  spur_cmd squeue -j "$1" -h -o '%R' 2>/dev/null || true
}
job_scontrol_state() {
  spur_cmd spur show job "$1" 2>/dev/null | sed -n 's/.*JobState=\([^ ]*\).*/\1/p' | head -n1
}

# Wait until the job is RUNNING. PENDING is the expected busy-cluster path -- do
# not scancel it. Heartbeats go to stderr so the caller can capture only the
# terminal state on stdout. If a PINNED job stays PENDING past PIN_WAIT_SECS the
# pinned nodes were most likely grabbed by someone else between probe and submit
# (a race); echo PENDING_TIMEOUT so the caller releases this pair and re-probes
# another one instead of blocking on the same busy nodes until the GHA timeout.
wait_until_running() {
  local jobid="$1"
  local state="" reason="" js="" last_log=0 now
  local pend_deadline=$(( $(date +%s) + ${PIN_WAIT_SECS:-600} ))
  while true; do
    state=$(job_state_of "${jobid}")
    reason=$(job_reason_of "${jobid}")
    now=$(date +%s)
    if [ "$((now - last_log))" -ge 60 ]; then
      echo "$(date -u +%H:%M:%SZ) job ${jobid} state=${state:-?} reason=${reason:-?}" >&2
      last_log="${now}"
    fi
    case "${reason}" in
      *JobLaunchFailure*|*NODE_FAIL*)
        echo "Launch failed (${reason}); cancelling ${jobid}" >&2
        echo "NODE_FAIL"
        return 1
        ;;
    esac
    case "${state}" in
      RUNNING)
        echo "RUNNING"
        return 0
        ;;
      PENDING|CONFIGURING)
        if [ "$(date +%s)" -ge "${pend_deadline}" ]; then
          echo "PENDING_TIMEOUT"
          return 1
        fi
        sleep 15
        continue
        ;;
      COMPLETED|FAILED|NODE_FAIL|CANCELLED|TIMEOUT)
        echo "${state}"
        return 1
        ;;
      "")
        js=$(job_scontrol_state "${jobid}")
        case "${js}" in
          RUNNING)
            echo "RUNNING"
            return 0
            ;;
          PENDING|CONFIGURING)
            if [ "$(date +%s)" -ge "${pend_deadline}" ]; then
              echo "PENDING_TIMEOUT"
              return 1
            fi
            sleep 15
            continue
            ;;
          COMPLETED|FAILED|NODE_FAIL|CANCELLED|TIMEOUT)
            echo "${js}"
            return 1
            ;;
          "")
            echo "gone"
            return 1
            ;;
        esac
        sleep 15
        ;;
      *)
        sleep 15
        ;;
    esac
  done
}

echo "Submitting sbatch job (account=${SALLOC_ACCOUNT} partition=${SALLOC_PARTITION} nodes=${SALLOC_NODES} time=${SALLOC_TIME})..."
if [ -n "${INITIAL_PIN}" ]; then
  echo "Starting from probe-pinned pair: ${INITIAL_PIN}"
fi
MAX_LAUNCH_TRIES="${MAX_LAUNCH_TRIES:-8}"
JOBID=""
FINAL_RC=""
launch_fails=0
first_attempt=1
while [ "${launch_fails}" -lt "${MAX_LAUNCH_TRIES}" ]; do
  rebuild_exclude_arg
  # First attempt uses the pair the probe step already pinned; any retry re-probes
  # (the pinned pair was taken, faulted, or got preempted), so the self-healing
  # skip-busy / blacklist-faulty logic still spans submit -> monitor.
  if [ "${first_attempt}" = 1 ] && [ -n "${INITIAL_PIN}" ]; then
    PIN="${INITIAL_PIN}"
    first_attempt=0
  else
    first_attempt=0
    echo "Looking for ${SALLOC_NODES} symmetric idle nodes (excluded: ${SALLOC_EXCLUDE:-none}${BAD_NODES:+; auto-bad: ${BAD_NODES}})..."
    PIN=$(probe_symmetric_pair "${SALLOC_NODES}" || true)
    if [ -z "${PIN}" ] || [ "$(echo "${PIN}" | tr ',' '\n' | sed '/^$/d' | wc -l)" -lt "${SALLOC_NODES}" ]; then
      echo "No matching pair this round; waiting before another probe"
      sleep 30
      continue
    fi
  fi
  echo "Pinning probed symmetric nodes: ${PIN}"
  : > "${SLURM_OUT}"
  # Do not pass --ntasks=1: SPUR then allocates 1 node regardless of -N.
  sbatch_out=$(spur_cmd sbatch --parsable \
    --job-name=ainic-crusoe \
    --output="${SLURM_OUT}" \
    --error="${SLURM_OUT}" \
    --partition="${SALLOC_PARTITION}" \
    ${ACCOUNT_ARG} \
    ${RESV_ARG} \
    ${EXCLUDE_ARG} \
    --nodelist="${PIN}" \
    --nodes="${SALLOC_NODES}" \
    --ntasks="${SALLOC_NODES}" \
    --ntasks-per-node=1 \
    --cpus-per-task=32 \
    --gpus-per-node=8 \
    --time="${SALLOC_TIME}" \
    "${BATCH_SCRIPT}" 2>&1) || true
  echo "${sbatch_out}"
  JOBID=$(printf '%s\n' "${sbatch_out}" | grep -oE '^[0-9]+$' | tail -1)
  if [ -z "${JOBID}" ]; then
    echo "sbatch produced no job id; retrying"
    launch_fails=$((launch_fails + 1))
    sleep 15
    continue
  fi
  echo "Submitted SPUR job ${JOBID} on ${PIN}; waiting until RUNNING"
  state=$(wait_until_running "${JOBID}") || true
  if [ "${state}" != "RUNNING" ]; then
    # Never launched: could be a genuine node fault, or just contention/
    # preemption before start. Blacklist only nodes that are actually in a fault
    # state; otherwise retry on the same pool without discarding healthy
    # hardware.
    spur_cmd scancel "${JOBID}" >/dev/null 2>&1 || true
    if [ "${state}" = "PENDING_TIMEOUT" ]; then
      # The pinned pair never started within PIN_WAIT_SECS: the nodes were taken
      # between probe and submit. Do not preempt and do not burn a launch try --
      # release this pair and go probe another one (polite wait for free nodes).
      echo "Pinned pair ${PIN} did not start within ${PIN_WAIT_SECS:-600}s (nodes taken/contention); releasing and re-probing another pair"
      JOBID=""
      sleep 5
      continue
    fi
    faulty=$(nodes_faulty "${PIN}")
    if [ -n "${faulty}" ]; then
      echo "Did not reach RUNNING (state=${state:-gone}); node fault on ${faulty} -- blacklisting and retrying"
      blacklist_nodes "${faulty}"
    else
      echo "Did not reach RUNNING (state=${state:-gone}); nodes look healthy (contention/preemption) -- retrying without blacklisting"
    fi
    JOBID=""
    launch_fails=$((launch_fails + 1))
    sleep 10
    continue
  fi

  # RUNNING: stream the allocation's output and watch its terminal state.
  echo "Job ${JOBID} RUNNING on ${PIN}; streaming output"
  tail -n +1 -F "${SLURM_OUT}" 2>/dev/null &
  TAIL_PID=$!
  job_state=""
  while true; do
    st=$(job_state_of "${JOBID}")
    if [ -z "${st}" ]; then
      job_state=$(job_scontrol_state "${JOBID}")
      break
    fi
    case "${st}" in
      COMPLETED|FAILED|CANCELLED|NODE_FAIL|TIMEOUT)
        job_state="${st}"
        break
        ;;
    esac
    sleep 15
  done
  sleep 2
  kill "${TAIL_PID}" 2>/dev/null || true
  wait "${TAIL_PID}" 2>/dev/null || true

  exitcode=$(spur_cmd spur show job "${JOBID}" | sed -n 's/.*[[:space:]]ExitCode=\([0-9-]*\):.*/\1/p' | head -n1)
  echo "sbatch job ${JOBID} state=${job_state:-gone} ExitCode=${exitcode:-?}"
  case "${job_state}" in
    NODE_FAIL)
      # The node itself failed under the running allocation -- exclude the whole
      # pinned set and retry elsewhere.
      echo "Allocation ended as NODE_FAIL; blacklisting ${PIN} and retrying"
      blacklist_nodes "${PIN}"
      JOBID=""
      launch_fails=$((launch_fails + 1))
      sleep 10
      continue
      ;;
    CANCELLED|"")
      # On this shared, oversubscribed cluster a live allocation that is
      # cancelled without our doing is almost always PREEMPTION: a
      # higher-priority job reclaimed the node seconds after ours started. The
      # hardware is fine, so do NOT blacklist it (that would burn through good
      # nodes) -- only exclude nodes actually in a fault state, and retry.
      faulty=$(nodes_faulty "${PIN}")
      if [ -n "${faulty}" ]; then
        echo "Allocation ended as ${job_state:-gone}; node fault on ${faulty} -- blacklisting and retrying"
        blacklist_nodes "${faulty}"
      else
        echo "Allocation ended as ${job_state:-gone} (likely preempted); nodes healthy -- retrying without blacklisting"
      fi
      JOBID=""
      launch_fails=$((launch_fails + 1))
      sleep 10
      continue
      ;;
    COMPLETED)
      FINAL_RC="${exitcode:-0}"
      break
      ;;
    *)
      # FAILED / TIMEOUT: the batch script ran to a natural exit, so this is a
      # genuine build/test result. Report it; do not blacklist or retry (that
      # would mask real regressions).
      FINAL_RC="${exitcode:-1}"
      if [ "${FINAL_RC}" = "0" ]; then FINAL_RC=1; fi
      break
      ;;
  esac
done

if [ -z "${FINAL_RC}" ]; then
  echo "ERROR: no ${SALLOC_NODES}-node GPU allocation survived to a workload exit after ${MAX_LAUNCH_TRIES} attempts."
  echo "If allocations kept ending as CANCELLED on healthy nodes, they are being preempted -- this account/QOS is preemptible on a busy cluster. Fix by submitting under a non-preemptible QOS (e.g. --qos=amd-frameworks-ci-qos once z1_coco is associated with it), reserving nodes, or running when the cluster is idle."
  echo "Auto-excluded faulty nodes this run: ${BAD_NODES:-none}"
  exit 1
fi

echo "sbatch job ${JOBID} state=${job_state} ExitCode=${exitcode:-?} rc=${FINAL_RC}"
exit "${FINAL_RC}"
