#!/bin/bash
# Shared probe / node-selection helpers for the RCCL AINIC Crusoe gfx950 runner.
#
# Sourced by BOTH:
#   - scripts/crusoe_probe_pin.sh    (the "Probe & pin GPU nodes" workflow step),
#     which does the initial patient probe and pins a symmetric idle pair, and
#   - scripts/crusoe_slurm_launch.sh (the "Submit & run" step), which re-probes
#     inside its retry loop so the self-healing (skip busy pairs, blacklist only
#     genuinely faulty nodes) still spans submit -> monitor.
#
# Keeping this logic in one file is why the probe can be its own workflow step
# without duplicating the fingerprint/selection code. Functions read their inputs
# (SALLOC_*, GITHUB_WORKSPACE, TEST_RUNNER_DIR, RUNNER_TEMP, SPUR_CONTROLLER_ADDR)
# from the environment; sourcing this file defines functions only (no side
# effects) -- call probe_lib_init_args once before using the *_ARG values.

# SPUR CLI can wedge; SIGTERM is not enough, so -k is required.
spur_cmd() { PATH="/usr/local/bin:/usr/bin:/bin:${PATH:-}" /usr/bin/timeout -k 5 30 "$@"; }

# Static sbatch args derived from SALLOC_* (reservation/exclude/account). The
# exclude here is the operator-provided base; rebuild_exclude_arg folds in the
# nodes auto-blacklisted during a run.
RESV_ARG=""
EXCLUDE_ARG=""
ACCOUNT_ARG=""
probe_lib_init_args() {
  if [ -n "${SALLOC_RESERVATION:-}" ]; then RESV_ARG="--reservation=${SALLOC_RESERVATION}"; else RESV_ARG=""; fi
  if [ -n "${SALLOC_EXCLUDE:-}" ]; then EXCLUDE_ARG="--exclude=${SALLOC_EXCLUDE}"; else EXCLUDE_ARG=""; fi
  if [ -n "${SALLOC_ACCOUNT:-}" ]; then ACCOUNT_ARG="--account=${SALLOC_ACCOUNT}"; else ACCOUNT_ARG=""; fi
}

expand_idle() {
  python3 -c 'import sys; from lib.test_executor import expand_slurm_hostlist; print("\n".join(expand_slurm_hostlist(sys.argv[1])))' "$1"
}

# Probe idle GPU nodes. Keep only hosts with 8 gfx950, ionic_0..7 in both sysfs
# and ibv_devices, GID-index-1 on every ionic, and the same CPU/NUMA/NIC
# fingerprint. stdout is host1,host2,... or empty (caller waits and retries).
probe_symmetric_pair() {
  local need="$1"
  local spec hosts n j waited probe_ids probe_dir nready
  spec=$(spur_cmd sinfo -p "${SALLOC_PARTITION}" -h -t idle -o '%N' 2>/dev/null | head -1 || true)
  hosts=$(expand_idle "${spec}")
  if [ "$(echo "${hosts}" | sed '/^$/d' | wc -l)" -lt "${need}" ]; then
    echo "Not enough idle nodes to probe (need ${need})" >&2
    return 1
  fi
  probe_dir="${RUNNER_TEMP}/probes"
  mkdir -p "${probe_dir}"
  rm -f "${probe_dir}"/*.out
  probe_ids=""
  local nprobe=0
  for n in ${hosts}; do
    [ "${nprobe}" -lt 16 ] || break
    nprobe=$((nprobe + 1))
    rm -f "${probe_dir}/${n}.out"
    j=$(spur_cmd sbatch --parsable \
      --job-name="ainic-probe-${n}" \
      --output="${probe_dir}/${n}.out" \
      --error="${probe_dir}/${n}.out" \
      --partition="${SALLOC_PARTITION}" \
      ${ACCOUNT_ARG} \
      ${RESV_ARG} \
      ${EXCLUDE_ARG} \
      --nodes=1 \
      --ntasks=1 \
      --cpus-per-task=8 \
      --gpus-per-node=8 \
      --nodelist="${n}" \
      --time=00:05:00 \
      --wrap "bash ${GITHUB_WORKSPACE}/${TEST_RUNNER_DIR}/scripts/crusoe_node_probe.sh" 2>&1 \
      | tee /dev/stderr | grep -oE '^[0-9]+$' | tail -1 || true)
    if [ -n "${j}" ]; then
      probe_ids="${probe_ids} ${j}"
    else
      echo "probe sbatch failed for ${n}" >&2
    fi
  done
  echo "Launched probes:${probe_ids}" >&2
  waited=0
  while [ "${waited}" -lt 90 ]; do
    sleep 10
    waited=$((waited + 10))
    nready=$( { grep -hE '^(OK|BAD) ' "${probe_dir}"/*.out 2>/dev/null || true; } | wc -l)
    if [ "${nready}" -ge "${nprobe}" ]; then
      break
    fi
  done
  for j in ${probe_ids}; do
    spur_cmd scancel "${j}" >/dev/null 2>&1 || true
  done
  echo "=== probe results ===" >&2
  grep -hE '^(OK|BAD) ' "${probe_dir}"/*.out 2>/dev/null >&2 || true
  local pair
  pair=$(python3 "${GITHUB_WORKSPACE}/${TEST_RUNNER_DIR}/scripts/crusoe_pick_symmetric.py" \
    "${need}" "${probe_dir}") || return 1
  [ -n "${pair}" ] || return 1
  # A pair can be identical on paper (same GPUs/NICs/fingerprint) yet still be
  # unable to bring up mpirun across the two nodes -- e.g. no common route on the
  # mgmt interface, the exact failure that made every suite die at launch with
  # "an ORTE daemon has unexpectedly failed". Prove the real OpenMPI launch path
  # works on this pair before returning it; if it does not, blacklist the pair
  # for this run and signal the caller to re-probe another one.
  if ! mpi_smoke_pair "${pair}"; then
    echo "MPI bootstrap smoke failed on ${pair}; blacklisting and re-probing" >&2
    blacklist_nodes "${pair}"
    return 1
  fi
  printf '%s\n' "${pair}"
}

# Verify OpenMPI can actually bootstrap across a candidate pair using the same
# launch path the suites use (plm slurm via scripts/spur_srun.sh, OOB/BTL pinned
# to PROBE_OOB_IF). Submits scripts/crusoe_mpi_smoke.sh as a short sbatch job on
# exactly the pinned nodes and looks for its MPI_SMOKE_OK marker. Returns 0 (pair
# usable) or 1 (pair rejected). Set PROBE_MPI_SMOKE=0 to skip (e.g. debugging).
PROBE_MPI_SMOKE="${PROBE_MPI_SMOKE:-1}"
mpi_smoke_pair() {
  [ "${PROBE_MPI_SMOKE}" = "1" ] || return 0
  local pair="$1"
  local probe_dir="${RUNNER_TEMP}/probes"
  local out="${probe_dir}/mpi_smoke.out"
  local nnodes j waited st
  mkdir -p "${probe_dir}"
  rm -f "${out}"
  nnodes=$(printf '%s\n' "${pair}" | tr ',' '\n' | sed '/^$/d' | wc -l)
  j=$(spur_cmd sbatch --parsable \
    --job-name="ainic-mpi-smoke" \
    --output="${out}" \
    --error="${out}" \
    --partition="${SALLOC_PARTITION}" \
    ${ACCOUNT_ARG} \
    ${RESV_ARG} \
    --nodelist="${pair}" \
    --nodes="${nnodes}" \
    --ntasks="${nnodes}" \
    --ntasks-per-node=1 \
    --cpus-per-task=8 \
    --gpus-per-node=8 \
    --time=00:05:00 \
    --wrap "PROBE_OOB_IF=${PROBE_OOB_IF:-ens3} bash ${GITHUB_WORKSPACE}/${TEST_RUNNER_DIR}/scripts/crusoe_mpi_smoke.sh" 2>&1 \
    | tee /dev/stderr | grep -oE '^[0-9]+$' | tail -1 || true)
  if [ -z "${j}" ]; then
    echo "mpi smoke sbatch failed to submit for ${pair}; treating pair as unusable" >&2
    return 1
  fi
  echo "MPI smoke job ${j} on ${pair}; waiting for result" >&2
  waited=0
  while [ "${waited}" -lt 180 ]; do
    sleep 10
    waited=$((waited + 10))
    if grep -qE '^MPI_SMOKE_(OK|FAIL)' "${out}" 2>/dev/null; then
      break
    fi
    st=$(spur_cmd squeue -j "${j}" -h -o '%T' 2>/dev/null | tr -d ' ')
    case "${st}" in
      COMPLETED|FAILED|CANCELLED|NODE_FAIL|TIMEOUT|"")
        sleep 3
        break
        ;;
    esac
  done
  spur_cmd scancel "${j}" >/dev/null 2>&1 || true
  echo "=== mpi smoke (${pair}) ===" >&2
  { grep -E '^(MPI_SMOKE_|MPI smoke:|An ORTE daemon|route found)' "${out}" 2>/dev/null || tail -n 8 "${out}" 2>/dev/null; } >&2 || true
  grep -qE '^MPI_SMOKE_OK' "${out}" 2>/dev/null
}

# Nodes proven bad during THIS run: a probe/allocation that never reached
# RUNNING, or a live allocation killed by the scheduler (CANCELLED/NODE_FAIL)
# before its workload exited cleanly. They are folded into --exclude on every
# subsequent probe and submit so the check self-heals around flaky hardware --
# no hand-maintained list and no re-push needed when a node starts misbehaving
# mid-run.
BAD_NODES="${BAD_NODES:-}"
# File-backed source of truth for the blacklist. probe_pin captures
# probe_symmetric_pair via $(...), so any BAD_NODES that blacklist_nodes sets
# inside that command-substitution subshell would be lost to the parent loop
# (the cap counter and the exclude list would never grow, and the probe would
# spin until the job timeout). Persisting to a file makes the blacklist survive
# the subshell; the parent syncs BAD_NODES back from it via sync_bad_nodes.
BAD_NODES_FILE="${BAD_NODES_FILE:-${RUNNER_TEMP:-/tmp}/crusoe_bad_nodes}"
# Start each probe run from a clean slate: RUNNER_TEMP is reused across runs on
# the self-hosted runner, so a stale file would carry blacklisted nodes (and a
# false rejection count) into an unrelated run.
reset_bad_nodes() {
  BAD_NODES=""
  : > "${BAD_NODES_FILE}" 2>/dev/null || true
}
sync_bad_nodes() {
  [ -f "${BAD_NODES_FILE}" ] || return 0
  BAD_NODES=$(paste -sd, "${BAD_NODES_FILE}" 2>/dev/null)
}
rebuild_exclude_arg() {
  sync_bad_nodes
  local combined="${SALLOC_EXCLUDE:-}"
  if [ -n "${BAD_NODES}" ]; then
    combined="${combined:+${combined},}${BAD_NODES}"
  fi
  if [ -n "${combined}" ]; then
    EXCLUDE_ARG="--exclude=${combined}"
  else
    EXCLUDE_ARG=""
  fi
}
blacklist_nodes() {
  local nn
  for nn in $(printf '%s' "$1" | tr ',' ' '); do
    [ -n "${nn}" ] || continue
    case ",${BAD_NODES}," in
      *",${nn},"*) : ;;
      *) BAD_NODES="${BAD_NODES:+${BAD_NODES},}${nn}" ;;
    esac
    # Persist so the parent loop (which runs probe_symmetric_pair in a $()
    # subshell) still sees this rejection after the subshell exits.
    touch "${BAD_NODES_FILE}" 2>/dev/null || true
    grep -qxF "${nn}" "${BAD_NODES_FILE}" 2>/dev/null || printf '%s\n' "${nn}" >> "${BAD_NODES_FILE}"
  done
}
# Of the given nodes, echo the comma-separated subset that is actually in a
# fault state (down/drain/fail/...). A CANCELLED allocation on this shared
# cluster is usually preemption by a higher-priority job -- the node is healthy
# and must NOT be blacklisted, or the loop would burn through good hardware.
# Only genuinely broken nodes get excluded.
nodes_faulty() {
  local nn stt out=""
  for nn in $(printf '%s' "$1" | tr ',' ' '); do
    [ -n "${nn}" ] || continue
    stt=$(spur_cmd sinfo -n "${nn}" -h -o '%t' 2>/dev/null | head -1 | tr -d ' ')
    case "${stt}" in
      *down*|*drain*|*drng*|*fail*|*inval*|*unk*|*boot*|*na*)
        out="${out:+${out},}${nn}" ;;
    esac
  done
  printf '%s' "${out}"
}
