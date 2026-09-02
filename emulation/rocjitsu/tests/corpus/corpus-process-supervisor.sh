#!/usr/bin/env bash

# Copyright (c) 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

# Run a command below a pinned session leader and ensure no process-group
# members survive the command or this supervisor.

set -uo pipefail
set +m

if (( $# == 0 )); then
  echo "Usage: $0 COMMAND [ARGUMENT ...]" >&2
  exit 2
fi

# Re-exec this script as the session and process-group leader. Keep parent death
# catchable by the real supervisor so its TERM trap can drain the process group.
# If a caller with job control made the first shell a process-group leader, let
# setsid fork and wait so the caller still observes the supervisor's exit status.
if [[ "${ROCJITSU_CORPUS_SUPERVISOR_SESSION-}" != 1 ]]; then
  current_pgid="$(ps -o pgid= -p "$$" 2>/dev/null | tr -d ' ')"
  if [[ "${current_pgid}" == "$$" ]]; then
    exec setsid --fork --wait setpriv --pdeathsig TERM -- \
      env ROCJITSU_CORPUS_SUPERVISOR_SESSION=1 "$0" "$@"
  fi
  exec setsid setpriv --pdeathsig TERM -- \
    env ROCJITSU_CORPUS_SUPERVISOR_SESSION=1 "$0" "$@"
fi

session_pgid="$(ps -o pgid= -p "$$" 2>/dev/null | tr -d ' ')"
if [[ ! "${session_pgid}" =~ ^[1-9][0-9]*$ || "${session_pgid}" != "$$" ]]; then
  echo "Corpus process supervisor did not become its process-group leader" >&2
  exit 2
fi

group_member_pids=()
collect_group_members() {
  group_member_pids=()
  local stat_path stat_record stat_suffix pid state process_pgid
  for stat_path in /proc/[1-9]*/stat; do
    IFS= read -r stat_record < "${stat_path}" 2>/dev/null || continue
    pid="${stat_record%% *}"
    stat_suffix="${stat_record##*) }"
    read -r state _ process_pgid _ <<< "${stat_suffix}"
    if [[ "${process_pgid}" == "${session_pgid}" && "${pid}" != "$$" &&
          "${state}" != Z ]]; then
      group_member_pids+=("${pid}")
    fi
  done
}

stop_child_group() {
  # The supervisor pins the PGID, so it cannot be recycled during cleanup.
  # Ignore our own TERM while delivering it to every other group member.
  trap '' HUP INT TERM
  kill -TERM -- "-${session_pgid}" 2>/dev/null || true
  for _ in {1..50}; do
    collect_group_members
    if (( ${#group_member_pids[@]} == 0 )); then
      return 0
    fi
    sleep 0.1
  done

  collect_group_members
  if (( ${#group_member_pids[@]} != 0 )); then
    kill -KILL -- "${group_member_pids[@]}" 2>/dev/null || true
  fi
  for _ in {1..50}; do
    collect_group_members
    if (( ${#group_member_pids[@]} == 0 )); then
      return 0
    fi
    sleep 0.1
  done

  echo "Corpus process cleanup left live process-group members: ${group_member_pids[*]}" >&2
  return 1
}

# shellcheck disable=SC2317 # Invoked indirectly by the signal traps below.
handle_signal() {
  local exit_status="$1"
  stop_child_group || true
  exit "${exit_status}"
}

trap 'handle_signal 129' HUP
trap 'handle_signal 130' INT
trap 'handle_signal 143' TERM

# Apply the death signal inside the new session. If the supervisor is killed,
# the immediate command is killed as well instead of becoming an orphan.
setpriv --pdeathsig KILL -- "$@" &
child_pid=$!
wait "${child_pid}"
exit_status=$?
cleanup_status=0
stop_child_group || cleanup_status=$?
if (( exit_status != 0 )); then
  exit "${exit_status}"
fi
exit "${cleanup_status}"
