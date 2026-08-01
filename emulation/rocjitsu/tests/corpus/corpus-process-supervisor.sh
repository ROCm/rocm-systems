#!/usr/bin/env bash

# Copyright (c) 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

# Run a command in its own process group and ensure no descendants survive the
# command or this supervisor. This closes the gap where a timed-out launcher is
# reaped while one of the applications it launched ignores TERM.

set -uo pipefail

if (( $# == 0 )); then
  echo "Usage: $0 COMMAND [ARGUMENT ...]" >&2
  exit 2
fi

child_pgid=

stop_child_group() {
  local pgid="${child_pgid}"
  if [[ -z "${pgid}" ]]; then
    return
  fi

  kill -TERM -- "-${pgid}" 2>/dev/null || true
  for _ in {1..50}; do
    if ! kill -0 -- "-${pgid}" 2>/dev/null; then
      child_pgid=
      return
    fi
    sleep 0.1
  done
  kill -KILL -- "-${pgid}" 2>/dev/null || true
  child_pgid=
}

handle_signal() {
  local exit_status="$1"
  trap - HUP INT TERM
  stop_child_group
  exit "${exit_status}"
}

trap 'handle_signal 129' HUP
trap 'handle_signal 130' INT
trap 'handle_signal 143' TERM

setsid "$@" &
child_pgid=$!
wait "${child_pgid}"
exit_status=$?
stop_child_group
exit "${exit_status}"
