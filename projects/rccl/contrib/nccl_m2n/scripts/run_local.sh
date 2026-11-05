#!/bin/bash
# Single-process, 8-GPU run (one pthread per rank). Run inside the container.
# Usage: run_local.sh <label> [test args...]
. "$(dirname "$0")/env.sh"
L=$1; shift
LOG=$WORK/logs/$L.log
{ echo "=== $L host=$(hostname -s) $(date -u) args: $*"
  timeout ${TMO:-3600} $BIN/basic_api_test_local "$@" --gtest_color=no
  echo "=== EXIT $? $(date -u)"; } > $LOG 2>&1
grep -E "^\[  PASSED|^\[  FAILED  \] [0-9]|^\[  SKIPPED \] [0-9]|^=== EXIT" $LOG
exit $(grep '^=== EXIT' $LOG | cut -d' ' -f3)
