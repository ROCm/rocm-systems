#!/bin/bash
# MPI run, 8 ranks per node. 1node runs on the current node; 2node uses ${HF:-$WORK/hostfile}.
# Runs basic_api_test_mpi unless MPI_BIN names another binary. Run inside the container on the first node.
# Usage: run_mpi.sh <label> <np> <1node|2node> [args...]
. "$(dirname "$0")/env.sh"
L=$1; NP=$2; LAY=$3; shift 3
LOG=$WORK/logs/$L.log
PLACE=()
[[ $LAY == 2node ]] && PLACE=(--hostfile ${HF:-$WORK/hostfile})
EXE=${MPI_BIN:-$BIN/basic_api_test_mpi}; X=(); [[ -z $MPI_BIN ]] && X=(--gtest_color=no)
{ echo "=== $L np=$NP $LAY host=$(hostname -s) $(date -u) args: $*"
  timeout ${TMO:-3600} "${M2N_MPIRUN[@]}" -np $NP "${PLACE[@]}" $EXE "$@" "${X[@]}"
  echo "=== EXIT $? $(date -u)"; } > $LOG 2>&1
grep -iE "^\[  PASSED|^\[  FAILED  \] [0-9]|^\[  SKIPPED \] [0-9]|validation (passed|failed)|^=== EXIT" $LOG | sort | uniq -c
exit $(grep '^=== EXIT' $LOG | cut -d' ' -f3)
