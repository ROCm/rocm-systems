#!/bin/bash
# PIPE in the basic matrix: PIPE keeps 32 persistent control slots per communicator, so run the Matrix
# cases one decade (10 cases x both API variants) per process, plus the non-Matrix tests once.
# Usage: pipe_chunks.sh <local|1node|2node>    (inside the allocation, like run_local.sh / run_mpi.sh)
D=$(dirname "$(readlink -f "$0")")
. $D/env.sh
LAY=$1; OUT=$WORK/logs/pipe_chunks/$LAY; mkdir -p $OUT
run() {  # label filter
  case $LAY in
    local) TMO=${TMO:-300} bash $D/run_local.sh pipe_chunks/$LAY/$1 --algorithm ring --copy-algorithm pipe --gtest_filter="$2" > /dev/null ;;
    1node) TMO=${TMO:-300} bash $D/run_mpi.sh pipe_chunks/$LAY/$1 8 1node --algorithm ring --copy-algorithm pipe --gtest_filter="$2" > /dev/null ;;
    2node) TMO=${TMO:-300} bash $D/run_mpi.sh pipe_chunks/$LAY/$1 16 2node --algorithm ring --copy-algorithm pipe --gtest_filter="$2" > /dev/null ;;
  esac
  pkill -9 -x basic_api_test_ 2>/dev/null
  [[ $LAY == 2node ]] && for h in $(awk '{print $1}' ${HF:-$WORK/hostfile}); do ssh $h pkill -9 -x basic_api_test_ 2>/dev/null; done
}
run nonmatrix '-Matrix/*'
for d in $(seq 0 29); do
  dd=$(printf '%03d' $d)
  run dec$dd "Matrix/*_case${dd}?_*"
done
printf "%-12s %-7s %-7s %-7s %s\n" chunk passed failed skipped exit > $OUT/summary.txt
for f in $OUT/*.log; do
  printf "%-12s %-7s %-7s %-7s %s\n" $(basename $f .log) \
    "$(grep -m1 -oE '^\[  PASSED  \] [0-9]+' $f | grep -oE '[0-9]+$' || echo 0)" \
    "$(grep -m1 -oE '^\[  FAILED  \] [0-9]+' $f | grep -oE '[0-9]+$' || echo 0)" \
    "$(grep -m1 -oE '^\[  SKIPPED \] [0-9]+' $f | grep -oE '[0-9]+$' || echo 0)" \
    "$(grep '^=== EXIT' $f | cut -d' ' -f3)" >> $OUT/summary.txt
done
awk 'NR>1{p+=$2;f+=$3;s+=$4; if($5!=0) bad=bad" "$1"("$5")"} END{print "TOTAL passed",p,"failed",f,"skipped",s,"nonzero-exit:",bad}' $OUT/summary.txt | tee -a $OUT/summary.txt
