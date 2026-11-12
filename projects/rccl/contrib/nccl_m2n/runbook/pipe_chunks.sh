#!/bin/bash
# PIPE in the basic matrix: PIPE keeps 32 persistent control slots per communicator, so run the Matrix
# cases one decade (10 cases x both API variants) per process, plus the non-Matrix tests once.
# Cases in KNOWN_HANG run in their own process and are excluded from their decade; a decade that still
# times out is rerun one case per process, so a hang only loses that case.
# Usage: pipe_chunks.sh <local|1node|2node>    (inside the allocation, like run_local.sh / run_mpi.sh)
D=$(dirname "$(readlink -f "$0")")
. $D/env.sh
LAY=$1; RUN=${RUN_ID:-$(date -u +%Y%m%dT%H%M%S)-$$}   # per invocation, so stale logs are not summarized
OUT=$WORK/logs/pipe_chunks/$LAY/$RUN; mkdir -p $OUT
# Single-node uneven_ratio 240x240 cases that hang (see "PIPE in the basic matrix" in RUNBOOK.md).
[[ $LAY == 2node ]] && KNOWN_HANG=${KNOWN_HANG-} || KNOWN_HANG=${KNOWN_HANG-0066 0078 0081 0093}
run() {  # label filter; prints the exit code
  case $LAY in
    local) TMO=${TMO:-300} bash $D/run_local.sh pipe_chunks/$LAY/$RUN/$1 --algorithm ring --copy-algorithm pipe --gtest_filter="$2" > /dev/null ;;
    1node) TMO=${TMO:-300} bash $D/run_mpi.sh pipe_chunks/$LAY/$RUN/$1 8 1node --algorithm ring --copy-algorithm pipe --gtest_filter="$2" > /dev/null ;;
    2node) TMO=${TMO:-300} bash $D/run_mpi.sh pipe_chunks/$LAY/$RUN/$1 16 2node --algorithm ring --copy-algorithm pipe --gtest_filter="$2" > /dev/null ;;
  esac
  pkill -9 -x basic_api_test_ 2>/dev/null
  [[ $LAY == 2node ]] && for h in $(awk '{print $1}' ${HF:-$WORK/hostfile}); do ssh $h pkill -9 -x basic_api_test_ 2>/dev/null; done
  grep '^=== EXIT' $OUT/$1.log | cut -d' ' -f3
}
run nonmatrix '-Matrix/*' > /dev/null
for c in $KNOWN_HANG; do run case$c "Matrix/*_case${c}_*" > /dev/null; done
for d in $(seq 0 29); do
  dd=$(printf '%03d' $d)
  ex=""; for c in $KNOWN_HANG; do [[ $c == ${dd}? ]] && ex+=":Matrix/*_case${c}_*"; done
  rc=$(run dec$dd "Matrix/*_case${dd}?_*${ex:+-${ex#:}}")
  if [[ $rc == 124 ]]; then
    mv $OUT/dec$dd.log $OUT/dec$dd.timeout
    for k in $(seq 0 9); do [[ " $KNOWN_HANG " == *" $dd$k "* ]] || run case$dd$k "Matrix/*_case${dd}${k}_*" > /dev/null; done
  fi
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
# Fail on failed tests or a missing/nonzero exit; only the KNOWN_HANG cases may time out (124).
ST=0
for f in $OUT/*.log; do
  b=$(basename $f .log); e=$(grep '^=== EXIT' $f | cut -d' ' -f3)
  nf=$(grep -m1 -oE '^\[  FAILED  \] [0-9]+' $f | grep -oE '[0-9]+$')
  if [[ -z $e ]]; then ST=1
  elif [[ $e == 124 && $b == case* && " $KNOWN_HANG " == *" ${b#case} "* ]]; then :
  elif [[ $e != 0 || -n $nf ]]; then ST=1; fi
done
exit $ST
