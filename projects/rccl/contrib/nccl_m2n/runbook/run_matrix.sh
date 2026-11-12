#!/bin/bash
# Full validated matrix: RING (PACK copy) and DIRECT, single-process / MPI 1 node / MPI 2 nodes.
# Run inside the container on the first node of the allocation.
# Usage: run_matrix.sh [local|1node|2node]...   (default: all three)
D=$(dirname "$(readlink -f "$0")")
. $D/env.sh
MODES=${*:-local 1node 2node}
RUNS=()
for m in $MODES; do
  case $m in
    local) RUNS+=("local_ring:local:--algorithm ring" "local_direct:local:--algorithm direct") ;;
    1node) RUNS+=("mpi1_ring:1node:--algorithm ring" "mpi1_direct:1node:--algorithm direct" "mpi1_all:1node:--algorithm all") ;;
    2node) RUNS+=("mpi2_ring:2node:--algorithm ring" "mpi2_direct:2node:--algorithm direct" "mpi2_all:2node:--algorithm all") ;;
  esac
done
ST=0
for r in "${RUNS[@]}"; do
  L=${r%%:*}; rest=${r#*:}; LAY=${rest%%:*}; ARGS=${rest#*:}
  echo ">>> $L ($LAY) $ARGS"
  case $LAY in
    local) bash $D/run_local.sh $L $ARGS >/dev/null || ST=1 ;;
    1node) bash $D/run_mpi.sh $L 8 1node $ARGS >/dev/null || ST=1 ;;
    2node) bash $D/run_mpi.sh $L 16 2node $ARGS >/dev/null || ST=1 ;;
  esac
done
printf "\n%-14s %-7s %-7s %-7s %s\n" run passed failed skipped exit
for r in "${RUNS[@]}"; do
  L=${r%%:*}; LOG=$WORK/logs/$L.log
  printf "%-14s %-7s %-7s %-7s %s\n" $L \
    "$(grep -m1 -oE '^\[  PASSED  \] [0-9]+' $LOG | grep -oE '[0-9]+$' || echo 0)" \
    "$(grep -m1 -oE '^\[  FAILED  \] [0-9]+' $LOG | grep -oE '[0-9]+$' || echo 0)" \
    "$(grep -m1 -oE '^\[  SKIPPED \] [0-9]+' $LOG | grep -oE '[0-9]+$' || echo 0)" \
    "$(grep '^=== EXIT' $LOG | cut -d' ' -f3)"
done
exit $ST
