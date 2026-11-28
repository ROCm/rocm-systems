#!/bin/bash
# Phases (space-separated) run inside the container on the first node:
#   m:<label>:<np>:<src>:<dst>:<shapes>:<configs>:<hf 0|1>[:<sizes,>]   reshard_bench matrix
#   auto:<label>:<np>:<src>:<dst>:<hf>                         one ALGORITHM=AUTO run (logged choice)
#   a2a:<label>:<np>:<hf>                                      rccl-tests alltoall_perf reference
D=$(dirname "$(readlink -f "$0")")
. $D/../scripts/env.sh
R=$WORK/logs/bench; mkdir -p $R
mpi() { local np=$1 hf=$2; shift 2; local P=(); [[ $hf == 1 ]] && P=(--hostfile $HF)
  "${M2N_MPIRUN[@]}" -np $np "${P[@]}" "$@"; }
MPIDEF="$(declare -p M2N_MPIRUN); $(declare -f mpi)"   # for the timeout'd bash -c below
ST=0
for ph in $PHASES; do
  IFS=: read -r kind a b c d e f g h <<<"$ph"
  echo ">>> $(date -u +%T) $ph"
  case $kind in
    m) hf=""; [[ $g == 1 ]] && hf=$HF
       SIZES="${h:-${SIZES:-}}" bash $D/bench_matrix.sh $a $b $c $d $e $f $hf > $R/$a.progress 2>&1 || ST=1; tail -1 $R/$a.progress ;;
    auto) mkdir -p $R/$a; L=$R/$a/auto_cross01_64M.log
       NCCL_RESHARD_LOG_LEVEL=INFO \
         timeout 300 bash -c "$MPIDEF; HF=$HF; mpi $b $e $BIN/reshard_bench --algorithm auto --src-mesh-dims 1,$c --dst-mesh-dims 1,$d \
         --src-shard-dim 0 --dst-shard-dim 1 --tensor-dims 2032,2032,17 --iterations 10 --warmup 2 --validate" > $L 2>&1
       rc=$?; echo "=== EXIT $rc" >> $L; res=FAIL_rc$rc
       (( rc == 0 )) && grep -qx '\*\*\* VALIDATION PASSED \*\*\*' $L && res=PASS; [[ $res == PASS ]] || ST=1
       echo "auto $res $(grep -m1 'Algorithm setting' $L) | $(grep -m1 -oE 'copy_algo=[0-9]+' $L) | $(grep -m1 'Bandwidth' $L)" ;;
    a2a) mkdir -p $R/$a; L=$R/$a/alltoall.log
       timeout 1200 bash -c "$MPIDEF; HF=$HF; mpi $b $c $WORK/rccl-tests/build/alltoall_perf -b 64K -e 4G -f 4 -g 1 -n 20 -w 5" > $L 2>&1
       rc=$?; echo "=== EXIT $rc" >> $L; (( rc == 0 )) || ST=1; echo "a2a rc=$rc"; grep -E "^\s+[0-9]+ " $L | tail -9 ;;
  esac
done
echo "PHASES_DONE status=$ST"
exit $ST
