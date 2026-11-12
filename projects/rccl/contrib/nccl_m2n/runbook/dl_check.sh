#!/bin/bash
# Copy-mode / deadlock check: reshard_bench --validate, 1 node x 8 ranks (4 -> 4), every copy mode,
# 4 shapes x 3 sizes, per-case timeout so a hang is detected (HANG + rocgdb backtrace of 2 ranks).
# Run inside the container after `build.sh bench`.
# Usage: bash dl_check.sh            (MODES="pack direct pipe default auto" by default)
#   pack|direct|pipe : NCCL_RESHARD_ALGORITHM=DIRECT, --copy-algorithm <mode>
#   default          : no --algorithm/--copy-algorithm (bench default: ALGORITHM=AUTO, PACK)
#   auto             : --algorithm auto (logs the resolved path with NCCL_RESHARD_LOG_LEVEL=INFO)
D=$(dirname "$(readlink -f "$0")")
. $D/env.sh
OUT=$WORK/logs/dl
mkdir -p $OUT
echo "HEAD=$(git -C $SRC rev-parse --short HEAD) host=$(hostname -s)"
MPI="mpirun -np 8 --map-by ppr:8:node --bind-to none --mca pml ob1 --mca btl ^vader,openib \
 --mca oob_tcp_if_include $NCCL_SOCKET_IFNAME --mca btl_tcp_if_include $NCCL_SOCKET_IFNAME \
 -x PATH -x LD_LIBRARY_PATH -x NCCL_GIN_ENABLE -x NCCL_GIN_TYPE -x NCCL_CUMEM_ENABLE -x NCCL_DMABUF_ENABLE \
 -x HSA_NO_SCRATCH_RECLAIM -x NCCL_ENV_PLUGIN -x NCCL_SOCKET_IFNAME -x NCCL_DEBUG -x NCCL_RESHARD_ALGORITHM \
 -x NCCL_RESHARD_LOG_LEVEL"

declare -A SHAPE=(
  [same00]="--src-mesh-dims 1,4 --dst-mesh-dims 1,4 --src-shard-dim 0 --dst-shard-dim 0"
  [cross01]="--src-mesh-dims 1,4 --dst-mesh-dims 1,4 --src-shard-dim 0 --dst-shard-dim 1"
  [cross10]="--src-mesh-dims 1,4 --dst-mesh-dims 1,4 --src-shard-dim 1 --dst-shard-dim 0"
  [mesh22]="--src-mesh-dims 1,4 --dst-mesh-dims 2,2 --src-shard-dim 0 --dst-shard-dim 1"
)
# reshard_bench validates (global index) % 256: with power-of-two dims the column/depth terms and the shard
# offsets vanish modulo 256 and source shards look identical. Odd local shard sizes keep them distinct.
declare -A SIZE=( [2M]="124,124,129 --iterations 20 --warmup 3" [64M]="508,508,257 --iterations 10 --warmup 2" [1G]="1020,1020,1021 --iterations 5 --warmup 1" )
declare -A TMO=( [2M]=180 [64M]=180 [1G]=300 )
MODES=${MODES:-pack direct pipe default auto}

printf "%-8s %-8s %-4s %-5s %-10s %s\n" mode shape size res "GB/s/rank" "total GB/s" > $OUT/summary.txt
NBAD=0
for m in $MODES; do
  unset NCCL_RESHARD_ALGORITHM NCCL_RESHARD_LOG_LEVEL
  case $m in
    default) ARGS="" ;;
    auto) ARGS="--algorithm auto"; export NCCL_RESHARD_LOG_LEVEL=INFO ;;
    *) ARGS="--algorithm direct --copy-algorithm $m"; export NCCL_RESHARD_ALGORITHM=DIRECT ;;
  esac
  for s in same00 cross01 cross10 mesh22; do
    for z in 2M 64M 1G; do
      L=$OUT/${m}_${s}_${z}.log
      read -r dims rest <<<"${SIZE[$z]}"
      echo "=== $m $s $z $(date -u +%T) args: $ARGS ${SHAPE[$s]} --tensor-dims $dims $rest" > $L
      $MPI $BIN/reshard_bench $ARGS ${SHAPE[$s]} --tensor-dims $dims $rest --validate >> $L 2>&1 &
      mp=$!
      t=0; while kill -0 $mp 2>/dev/null && (( t < ${TMO[$z]} )); do sleep 2; t=$((t+2)); done
      if kill -0 $mp 2>/dev/null; then
        res=HANG
        echo "=== HANG after ${TMO[$z]}s" >> $L
        for p in $(pgrep -x reshard_bench | sed -n '1p;5p'); do
          echo "--- pid $p $(tr '\0' '\n' < /proc/$p/environ | grep -m1 OMPI_COMM_WORLD_RANK=)" >> $L.bt
          timeout 60 rocgdb -p $p -batch -ex "set pagination off" -ex "thread apply all bt 12" >> $L.bt 2>&1
        done
        pkill -x reshard_bench; sleep 3; pkill -9 -x reshard_bench; kill $mp 2>/dev/null; wait $mp 2>/dev/null
      else
        wait $mp; rc=$?; echo "=== EXIT $rc" >> $L
        # The marker is printed before the timed iterations and teardown, so the exit code must be 0 too.
        if (( rc == 0 )) && grep -qx '\*\*\* VALIDATION PASSED \*\*\*' $L; then res=PASS; else res="FAIL($rc)"; fi
      fi
      bw=$(grep -A2 -- '--- Sources only' $L | grep -m1 'Bandwidth' | grep -oE 'Avg=[0-9.]+' | cut -d= -f2)  # per source rank
      tot=$(grep -m1 'Total data throughput' $L | grep -oE '[0-9.]+ GB/s' | cut -d' ' -f1)
      printf "%-8s %-8s %-4s %-5s %-10s %s\n" $m $s $z "$res" "${bw:--}" "${tot:--}" | tee -a $OUT/summary.txt
      [[ $res == PASS ]] || NBAD=$((NBAD+1))
      [[ $res == HANG ]] && break
    done
  done
done
echo "DL_DONE failed=$NBAD"
(( NBAD == 0 ))
