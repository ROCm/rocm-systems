#!/bin/bash
# reshard_bench validation matrix. Run inside the container on the first node.
# Usage: bench_matrix.sh <label> <np> <src_ranks> <dst_ranks> <shapes> <configs> [hostfile]
#   shapes : comma list of same00,cross01,cross10,mesh2 (+ rcross01,rmesh for > 16 ranks per side)
#   configs: comma list of D_PACK,D_DIRECT,D_PIPE,R_PACK
# Env: SIZES (default 64K..4G x4, space or comma separated), ROUNDS (4 launches per point, ABBA order).
# Each point first runs a validate-only launch (round v) on non-aliasing dims, then the aligned perf rounds.
D=$(dirname "$(readlink -f "$0")")
. $D/env.sh
LAB=$1; NP=$2; S=$3; D=$4; SHAPES=${5//,/ }; CONFIGS=${6//,/ }; HF=$7
OUT=$WORK/logs/bench/$LAB
mkdir -p $OUT
CSV=$OUT/results.csv
[[ -f $CSV ]] || echo "label,np,src,dst,shape,size,config,round,validate,result,bw_rank_avg,bw_rank_min,total_gbs,time_ms_max,log" > $CSV
SIZES=${SIZES:-64K 256K 1M 4M 16M 64M 256M 1G 4G}; SIZES=${SIZES//,/ }
ROUNDS=${ROUNDS:-4}
# Per-invocation tag: reruns append to results.csv, so each launch needs its own log.
RUN=${RUN_ID:-$(date -u +%Y%m%dT%H%M%S)-$$}
NBAD=0
PLACE=(); [[ -n $HF ]] && PLACE=(--hostfile $HF)
MPI=(mpirun -np $NP "${PLACE[@]}" --map-by ppr:8:node --bind-to none --mca pml ob1 --mca btl ^vader,openib
  --mca oob_tcp_if_include $NCCL_SOCKET_IFNAME --mca btl_tcp_if_include $NCCL_SOCKET_IFNAME
  --mca plm_rsh_agent "ssh -p $SSH_PORT -i $SSH_KEY -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -o LogLevel=ERROR"
  -x PATH -x LD_LIBRARY_PATH -x NCCL_GIN_ENABLE -x NCCL_GIN_TYPE -x NCCL_CUMEM_ENABLE -x NCCL_DMABUF_ENABLE
  -x HSA_NO_SCRATCH_RECLAIM -x NCCL_ENV_PLUGIN -x NCCL_SOCKET_IFNAME -x NCCL_DEBUG -x NCCL_RESHARD_ALGORITHM
  -x NCCL_RESHARD_LOG_LEVEL)

dims() { case $1 in 64K) echo 64,64,16;; 256K) echo 128,128,16;; 1M) echo 256,256,16;; 4M) echo 512,512,16;;
  16M) echo 1024,1024,16;; 64M) echo 2048,2048,16;; 256M) echo 4096,4096,16;; 1G) echo 8192,8192,16;;
  4G) echo 16384,16384,16;; 16G) echo 32768,32768,16;; esac; }
# reshard_bench validates (g0 + g1*G1 + g2*G1*G2) % 256 (G = global dims); with the power-of-two dims above
# shards and columns alias. Per axis, dim = M*m with M = lcm of the shard counts on that axis (both sides) and m odd:
# every shard count divides it, dim0 offsets stay distinct modulo 256, and dim1 offsets (scaled by G1) stay distinct
# because at most 16 shards lie on dim1 in these shapes. Depth 17 keeps the column/depth terms nonzero.
gcd() { local a=$1 b=$2 t; while (( b )); do t=$((a % b)); a=$b; b=$t; done; echo $a; }
lcm() { echo $(( $1 / $(gcd $1 $2) * $2 )); }
vdims() {  # <aligned dims> <shape>
  local n=${1%%,*} M=(1 1 1) a i m d out=()
  a=($(shape_args $2))
  for i in 1 3; do   # shape_args: --src-mesh-dims r,s --dst-mesh-dims r,s --src-shard-dim k --dst-shard-dim k
    d=${a[i+4]}; M[d]=$(lcm ${M[d]} ${a[i]#*,})
  done
  (( M[1] > 16 )) && echo "WARNING: $2: ${M[1]} shards on dim1; the % 256 pattern can tell at most 16 apart" >&2
  for d in 0 1; do m=$(( n / M[d] )); (( m % 2 )) || m=$(( m > 1 ? m - 1 : 1 )); out+=($(( M[d] * m ))); done
  (( M[2] == 1 )) && out+=(17) || out+=($(( M[2] * 17 )))
  local IFS=,; echo "${out[*]}"
}
iters() { case $1 in 64K|256K|1M) echo "--iterations 50 --warmup 5";; 4M|16M|64M) echo "--iterations 20 --warmup 3";;
  256M|1G) echo "--iterations 10 --warmup 2";; *) echo "--iterations 5 --warmup 1";; esac; }
tmo() { case $1 in 4G|16G) echo 600;; 1G) echo 400;; *) echo 240;; esac; }
shape_args() { case $1 in
  same00)  echo "--src-mesh-dims 1,$S --dst-mesh-dims 1,$D --src-shard-dim 0 --dst-shard-dim 0";;
  cross01) echo "--src-mesh-dims 1,$S --dst-mesh-dims 1,$D --src-shard-dim 0 --dst-shard-dim 1";;
  cross10) echo "--src-mesh-dims 1,$S --dst-mesh-dims 1,$D --src-shard-dim 1 --dst-shard-dim 0";;
  mesh2)   echo "--src-mesh-dims 1,$S --dst-mesh-dims 2,$((D/2)) --src-shard-dim 0 --dst-shard-dim 1";;
  # >16 shards per side exceeds MAX_SOURCES=16 (reshard_limits.h) for cross-dim; replicate the meshes instead.
  rcross01) echo "--src-mesh-dims $((S/16)),16 --dst-mesh-dims $((D/16)),16 --src-shard-dim 0 --dst-shard-dim 1";;
  rmesh)    echo "--src-mesh-dims $((S/16)),16 --dst-mesh-dims $((D/8)),8 --src-shard-dim 0 --dst-shard-dim 1";;
  esac; }
cfg_args() { case $1 in
  D_PACK) echo "--algorithm direct --copy-algorithm pack";; D_DIRECT) echo "--algorithm direct --copy-algorithm direct";;
  D_PIPE) echo "--algorithm direct --copy-algorithm pipe";; R_PACK) echo "--algorithm ring --copy-algorithm pack";;
  esac; }

cleanup_procs() {
  pkill -x reshard_bench; [[ -n $HF ]] && for h in $(awk '{print $1}' $HF); do
    ssh -p $SSH_PORT -i $SSH_KEY -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -o LogLevel=ERROR $h pkill -x reshard_bench; done
  sleep 3; pkill -9 -x reshard_bench; pkill -9 -x mpirun; sleep 2
}

declare -A DEAD
for shp in $SHAPES; do
  for z in $SIZES; do
    for r in v $(seq 0 $((ROUNDS-1))); do
      list=($CONFIGS); [[ $r != v ]] && (( r % 2 )) && list=($(printf '%s\n' "${list[@]}" | tac))
      for c in "${list[@]}"; do
        key=$shp/$z/$c; [[ -n ${DEAD[$key]} ]] && continue
        V=""; [[ $r == v || $r == 0 ]] && V="--validate"
        TD=$(dims $z); [[ $r == v ]] && TD=$(vdims $TD $shp)
        L=$OUT/${shp}_${z}_${c}_r${r}_$RUN.log
        A="$(cfg_args $c) $(shape_args $shp) --tensor-dims $TD $(iters $z) $V"
        echo "=== $LAB $key r$r $(date -u +%FT%T) args: $A" > $L
        "${MPI[@]}" $BIN/reshard_bench $A >> $L 2>&1 &
        mp=$!; t=0; T=$(tmo $z)
        while kill -0 $mp 2>/dev/null && (( t < T )); do sleep 1; t=$((t+1)); done
        if kill -0 $mp 2>/dev/null; then
          res=HANG; echo "=== HANG after ${T}s" >> $L
          pgrep -x reshard_bench | sed -n '1p' | while read p; do
            timeout 60 rocgdb -p $p -batch -ex "set pagination off" -ex "thread apply all bt 10" > $L.bt 2>&1; done
          cleanup_procs; wait $mp 2>/dev/null; DEAD[$key]=1
        else
          wait $mp; rc=$?; echo "=== EXIT $rc" >> $L
          # Markers are printed before teardown: require a zero exit as well.
          if (( rc != 0 )); then res=FAIL_rc$rc
          elif [[ -n $V ]]; then grep -qx '\*\*\* VALIDATION PASSED \*\*\*' $L && res=PASS || res=FAIL_rc$rc
          else grep -q 'Benchmark completed successfully' $L && res=OK || res=FAIL_rc$rc; fi
          [[ $res == FAIL* ]] && { DEAD[$key]=1; cleanup_procs; }
        fi
        # per source rank; Overall mixes src/dst bytes
        bw=$(grep -A2 -- '--- Sources only' $L | grep -m1 'Bandwidth'); avg=$(grep -oE 'Avg=[0-9.]+' <<<"$bw" | cut -d= -f2); mn=$(grep -oE 'Min=[0-9.]+' <<<"$bw" | cut -d= -f2)
        tot=$(grep -m1 'Total data throughput' $L | grep -oE '[0-9.]+ GB/s' | cut -d' ' -f1)
        tm=$(grep -m1 'Time per iteration' $L | grep -oE 'Max=[0-9.]+' | cut -d= -f2)
        [[ $res == PASS || $res == OK ]] || NBAD=$((NBAD+1))
        echo "$LAB,$NP,$S,$D,$shp,$z,$c,$r,${V:+1},$res,$avg,$mn,$tot,$tm,$(basename $L)" >> $CSV
        echo "$(date -u +%T) $key r$r $res avg=$avg tot=$tot"
      done
    done
  done
done
echo "MATRIX_DONE $LAB failed=$NBAD"
(( NBAD == 0 ))
