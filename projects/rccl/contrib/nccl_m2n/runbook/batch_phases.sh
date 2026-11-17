#!/bin/bash
# reshard_batch_bench_user_window: N tensors sequential vs concurrent across C comms.
# BATCH_RUNS="<label>:<np>:<src>:<dst>:<hf 0|1>" ...; per copy mode x --num-comms, 3 launches
# (first with --validate), ABBA order over num-comms, after a validate-only launch (round v) on
# non-aliasing tensors (the batch bench validates (global index) % 256; power-of-two dims alias).
D=$(dirname "$(readlink -f "$0")")
. $D/env.sh
R=$WORK/logs/bench/batch; mkdir -p $R
CSV=$R/results.csv
[[ -f $CSV ]] || echo "label,np,src,dst,copy,num_comms,num_tensors,round,validate,result,tensor,pattern,seq_ms,con_ms,speedup,eff,seq_gbs,con_gbs,log" > $CSV
SSHA="ssh -p $SSH_PORT -i $SSH_KEY -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -o LogLevel=ERROR"
NT=${NUM_TENSORS:-4}
COMMS=${COMMS:-1 2 4}
COPIES=${COPIES:-pack direct pipe}
BATCH_RUNS=${BATCH_RUNS:-$PHASES}
ST=0
for run in $BATCH_RUNS; do
  IFS=: read -r lab np s d hf <<<"$run"
  P=(); [[ $hf == 1 ]] && P=(--hostfile $HF)
  for cp in $COPIES; do
    for r in v 0 1 2; do
      list=($COMMS); [[ $r != v ]] && (( r % 2 )) && list=($(printf '%s\n' "${list[@]}" | tac))
      for nc in "${list[@]}"; do
        V=""; [[ $r == v || $r == 0 ]] && V="--validate"
        TD=1024,1024:4096,4096:16384,16384; [[ $r == v ]] && TD=1008,1008:4080,4080:16368,16368
        L=$R/${lab}_${cp}_c${nc}_r$r.log
        A="--algorithm direct --copy-algorithm $cp --src-mesh-dims 1,$s --dst-mesh-dims 1,$d \
           --tensor-dims $TD --src-shard-dims 0,0 --dst-shard-dims 0,1 \
           --num-comms $nc --num-tensors $NT --iterations 10 --warmup 2 $V"
        echo "=== $lab $cp c$nc r$r $(date -u +%FT%T) args: $A" > $L
        NCCL_RESHARD_ALGORITHM=DIRECT timeout 900 mpirun -np $np "${P[@]}" --map-by ppr:8:node --bind-to none --mca pml ob1 \
          --mca btl ^vader,openib --mca oob_tcp_if_include $NCCL_SOCKET_IFNAME --mca btl_tcp_if_include $NCCL_SOCKET_IFNAME \
          --mca plm_rsh_agent "$SSHA" -x PATH -x LD_LIBRARY_PATH -x NCCL_GIN_ENABLE -x NCCL_GIN_TYPE -x NCCL_CUMEM_ENABLE \
          -x NCCL_DMABUF_ENABLE -x HSA_NO_SCRATCH_RECLAIM -x NCCL_ENV_PLUGIN -x NCCL_SOCKET_IFNAME -x NCCL_DEBUG \
          -x NCCL_RESHARD_ALGORITHM $BIN/reshard_batch_bench_user_window $A >> $L 2>&1
        rc=$?; echo "=== EXIT $rc" >> $L
        if (( rc == 124 )); then res=HANG; pkill -9 -x reshard_batch_b
          [[ $hf == 1 ]] && for h in $(awk '{print $1}' $HF); do $SSHA $h pkill -9 -x reshard_batch_b; done
        elif (( rc == 0 )) && grep -q 'Benchmark completed successfully' $L && ! grep -q 'VALIDATION FAILED' $L; then res=$([[ -n $V ]] && echo PASS || echo OK)
        else res=FAIL_rc$rc; fi
        n=0
        while read -r t p sq cn sp ef sg cg; do
          echo "$lab,$np,$s,$d,$cp,$nc,$NT,$r,${V:+1},$res,${t//,/x},$p,$sq,$cn,$sp,${ef%\%},$sg,$cg,$(basename $L)" >> $CSV; n=1
        done < <(grep -E '^\[[0-9,]+\] ' $L)
        (( n )) || echo "$lab,$np,$s,$d,$cp,$nc,$NT,$r,${V:+1},$res,,,,,,,,,$(basename $L)" >> $CSV
        [[ $res == PASS || $res == OK ]] || ST=1
        echo "$(date -u +%T) $lab $cp c$nc r$r $res"
      done
    done
  done
done
echo "BATCH_DONE status=$ST"
exit $ST
