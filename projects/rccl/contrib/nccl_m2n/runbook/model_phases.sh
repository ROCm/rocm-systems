#!/bin/bash
# reshard_model_bench runs: MODEL_RUNS="<label>:<np>:<hf 0|1>:<system cfg path>[:<model cfg path>]" ..., ALGO modes auto + direct/pack.
D=$(dirname "$(readlink -f "$0")")
. $D/../scripts/env.sh
R=$WORK/logs/bench/model; mkdir -p $R
MC=$SRC/projects/rccl/contrib/nccl_m2n/benchmarks/configs/model_configs/qwen3-32b.model.json
ST=0
for run in $MODEL_RUNS; do
  IFS=: read -r lab np hf sc mc <<<"$run"
  MCR=${mc:-$MC}
  P=(); [[ $hf == 1 ]] && P=(--hostfile $HF)
  for mode in auto direct_pack; do
    A="--algorithm auto"; [[ $mode == direct_pack ]] && A="--algorithm direct --copy-algorithm pack"
    for r in 0 1 2 3; do
      V=""; (( r == 0 )) && V="--validate --validate-iterations 1"
      L=$R/${lab}_${mode}_r$r.log
      echo "=== $lab $mode r$r $(date -u +%FT%T) cfg=$sc args: $A $V" > $L
      timeout 900 "${M2N_MPIRUN[@]}" -np $np "${P[@]}" $BIN/reshard_model_bench --model-config $MCR --system-config $sc --gpus-per-node 8 --iterations 10 --warmup 2 $A $V >> $L 2>&1
      rc=$?; echo "=== EXIT $rc" >> $L
      if (( rc == 124 )); then res=HANG; elif (( rc != 0 )); then res=FAIL_rc$rc
      elif [[ -n $V ]]; then grep -qi 'validation passed' $L && ! grep -qi 'validation failed' $L && res=PASS || res=FAIL_val
      else res=OK; fi
      [[ $res == PASS || $res == OK ]] || ST=1
      echo "$(date -u +%T) $lab $mode r$r $res $(grep -iE -m2 'validation (passed|failed)|total' $L | tr '\n' ' ' | cut -c1-200)"
      (( rc == 124 )) && { pkill -9 -x reshard_model_b; [[ $hf == 1 ]] && for h in $(awk '{print $1}' $HF); do $M2N_SSH $h pkill -9 -x reshard_model_b; done; break; }
    done
  done
done
echo "MODEL_DONE status=$ST"
exit $ST
