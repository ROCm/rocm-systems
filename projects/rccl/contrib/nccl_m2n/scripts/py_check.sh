#!/bin/bash
# nccl.m2n checks inside the container: python reshard_bench --validate (torch buffers, cross-dim, ~1 GB; 8000 avoids power-of-two aliasing in the bench data pattern)
# and the DTensor reshard check with the communicator from nccl.core (--comm rccl) and from
# ProcessGroupNCCL._comm_ptr() (--comm pg). Usage: py_check.sh [np]   (np 16 + HF=<hostfile> for 2 nodes)
D=$(dirname "$(readlink -f "$0")")
. $D/py_env.sh
NP=${1:-8}; H=$((NP / 2)); PLACE=(); [[ -n $HF ]] && PLACE=(--hostfile $HF)
L=$WORK/logs/py; mkdir -p $L; cd /tmp
ST=0
run() {  # label marker args...
  local lab=$1 mark=$2; shift 2
  timeout ${TMO:-900} mpirun -np $NP "${PLACE[@]}" "${PYMPI_ARGS[@]}" "$@" > $L/$lab.log 2>&1; local rc=$?
  echo "=== EXIT $rc" >> $L/$lab.log
  local res=FAIL_rc$rc; (( rc == 0 )) && grep -q "$mark" $L/$lab.log && res=PASS
  [[ $res == PASS ]] || ST=1
  echo "$lab np=$NP $res $(grep -A2 -- '--- Sources only' $L/$lab.log | grep -m1 Bandwidth)"
}
run bench_np$NP '\*\*\* VALIDATION PASSED \*\*\*' -m nccl.m2n.benchmarks.reshard_bench --validate --buffer-provider torch \
  --src-mesh-dims 1,$H --dst-mesh-dims 1,$H --src-shard-dim 0 --dst-shard-dim 1 --tensor-dims 8000,8000,16 \
  --algorithm direct --copy-algorithm pack --iterations 10 --warmup 2
for c in rccl pg; do
  run dtensor_${c}_np$NP 'RESULT PASS failures=0' $M2N_PY/tests/rocm/xdtensor_reshard_check.py --comm $c
done
exit $ST
