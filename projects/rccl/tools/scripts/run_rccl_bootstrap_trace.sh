#!/usr/bin/env bash
# Run rccl-tests over the socket OOB bootstrap with the new TCP_INFO trace
# counters enabled, and collect per-rank binary dumps for offline analysis with
# merge_bootstrap_trace.py --net-csv.
#
# Launch model: OpenMPI mpirun (this cluster's Slurm pmix launcher times out and
# its OpenMPI lacks pmi2, so we grab a node set via srun then use mpirun).
#
# Usage:
#   run_rccl_bootstrap_trace.sh <nodes> <ranks_per_node> <bidir:0|1> <iters> <node_csv>
# Example (single node, healthy GPU node):
#   run_rccl_bootstrap_trace.sh 1 8 1 10 dell300x-ccs-aus-k13-03.cs-aus.dcgpu
set -uo pipefail

NODES="${1:-1}"
RPN="${2:-8}"
BIDIR="${3:-1}"
ITERS="${4:-10}"
NODELIST="${5:-}"            # optional explicit -w node list (comma-separated)

ROOT=/home/ilkosare/development/rocm-systems/projects
RCCL_LIB="$ROOT/rccl/build_instr_verify"
TEST="$ROOT/rccl-tests/build_bench_gfx942_mpi/all_reduce_perf"
TS=$(date +%Y%m%d_%H%M%S)
OUT="$HOME/.bnp_rccl/${TS}_n${NODES}x${RPN}_bidir${BIDIR}"
mkdir -p "$OUT"
NP=$((NODES * RPN))

wflag=""
[ -n "$NODELIST" ] && wflag="-w $NODELIST"

echo "[run] out=$OUT np=$NP nodes=$NODES rpn=$RPN bidir=$BIDIR iters=$ITERS"

for it in $(seq 1 "$ITERS"); do
  tdir="$OUT/iter_$(printf '%03d' "$it")"
  mkdir -p "$tdir"
  # One fresh ncclCommInitRank == one bootstrap per invocation.
  timeout 180 srun $wflag -N"$NODES" -n1 -c64 --gres=gpu:"$RPN" -t4 \
    bash -c '
      unset HIP_VISIBLE_DEVICES
      cd '"$ROOT"'
      mpirun --oversubscribe -np '"$NP"' \
        --map-by ppr:'"$RPN"':node ${HOSTFILE:+--hostfile $HOSTFILE} \
        -x LD_LIBRARY_PATH='"$RCCL_LIB"' \
        -x NCCL_SOCKET_IFNAME=eno8303 \
        -x NCCL_OOB_NET_ENABLE=0 \
        -x NCCL_BOOTSTRAP_BIDIR_ALLGATHER='"$BIDIR"' \
        -x NCCL_BOOTSTRAP_TRACE=1 \
        -x NCCL_BOOTSTRAP_TRACE_DIR='"$tdir"' \
        '"$TEST"' -b 1K -e 1K -f 2 -g 1 -n 3 -w 1
    ' > "$tdir/run.log" 2>&1
  rc=$?
  nbin=$(ls "$tdir"/*.bin 2>/dev/null | wc -l)
  echo "[run] iter $it rc=$rc bins=$nbin"
done

echo "[run] done. Analyze with:"
echo "  python3 $ROOT/rccl/tools/scripts/merge_bootstrap_trace.py $OUT/iter_* --net-csv $OUT/net.csv --summary"
echo "$OUT"
