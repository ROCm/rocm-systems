#!/usr/bin/env bash
# Compile and run bootstrap_net_probe across a Slurm allocation, sweeping
# TCP_NODELAY / TCP_QUICKACK so we can attribute bootstrap-ring variance to
# transport behaviour (delayed-ACK, retransmits, RTT). Results land as
# per-rank CSVs under $OUT/<tag>/.
#
# Usage:
#   tools/scripts/run_bootstrap_net_probe.sh [NODES] [TASKS_PER_NODE] [ITERS]
#
# Env overrides: PROBE_IFACE, PROBE_MSG_BYTES, OUTROOT
set -euo pipefail

# Default to 1 rank/node: this exercises the pure inter-node TCP ring, which is
# the path RCCL's slow bootstrap phases (recv_from_root / forward_connect /
# ring_step) live on. Multi-rank-per-node mixes intra-node loopback links whose
# rendezvous/accept timing is a known harness limitation on this cluster.
NODES="${1:-8}"
TPN="${2:-1}"
ITERS="${3:-2000}"
IFACE="${PROBE_IFACE:-eno8303}"
MSG="${PROBE_MSG_BYTES:-480}"
HERE="$(cd "$(dirname "$0")" && pwd)"
SRC="$HERE/bootstrap_net_probe.c"
BIN="$HERE/bootstrap_net_probe"
OUTROOT="${OUTROOT:-$HOME/.bnp_runs/$(date +%Y%m%d_%H%M%S)}"

echo "[run] compiling $SRC -> $BIN"
cc -O2 -Wall -pthread -o "$BIN" "$SRC"

mkdir -p "$OUTROOT"
echo "[run] output root: $OUTROOT"
echo "[run] nodes=$NODES tpn=$TPN iters=$ITERS iface=$IFACE msg=$MSG"

run_one() {
  local tag="$1" nodelay="$2" quickack="$3"
  local out="$OUTROOT/$tag"
  # Rendezvous dir MUST be unique per run and cleaned: a shared/stale dir lets a
  # rank read a previous run's (possibly different-interface) address and wire
  # the ring to the wrong peer, which deadlocks. Keep it under the timestamped
  # OUTROOT and wipe it first.
  local rdv="$OUTROOT/.rdv_$tag"
  rm -rf "$rdv"
  mkdir -p "$out" "$rdv"
  echo "[run] === $tag (nodelay=$nodelay quickack=$quickack) ==="
  PROBE_IFACE="$IFACE" PROBE_ITERS="$ITERS" PROBE_WARMUP=5 \
  PROBE_MSG_BYTES="$MSG" PROBE_NODELAY="$nodelay" PROBE_QUICKACK="$quickack" \
  PROBE_OUT="$out" PROBE_RDV="$rdv" PROBE_TAG="$tag" \
    srun -N"$NODES" --ntasks-per-node="$TPN" --cpu-bind=cores -t5 "$BIN"
}

# Baseline mirrors the RCCL socket bootstrap defaults (NODELAY on).
run_one "nodelay_on"        1 0
# Nagle on (NODELAY off): exposes delayed-ACK / Nagle interaction stalls.
run_one "nodelay_off"       0 0
# QUICKACK on top of NODELAY: suppresses delayed-ACK on the receiver.
run_one "nodelay_on_qack"   1 1

echo "[run] all done. analyze with:"
echo "  python3 $HERE/analyze_bootstrap_net_probe.py $OUTROOT"
echo "$OUTROOT"
