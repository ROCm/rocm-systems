#!/bin/bash
# PROOF OF CONCEPT coordinator for the init-parallel / execute-serial idea.
#
# Compares two ways of running K copies of a test:
#   SERIAL   : one process after another, each paying full (device-load + execute).
#   PIPELINE : all K processes warm up (RCCL device-code load) CONCURRENTLY via the
#              UT_INIT_BARRIER hook, then are released to EXECUTE one at a time.
# The pipeline should overlap the (dominant) device-load across the K processes while
# keeping execution strictly serial -- no GPU/HCA co-tenancy during the actual test.
#
# Requires a rccl-UnitTests built with the ut_init_barrier_poc() hook in common/main.cpp.
# Env: BIN, K, FILTER, GPUS (for warm-dev round-robin).
set -u
BIN=${BIN:-./build/debug/test/rccl-UnitTests}
K=${K:-6}
FILTER=${FILTER:-AllReduce.OutOfPlace}
GPUS=${GPUS:-8}
TMP=$(mktemp -d)
now(){ date +%s%3N; }

echo "BIN=$BIN  K=$K  FILTER=$FILTER"
common_env() { export UT_PROCESS_MASK=1 UT_MIN_GPUS=1 UT_MAX_GPUS=1 UT_SHOW_TIMING=0; }

echo "===== SERIAL baseline (K procs sequential, full init+execute each) ====="
common_env
t0=$(now)
for i in $(seq 1 "$K"); do
  "$BIN" --gtest_filter="$FILTER" > "$TMP/serial_$i.log" 2>&1
done
serial=$(( $(now) - t0 ))
echo "SERIAL wall = ${serial} ms"

echo "===== PIPELINE (init overlapped, execute serial) ====="
common_env
t0=$(now)
declare -a pids
for i in $(seq 1 "$K"); do
  UT_INIT_BARRIER="$TMP/p$i" UT_INIT_WARM_DEV=$(( (i-1) % GPUS )) \
    "$BIN" --gtest_filter="$FILTER" > "$TMP/pipe_$i.log" 2>&1 &
  pids[$i]=$!
done
# Barrier: wait until every process has finished its (overlapped) init.
for i in $(seq 1 "$K"); do while [ ! -f "$TMP/p$i.ready" ]; do sleep 0.02; done; done
tready=$(( $(now) - t0 ))
echo "all $K inits complete (OVERLAPPED) at +${tready} ms"
# Release EXECUTE one at a time -> strictly serial execution.
for i in $(seq 1 "$K"); do
  touch "$TMP/p$i.go"
  wait "${pids[$i]}"
done
pipe=$(( $(now) - t0 ))
echo "PIPELINE wall = ${pipe} ms  (init-overlap window = ${tready} ms, execute-serial = $((pipe-tready)) ms)"

echo "===== RESULT ====="
echo "serial=${serial}ms  pipeline=${pipe}ms  saved=$((serial-pipe))ms  speedup=$(awk "BEGIN{printf \"%.2fx\", ${serial}/${pipe}}")"
echo "(expected: pipeline hides ~(K-1) device-loads; serial pays all K)"
rm -rf "$TMP"
