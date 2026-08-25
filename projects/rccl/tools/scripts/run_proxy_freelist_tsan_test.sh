#!/usr/bin/env bash
# Manual check that the proxy-op free-list handoff in src/proxy.cc is ordered.
#
# WHAT IT CHECKS
#   The proxy-op free list is handed between the main thread and the proxy thread
#   through shared memory: the proxy thread links returned ops and publishes the
#   list head, the main thread takes it. Since NCCL 2.28.9 (RCCL 80adf7fc56,
#   release note "Fixed operation ordering between main thread and proxy thread to
#   prevent hangs at large scale") that handoff uses an acquire/release protocol;
#   before it, plain stores and a plain read left producer and consumer
#   unsynchronised.
#
#   The test builds the real librccl from the tree as it is, with proxy.cc
#   instrumented, runs a real rccl-tests collective under ThreadSanitizer, and
#   fails if the handoff is reported as racy.
#
#   Scope: a report means the synchronisation is missing. It does not mean a
#   hang: on x86 the store ordering happens to work out regardless, so the hang
#   the fix prevents needs weakly ordered hardware. Races reported elsewhere in
#   proxy.cc are printed but do not fail the test.
#
# WHAT RE-RUNS IT
#   Nothing, today: it is run by hand. Wiring it into CI is possible but is its own
#   piece of work. The suites under tools/ci/lib are single-node sbatch scripts
#   driven by .github/workflows/rccl-suite.yml, and this needs two nodes, so it
#   would take a two-node sbatch of its own, a caller workflow gated on changes to
#   src/proxy.cc, and a reservation that can hold both nodes for the length of a
#   ROCm build plus two short runs.
#
# CHECKING THAT THE TEST CAN FAIL
#   Restore the pre-2.28.9 handoff by hand and re-run: in ncclLocalOpAppend
#   replace the acquire COMPILER_ATOMIC_EXCHANGE on pool->freeOps with a plain
#   read plus __sync_val_compare_and_swap, and in ncclProxyGetPostedOps replace
#   the release COMPILER_ATOMIC_COMPARE_EXCHANGE loop with the plain store of
#   pool->freeOps[i]. Both hunks are the reverse of the diff in 80adf7fc56. The
#   run then reports the race, in the order of ten to twenty times.
#
# WHY THE TEST-ONLY SHIMS ARE NEEDED
#   Two of them are there because otherwise the test cannot observe the handoff at
#   all and would pass no matter what the code does: the pool is mapped twice per
#   process so a sanitizer cannot correlate producer and consumer, and
#   MAX_OPS_PER_PEER is large enough that a short run never reaches the shared
#   list. The third makes the take path say once that it ran, which is what lets
#   this test tell "no race" apart from "never looked"; a run without that line
#   fails, and the answer is usually a lower OPS_PER_PEER. All three are applied to
#   a copy of the hipified proxy.cc inside the build directory, so the source tree
#   is never modified. See proxy_tsan_shims.py.
#
# REQUIREMENTS
#   Two hosts that expose GPUs, ROCm with the ThreadSanitizer runtime, Open MPI,
#   and an rccl-tests all_reduce_perf built against this RCCL. Slurm is used for
#   host discovery and for the build when available, but is not required.
#
# ENVIRONMENT (everything is auto-detected; set to override)
#   TEST_HOSTS      comma-separated hosts to use, e.g. "nodeA,nodeB".
#                   Default: two idle Slurm nodes that expose enough GPUs.
#   TEST_BIN        path to rccl-tests all_reduce_perf.
#                   Default: first match under ../rccl-tests/build*/.
#   TEST_IFACE      NIC for MPI and RCCL out-of-band traffic.
#                   Default: the interface the launch host routes to its peer
#                   over. If that cannot be determined, MPI and RCCL are left to
#                   choose, which may be slower or pick an unusable link.
#   GPUS_PER_PROC   GPUs per rank (default 4); hosts with fewer are skipped.
#   OPS_PER_PEER    ops each peer starts with (default 32, 1..256); see the shims.
#   NODE_DOMAIN     domain appended when a bare host name does not resolve.
#                   Default: the domain of this host's own FQDN.
#   TSANRT          path to libclang_rt.tsan-x86_64.so on the hosts.
#                   Default: asked from the compiler.
#   ROCM_PATH       ROCm prefix on the hosts (default /opt/rocm).
#   OUT             where the instrumented library and logs are written.
#   BUILD_CPUS, BUILD_MINUTES  srun sizing for the build.
#   REMOTE_TIMEOUT  seconds allowed for each remote command (default 900).
#
# Exit status: 0 clean, 1 racy or the run did not complete, 3 no usable hosts.
#
# Usage: run_proxy_freelist_tsan_test.sh [build_dir]
set -uo pipefail

BD="${1:-build_proxy_tsan}"
HERE="$(cd "$(dirname "$0")" && pwd)"
RCCL="$(cd "$HERE/../.." && pwd)"
PROJECTS="$(cd "$RCCL/.." && pwd)"
OUT="${OUT:-$HOME/.proxy_freelist_tsan}"
SUPP="$HERE/proxy_tsan.supp"
ROCM_PATH="${ROCM_PATH:-/opt/rocm}"
GPUS_PER_PROC="${GPUS_PER_PROC:-4}"
OPS_PER_PEER="${OPS_PER_PEER:-32}"
BUILD_CPUS="${BUILD_CPUS:-64}"
BUILD_MINUTES="${BUILD_MINUTES:-45}"
NODE_DOMAIN="${NODE_DOMAIN:-$(hostname -f 2>/dev/null | sed -n 's/^[^.]*\.//p')}"
LOG="$OUT/run.log"
# The consumer side of the handoff, used twice: reports are attributed to it, and
# the build keeps it out of line and refuses to go on if the symbol is missing.
# Without that guard an inlined or renamed function would leave reports
# unattributed, and racy code would read as clean.
CONSUMER_FN="${CONSUMER_FN:-ncclLocalOpAppend}"
rc=0

cd "$RCCL" || exit 1
[ -f "$BD/src/CMakeFiles/rccl.dir/flags.make" ] || { echo "[FAIL] no usable build dir: $BD"; exit 1; }

if [ -z "${TEST_BIN:-}" ]; then
  # Several rccl-tests build directories are common. Take the most recent one
  # instead of whatever the glob lists first, and say so when there is a choice,
  # because only the caller knows which build goes with this RCCL.
  found=0
  for cand in "$PROJECTS"/rccl-tests/build*/all_reduce_perf; do
    [ -x "$cand" ] || continue
    found=$((found + 1))
    if [ -z "${TEST_BIN:-}" ] || [ "$cand" -nt "$TEST_BIN" ]; then TEST_BIN="$cand"; fi
  done
  [ "$found" -gt 1 ] && echo "== note: $found rccl-tests builds found, using the newest; set TEST_BIN to pin one"
fi
[ -x "${TEST_BIN:-}" ] || { echo "[FAIL] no rccl-tests all_reduce_perf found under $PROJECTS/rccl-tests; set TEST_BIN"; exit 1; }
echo "== rccl-tests binary: $TEST_BIN"
mkdir -p "$OUT"

# --- helpers ---------------------------------------------------------------
# Bare node names are not always resolvable, so fall back to the FQDN form.
resolve_host() {
  local h="$1"
  getent hosts "$h" >/dev/null 2>&1 && { echo "$h"; return 0; }
  [ -n "$NODE_DOMAIN" ] && getent hosts "$h.$NODE_DOMAIN" >/dev/null 2>&1 && { echo "$h.$NODE_DOMAIN"; return 0; }
  return 1
}

# Run where ROCm and the GPUs are. Only the local host needs no ssh.
remote() {
  local host="$1"; shift
  case "$host" in
    "$(hostname)"|"$(hostname -f 2>/dev/null)"|localhost) bash -c "$*" ;;
    *) timeout "${REMOTE_TIMEOUT:-900}" ssh -o StrictHostKeyChecking=no -o ConnectTimeout=8 "$host" "$*" ;;
  esac
}

# The build needs a machine with ROCm; use Slurm for it when present.
build_somewhere() {
  if command -v srun >/dev/null 2>&1; then
    srun -N1 -n1 -c"$BUILD_CPUS" -t"$BUILD_MINUTES" bash -c "$1"
  else
    bash -c "$1"
  fi
}

# --- pick two hosts that really expose GPUs --------------------------------
if [ -n "${TEST_HOSTS:-}" ]; then
  IFS=, read -r -a CAND <<< "$TEST_HOSTS"
elif command -v sinfo >/dev/null 2>&1; then
  mapfile -t CAND < <(sinfo -t idle -h -o "%n")
else
  echo "[FAIL] no Slurm for host discovery; set TEST_HOSTS=hostA,hostB"; exit 1
fi

NODES=()
for n in "${CAND[@]}"; do
  [ "${#NODES[@]}" -ge 2 ] && break
  [ -n "$n" ] || continue
  addr=$(resolve_host "$n") || { echo "  [preflight] skip $n (does not resolve)"; continue; }
  gpus=$(remote "$addr" "'$ROCM_PATH/bin/rocminfo' 2>/dev/null | grep -c 'Device Type:.*GPU'" 2>/dev/null | tr -d '[:space:]')
  if [ "${gpus:-0}" -ge "$GPUS_PER_PROC" ] 2>/dev/null; then
    NODES+=("$addr")
  else
    echo "  [preflight] skip $n (GPUs=${gpus:-unreachable}, need $GPUS_PER_PROC)"
  fi
done
[ "${#NODES[@]}" -ge 2 ] || { echo "[skip] need 2 reachable hosts with >= $GPUS_PER_PROC GPUs, found ${#NODES[@]}"; exit 3; }
HOSTS="${NODES[0]}:1,${NODES[1]}:1"
LAUNCH="${NODES[0]}"
echo "== hosts: $HOSTS"

# The library carries the instrumentation but not the sanitizer runtime (clang
# does not link it into a shared object), so the runtime is preloaded into the
# process. Ask the compiler where it lives instead of assuming a ROCm layout.
if [ -z "${TSANRT:-}" ]; then
  TSANRT=$(remote "$LAUNCH" "for c in '$ROCM_PATH/bin/amdclang++' amdclang++ clang++; do
      p=\$(\$c -print-file-name=libclang_rt.tsan-x86_64.so 2>/dev/null); [ -f \"\$p\" ] && { echo \"\$p\"; break; }; done" \
      2>/dev/null | tr -d '[:space:]')
fi
[ -n "$TSANRT" ] || { echo "[FAIL] could not locate libclang_rt.tsan-x86_64.so on $LAUNCH (set TSANRT)"; exit 1; }
# A runtime that does not preload would produce a silent, meaningless clean run,
# so prove here that it loads.
remote "$LAUNCH" "LD_PRELOAD='$TSANRT' /bin/true" >/dev/null 2>&1 \
  || { echo "[FAIL] TSAN runtime does not preload on $LAUNCH: $TSANRT"; exit 1; }
echo "== tsan runtime: $TSANRT"

# MPI and RCCL both need to agree on the out-of-band link. Whatever the launch
# host uses to reach its peer is the safe answer, and it keeps the run off any
# interface that cannot carry it.
if [ -z "${TEST_IFACE:-}" ]; then
  TEST_IFACE=$(remote "$LAUNCH" "ip=\$(getent hosts '${NODES[1]}' | awk '{print \$1; exit}')
      [ -n \"\$ip\" ] && ip -o route get \"\$ip\" 2>/dev/null | awk '{for(i=1;i<=NF;i++) if(\$i==\"dev\"){print \$(i+1); exit}}'" \
      2>/dev/null | tr -d '[:space:]')
fi
IFACE_FLAGS=""
if [ -n "$TEST_IFACE" ]; then
  IFACE_FLAGS="--mca oob_tcp_if_include $TEST_IFACE --mca btl_tcp_if_include $TEST_IFACE -x NCCL_SOCKET_IFNAME=$TEST_IFACE"
  echo "== interface: $TEST_IFACE"
else
  echo "== interface: not detected, leaving the choice to MPI and RCCL"
fi

# --- build the instrumented library ----------------------------------------
echo
echo "== building librccl with TSAN on proxy.cc =="
# The shims are applied to the hipified copy, so it has to be a faithful copy of
# the current source. hipify only regenerates it on a newer timestamp, so drop it
# and let the build write it again; that also clears a shimmed copy left behind by
# an earlier run. Only the build directory is affected.
rm -f "$BD/hipify/src/proxy.cc"
build_somewhere "cd '$RCCL' && cmake --build '$BD' -j $BUILD_CPUS" > "$OUT/build.log" 2>&1 \
  || { echo "[FAIL] build failed, see $OUT/build.log"; exit 1; }
grep -q "TEST-ONLY" "$BD/hipify/src/proxy.cc" \
  && { echo "[FAIL] $BD/hipify/src/proxy.cc still carries test shims from an earlier run; rebuild $BD"; exit 1; }
python3 "$HERE/proxy_tsan_shims.py" --in "$BD/hipify/src/proxy.cc" --out "$OUT/proxy_shimmed.cc" \
        --ops-per-peer "$OPS_PER_PEER" || { echo "[FAIL] could not apply the test shims"; exit 1; }
build_somewhere "CONSUMER_FN='$CONSUMER_FN' '$HERE/build_tsan_proxy_lib.sh' '$BD' '$OUT/lib' '$OUT/proxy_shimmed.cc'" >> "$OUT/build.log" 2>&1 \
  || { echo "[FAIL] instrumented build failed, see $OUT/build.log"; exit 1; }
grep -E "^\[tsan\]" "$OUT/build.log" | sed 's/^/   /'

# --- run --------------------------------------------------------------------
# PMIX_MCA_gds=hash keeps PMIx off its shared-memory component, which collides
# with the address space TSAN reserves.
echo
echo "== running the collective under TSAN =="
remote "$LAUNCH" "
  cd '$PROJECTS'
  export PMIX_MCA_gds=hash
  mpirun --mca plm rsh --mca plm_rsh_agent 'ssh -o StrictHostKeyChecking=no' \
    $IFACE_FLAGS -H $HOSTS -np 2 \
    -x PMIX_MCA_gds -x LD_LIBRARY_PATH=$OUT/lib -x LD_PRELOAD=$TSANRT \
    -x TSAN_OPTIONS='halt_on_error=0 history_size=7 suppressions=$SUPP' \
    -x NCCL_MIN_NCHANNELS=8 \
    bash -c 'unset HIP_VISIBLE_DEVICES ROCR_VISIBLE_DEVICES; cd $PROJECTS; exec setarch \$(uname -m) -R $TEST_BIN -b 8 -e 1K -f 2 -g $GPUS_PER_PROC -n 50 -w 5'
" > "$LOG" 2>&1
RUN_RC=$?

# Reports are attributed to the consumer by the '#0' frames only, since matching
# the name anywhere in the log would also pick up unrelated reports that merely
# pass through the proxy thread. Keying on the pool's address instead is not an
# option: the same shm object also carries the unrelated pool->nextOps reports.
# The build guarantees the frame exists, so a rename or an inlining decision cannot
# turn this into a silent zero.
freelist_races() {
  awk -v fn="$CONSUMER_FN" '
    /WARNING: ThreadSanitizer: data race/ { inblk = 1; hit = 0 }
    inblk && /^ +#0 / && index($0, fn) { hit = 1 }
    inblk && /^SUMMARY: ThreadSanitizer/ { n += hit; inblk = 0 }
    END { print n + 0 }
  ' "$LOG" 2>/dev/null || echo 0
}

RACES=$(freelist_races)
TOTAL=$(grep -c "WARNING: ThreadSanitizer: data race" "$LOG" 2>/dev/null || true)
DONE=$(grep -c "Collective test concluded" "$LOG" 2>/dev/null || true)
TOOK=$(grep -c "main thread took the shared free list" "$LOG" 2>/dev/null || true)
echo "   collective completed: $DONE   shared-list takes seen: $TOOK   free-list races: $RACES"
echo "   exit status: $RUN_RC   (data-race reports in total: $TOTAL)"

echo
# A run that died early would report no races either, so it cannot count as clean.
if [ "${DONE:-0}" -gt 0 ]; then
  echo "[ok] the collective ran to completion"
else
  echo "[FAIL] the collective did not complete - see $LOG"
  rc=1
fi
# 66 is ThreadSanitizer's own exit code for "reported something", which is expected
# here because proxy.cc carries unrelated findings. Anything else non-zero means the
# run itself went wrong, and then a clean verdict would be meaningless.
if [ "${RUN_RC:-1}" -eq 0 ] || [ "${RUN_RC:-1}" -eq 66 ]; then
  echo "[ok] the run exited as expected ($RUN_RC)"
else
  echo "[FAIL] the run exited with $RUN_RC, which is neither success nor TSAN's 66 - see $LOG"
  rc=1
fi
# Without this the test cannot tell "no race" from "never looked".
if [ "${TOOK:-0}" -gt 0 ]; then
  echo "[ok] the shared free-list handoff was exercised"
else
  echo "[FAIL] the shared free-list handoff never ran, so nothing was checked - lower OPS_PER_PEER"
  rc=1
fi
if [ "${RACES:-0}" -eq 0 ]; then
  echo "[ok] the free-list handoff is race free"
else
  echo "[FAIL] the free-list handoff is reported as racy $RACES time(s) - see $LOG"
  rc=1
fi

echo
echo "note: $BD now holds the instrumented, shimmed proxy.cc object; rebuild it before using it for anything else"
[ "$rc" = 0 ] && echo "RESULT: PASS (logs in $OUT)" || echo "RESULT: FAIL (logs in $OUT)"
exit "$rc"
