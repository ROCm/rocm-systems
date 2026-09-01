#!/bin/bash
# Run every experiment and write a self-describing result set.
#
# The only run entry point. The previous one omitted four of the five most
# important measurements, which meant "I ran run_all.sh" and "I ran the
# experiments" were different statements.
#
# Output lands in results/<UTC timestamp>/:
#   provenance.txt      machine, ROCm, revision, clocks, both cache sizes
#   isa_check.txt       the variants compile to what they claim
#   <experiment>.txt    full human-readable output
#   rows.tsv            every machine-readable row from every experiment
#   summary.txt         the handful of numbers the report quotes
#
# rows.tsv is the point: report tables are regenerated from it rather than
# hand-copied, because two published tables had already gone stale by hand.
#
#   ./run_all.sh                            everything
#   ./run_all.sh quick                      fewer iterations, for a smoke test
#   ./run_all.sh only concurrency isolated_copy
#   ./run_all.sh summarise results/<stamp>  rebuild summary.txt from rows.tsv alone
set -uo pipefail
cd "$(dirname "$0")"

MODE=${1:-full}
STAMP=$(date -u '+%Y%m%d_%H%M%S')
OUT="results/$STAMP"

# Regenerating the summary from rows.tsv without re-measuring is the mechanism
# that keeps report tables from being hand-copied, so it has to be usable on its
# own rather than only as the tail of a full run.
SUMMARISE_ONLY=0
if [ "$MODE" = "summarise" ]; then
  OUT=${2:?"usage: ./run_all.sh summarise results/<stamp>"}
  [ -f "$OUT/rows.tsv" ] || { echo "no $OUT/rows.tsv"; exit 1; }
  SUMMARISE_ONLY=1
  STAMP=$(basename "$OUT")
fi

mkdir -p "$OUT"

# Runtime chatter that carries no information and would otherwise bury the
# tables. Kept in one place so no experiment invents its own filter.
NOISE='rj warn|Resource leak|LoadLib|Secondary CUID|AMDCUID'

if [ "$MODE" = "quick" ]; then
  ITERS_FAST=10; ITERS_SLOW=8;  WARMUP=3
else
  ITERS_FAST=120; ITERS_SLOW=25; WARMUP=8
fi

# name : arguments. Iteration counts differ because a case timing 12 us needs far
# more repeats to resolve a percent than one timing 5 ms.
declare -a EXPERIMENTS=(
  "cache_capacity     --iters $ITERS_SLOW --warmup $WARMUP"
  "flush_sensitivity  --iters $ITERS_SLOW --warmup $WARMUP"
  "isolated_copy      --iters $ITERS_SLOW --warmup $WARMUP"
  "size_curve         --iters $ITERS_SLOW --warmup $WARMUP"
  "small_copy         --iters $ITERS_FAST --warmup $WARMUP"
  "adversarial        --iters $ITERS_FAST --warmup $WARMUP"
  "concurrency        --iters $ITERS_SLOW --warmup $WARMUP"
  "residency          --iters $ITERS_SLOW --warmup $WARMUP"
)

failed=0
ISA_RC=0

if [ "${1:-}" = "only" ]; then
  shift
  keep=()
  for want in "$@"; do
    for e in "${EXPERIMENTS[@]}"; do
      [ "${e%% *}" = "$want" ] && keep+=("$e")
    done
  done
  EXPERIMENTS=("${keep[@]}")
  ITERS_FAST=120; ITERS_SLOW=25; WARMUP=8
fi

# ---------------------------------------------------------------------------
# Provenance first, so a result set is interpretable without asking anyone.
# ---------------------------------------------------------------------------
if [ $SUMMARISE_ONLY -eq 0 ]; then
{
  echo "AIRUNTIME-28 result set $STAMP"
  echo "mode: $MODE"
  echo
  ./machine_state.sh 2>&1
  echo
  echo "=================== SOURCE REVISION ==================="
  git -C . rev-parse HEAD 2>/dev/null || echo "(not a git checkout on this host)"
  git -C . status --porcelain 2>/dev/null | sed 's/^/  dirty: /' || true
  echo
  echo "=================== CLR UNDER TEST ==================="
  if [ -d ~/airuntime28-clr-install ]; then
    echo "patched CLR present at ~/airuntime28-clr-install"
    ls -la ~/airuntime28-clr-install/lib/libamdhip64.so* 2>/dev/null | head -3
  else
    echo "no patched CLR build; e2e_memcpy will exercise stock ROCm instead"
  fi
  echo
  echo "=================== CACHE SIZE: REPORTED vs MEASURED ==================="
  echo "The driver's l2CacheSize and the measured GL2 capacity disagree by 24x."
  echo "Every footprint in this suite is sized against the measured figure."
  ./build/isolated_copy --size-mib 16 --iters 1 --warmup 0 2>/dev/null \
    | grep -E 'GL2 capacity|flush|device|clock' || true
  echo
  echo "=================== CONFIGURED CONSTANTS ==================="
  grep -E '^constexpr (u64|int|unsigned)' src/common/config.h
} > "$OUT/provenance.txt" 2>&1
echo "provenance -> $OUT/provenance.txt"

# Clock behaviour during the run, since this part cannot be clock-pinned.
if [ -x ./clock_watch.sh ]; then
  ./clock_watch.sh 600 > "$OUT/clocks.txt" 2>&1 &
  CLOCK_PID=$!
else
  CLOCK_PID=""
fi

# ---------------------------------------------------------------------------
# The variants must compile to what they claim before any timing is trusted.
# ---------------------------------------------------------------------------
./isa_check.sh > "$OUT/isa_check.txt" 2>&1
ISA_RC=$?
if [ $ISA_RC -ne 0 ]; then
  echo "ISA CHECK FAILED - see $OUT/isa_check.txt. Timing results are not trustworthy."
else
  echo "isa_check          ok"
fi

# ---------------------------------------------------------------------------
for entry in "${EXPERIMENTS[@]}"; do
  name=${entry%% *}
  args=${entry#* }
  [ -x "build/$name" ] || { printf '  %-18s SKIP (not built)\n' "$name"; continue; }
  start=$(date +%s)
  # shellcheck disable=SC2086
  ./build/"$name" $args > "$OUT/$name.raw" 2>&1
  rc=$?
  grep -vE "$NOISE" "$OUT/$name.raw" | grep -v '^ROW' > "$OUT/$name.txt"
  grep '^ROW' "$OUT/$name.raw" | cut -f2- >> "$OUT/rows.tsv"
  rm -f "$OUT/$name.raw"
  elapsed=$(( $(date +%s) - start ))
  if [ $rc -ne 0 ]; then
    printf '  %-18s FAILED (rc=%d, %ds) - assertions in %s.txt\n' "$name" $rc $elapsed "$name"
    failed=1
  else
    printf '  %-18s ok (%ds)\n' "$name" $elapsed
  fi
done

# TERM, not KILL: the watcher writes its summary on the way out.
if [ -n "$CLOCK_PID" ]; then
  kill -TERM $CLOCK_PID 2>/dev/null
  wait $CLOCK_PID 2>/dev/null
fi
fi  # SUMMARISE_ONLY

# ---------------------------------------------------------------------------
# The numbers the report quotes, pulled from rows.tsv rather than re-typed.
# ---------------------------------------------------------------------------
{
  echo "AIRUNTIME-28 summary, result set $STAMP"
  echo
  echo "--- isolated 1 GiB copy: effect of each change against its own control ---"
  awk -F'\t' '$1=="isolated_copy_controlled" {printf "  %-20s %+8.2f%%  %s\n", $2, $4, $6}' \
    "$OUT/rows.tsv" 2>/dev/null
  echo
  echo "--- size curve: nt-store-128 against plain-128 ---"
  awk -F'\t' '$1=="size_curve" && $2=="nt-store-128" && $3=="delta_pct" \
    {split($6, f, " "); sub("size=", "", f[1]); \
     printf "  %-10s %+8.2f%%  %s\n", f[1], $4, ($6 ~ /sig=1/ ? "significant" : "")}' \
    "$OUT/rows.tsv" 2>/dev/null
  echo
  echo "--- concurrency: victim time, by victim working set ---"
  awk -F'\t' '$1=="concurrency" && $2=="nt-store-128" && $3=="victim_delta_pct" \
    {split($6, f, " "); sub("ws_mib=", "", f[1]); \
     printf "  %6s MiB %+8.2f%%  %s\n", f[1], $4, ($6 ~ /sig=1/ ? "significant" : "")}' \
    "$OUT/rows.tsv" 2>/dev/null
  echo
  echo "--- adversarial: did any case make the hint lose ---"
  awk -F'\t' '$1=="adversarial" && $2=="-" {printf "  %-24s %s %s\n", $3, $4, $6}' \
    "$OUT/rows.tsv" 2>/dev/null
  echo
  echo "--- cache facts ---"
  awk -F'\t' '$1=="cache_capacity" && $2=="capacity" {printf "  measured capacity %s MiB\n", $4}' \
    "$OUT/rows.tsv" 2>/dev/null
  awk -F'\t' '$1=="cache_capacity" && $2=="plateau" {printf "  cached plateau    %.0f GB/s\n", $4}' \
    "$OUT/rows.tsv" 2>/dev/null
  awk -F'\t' '$1=="cache_capacity" && $2=="hbm_floor" {printf "  HBM floor         %.0f GB/s\n", $4}' \
    "$OUT/rows.tsv" 2>/dev/null
  awk -F'\t' '$1=="residency" && $2=="-" {printf "  max cold/warm     %s x  (1.0 = nothing survives a dispatch)\n", $4}' \
    "$OUT/rows.tsv" 2>/dev/null
  awk -F'\t' '$1=="residency_by_alloc" && $2=="-" {printf "  alloc kinds retaining across a dispatch: %s\n", $4}' \
    "$OUT/rows.tsv" 2>/dev/null
  echo
  echo "--- flush sizing ---"
  awk -F'\t' '$1=="flush_sensitivity" && $3=="smallest_adequate_mib" \
    {printf "  smallest adequate flush     %.0f MiB\n", $4}' "$OUT/rows.tsv" 2>/dev/null
  awk -F'\t' '$1=="flush_sensitivity" && $3=="cold_latency_spread_pct" \
    {printf "  cold latency spread         %.2f%% across a 16x range of flush sizes\n", $4}' \
    "$OUT/rows.tsv" 2>/dev/null
  echo
  [ $ISA_RC -ne 0 ] && echo "WARNING: ISA check failed; see isa_check.txt"
  [ $failed -ne 0 ] && echo "WARNING: at least one experiment reported a failed assertion"
  echo "e2e_memcpy is not run here: it needs a patched CLR build. See e2e_run.sh."
  echo
  echo "Regenerate this file from rows.tsv without re-measuring:"
  echo "  ./run_all.sh summarise $OUT"
} > "$OUT/summary.txt" 2>&1

echo
cat "$OUT/summary.txt"
echo
echo "result set: $OUT"
exit $(( failed || ISA_RC ))
