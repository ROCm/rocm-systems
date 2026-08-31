#!/bin/bash
# Run the hip-tests memory suites against the patched CLR, flag off then on.
# The suites are linked against /opt/rocm; LD_LIBRARY_PATH substitutes the
# patched libamdhip64 at run time (verified via AMD_LOG_LEVEL in e2e_run.sh).
set -uo pipefail

TESTS=~/airuntime28-hiptests/catch_tests/unit/memory
INSTALL=~/airuntime28-clr-install
OUT=~/airuntime28/results

export LD_LIBRARY_PATH="$INSTALL/lib:/opt/rocm/lib:${LD_LIBRARY_PATH:-}"

for BIN in MemoryTest1 MemoryTest2 DeviceMemoryTest; do
  P="$TESTS/$BIN"
  [ -x "$P" ] || P=$(find ~/airuntime28-hiptests -name "$BIN" -type f -perm -u+x 2>/dev/null | head -1)
  [ -x "$P" ] || { echo "SKIP $BIN (not found)"; continue; }
  for NT in 0 1; do
    LOG="$OUT/hiptests_${BIN}_nt${NT}.log"
    DEBUG_CLR_BLIT_NONTEMPORAL=$NT timeout 3600 "$P" \
        --reporter compact --durations no > "$LOG" 2>&1
    rc=$?
    LINE=$(grep -E '^(Passed|Failed|assertions|test cases)' "$LOG" | tail -2 | tr '\n' ' ')
    [ -z "$LINE" ] && LINE=$(tail -2 "$LOG" | tr '\n' ' ')
    printf '%-18s NT=%s exit=%-3s %s\n' "$BIN" "$NT" "$rc" "$LINE"
  done
done

echo
echo "=== failed assertions across all runs (should be identical off vs on) ==="
for BIN in MemoryTest1 MemoryTest2 DeviceMemoryTest; do
  for NT in 0 1; do
    L="$OUT/hiptests_${BIN}_nt${NT}.log"
    [ -f "$L" ] || continue
    n=$(grep -cE 'FAILED|error:' "$L" 2>/dev/null || echo 0)
    printf '  %-18s NT=%s  failing lines=%s\n' "$BIN" "$NT" "$n"
  done
done
echo "HIPTESTS_RUN_DONE"
