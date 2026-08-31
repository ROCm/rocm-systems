#!/bin/bash
# Sample the GPU clock, for the duration given, while something else runs.
#
# This part cannot be clock-pinned - three separate attempts to force a fixed
# frequency were abandoned - so the honest alternative is to measure the clock
# instead of controlling it and report what it did. If it sits at its ceiling
# throughout, clock state cannot explain a run-to-run difference and the cause is
# elsewhere.
#
# Run standalone or, as run_all.sh does, in the background alongside a run:
#   ./clock_watch.sh 120 > clocks.txt &
set -uo pipefail
SMI=${SMI:-/opt/rocm/bin/amd-smi}
SECONDS_TO_WATCH=${1:-120}
INTERVAL=0.35
SAMPLES=$(python3 -c "print(int($SECONDS_TO_WATCH / $INTERVAL))")

TMP=$(mktemp)

# The summary is printed on the way out as well as at the end, because the usual
# caller stops this watcher as soon as the run it is watching finishes. Without
# the trap the samples were collected and then thrown away.
summarise() {
  n=$(wc -l < "$TMP")
  echo
  echo "=== gfx clock samples (${n} taken, ${INTERVAL}s apart) ==="
  if [ "$n" -gt 0 ]; then
    sort "$TMP" | uniq -c | sort -rn
    echo
    echo "distinct values: $(sort -u "$TMP" | wc -l)"
    echo "min: $(grep -oE '[0-9]+' "$TMP" | sort -n | head -1) MHz"
    echo "max: $(grep -oE '[0-9]+' "$TMP" | sort -n | tail -1) MHz"
  else
    echo "no samples captured (amd-smi output format may have changed)"
  fi
  echo "CLOCK_WATCH_DONE"
  rm -f "$TMP"
}
trap 'summarise; exit 0' TERM INT
trap 'rm -f "$TMP"' EXIT

for _ in $(seq 1 "$SAMPLES"); do
  s=$($SMI metric -g 0 2>/dev/null | grep -A2 'GFX_0:' | grep -oE 'CLK: [0-9]+ MHz' | head -1)
  [ -n "$s" ] && echo "$s" >> "$TMP"
  sleep $INTERVAL
done
summarise
