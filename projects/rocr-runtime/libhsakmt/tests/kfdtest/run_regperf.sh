#!/usr/bin/env bash
#
# Run the SVM register/munmap, prefetch, and register+map micro-benchmarks N
# times and report the average times per case. Build kfdtest with the desired
# context (primary or HSAKMT_SECONDARY_CTX) before running this script.
#
# To compare SVM vs userptr for the register+map case, run twice:
#   HSA_USE_SVM=1 ./run_regperf.sh   # SVM path  (regmapsvm)
#   HSA_USE_SVM=0 ./run_regperf.sh   # userptr path (regmapuserptr, primary ctx only)
#
# Usage: ./run_regperf.sh [runs] [kfdtest_binary]
set -euo pipefail

RUNS="${1:-100}"
KFDTEST="${2:-./build/kfdtest}"
FILTER='*KFDSVMRangeTest*PrefetchPerfTest/0'

if [[ ! -x "$KFDTEST" ]]; then
    echo "kfdtest binary not found or not executable: $KFDTEST" >&2
    exit 1
fi

LOG="$(mktemp)"
trap 'rm -f "$LOG"' EXIT

echo "Running '$KFDTEST --gtest_filter=$FILTER' $RUNS times..."
for ((i = 1; i <= RUNS; i++)); do
    "$KFDTEST" --gtest_filter="$FILTER" >>"$LOG" 2>&1 || {
        echo "Run $i failed; see output:" >&2
        tail -n 20 "$LOG" >&2
        exit 1
    }
    printf '\r  %d/%d done' "$i" "$RUNS"
done
echo

# Average the two timed metrics per case across all runs. Register/hole lines
# look like "gap NKB: register ... = R ms; munmap = M ms"; prefetch lines look
# like "prefetch ... to_gpu = G ms; to_cpu = C ms".
awk '
    /gap [0-9]+KB:/ {
        label = $0; sub(/:.*/, "", label); gsub(/ /, "", label)   # e.g. gap0KB

        a = $0; sub(/.*register[^=]*= /, "", a); sub(/ ms.*/, "", a)
        b = $0; sub(/.*munmap = /, "", b);       sub(/ ms.*/, "", b)

        aSum[label] += a; bSum[label] += b; cnt[label]++
        aName[label] = "register"; bName[label] = "munmap"
        next
    }
    /prefetch .* to_gpu = / {
        label = "prefetch"

        a = $0; sub(/.*to_gpu = /, "", a); sub(/ ms.*/, "", a)
        b = $0; sub(/.*to_cpu = /, "", b); sub(/ ms.*/, "", b)

        aSum[label] += a; bSum[label] += b; cnt[label]++
        aName[label] = "to_gpu"; bName[label] = "to_cpu"
        next
    }
    /regmap (svm|userptr):/ {
        label = $0; sub(/:.*/, "", label); gsub(/ /, "", label)   # regmapsvm / regmapuserptr

        a = $0; sub(/.*register\+map[^=]*= /, "", a); sub(/ ms.*/, "", a)
        b = $0; sub(/.*unmap\+dereg = /, "", b);      sub(/ ms.*/, "", b)

        aSum[label] += a; bSum[label] += b; cnt[label]++
        aName[label] = "reg+map"; bName[label] = "unmap+dereg"
        next
    }
    END {
        printf "%-14s %8s   %-11s %11s   %-11s %11s\n", \
               "case", "samples", "metric", "avg", "metric", "avg"
        for (l in cnt)
            printf "%-14s %8d   %-11s %8.3f ms   %-11s %8.3f ms\n", \
                   l, cnt[l], aName[l], aSum[l] / cnt[l], bName[l], bSum[l] / cnt[l]
    }
' "$LOG"

