#!/usr/bin/env bash
#
# Compare SVM vs userptr for the read-only gemma-pt load simulation, with a
# per-phase breakdown (mmap/register/map/sdma/unmap/deregister/munmap) averaged
# over N runs. Pinpoints where each pin path spends its time.
# Requires the PRIMARY-ctx build (userptr/SVM pin needs the primary context).
#
# Usage: ./run_gemma_ro.sh [runs] [kfdtest_binary]
set -euo pipefail

RUNS="${1:-10}"
KFDTEST="${2:-./build-primary/kfdtest}"
FILTER='*GemmaModelLoadSimTest/0'

if [[ ! -x "$KFDTEST" ]]; then
    echo "kfdtest binary not found or not executable: $KFDTEST" >&2
    exit 1
fi

LOG="$(mktemp)"
trap 'rm -f "$LOG"' EXIT

for sv in 1 0; do
    path=$([ "$sv" = 1 ] && echo svm || echo userptr)
    echo "Running $path (read-only) x $RUNS..."
    for ((i = 1; i <= RUNS; i++)); do
        HSA_USE_SVM=$sv GEMMA_READONLY=1 "$KFDTEST" --gtest_filter="$FILTER" 2>&1 \
            | grep -E 'gemma-load|gemma-phase' >>"$LOG" || true
        printf '\r  %d/%d' "$i" "$RUNS"
    done
    echo
done

echo
echo "=== totals (median/mean over runs) ==="
awk '
    /gemma-load/ {
        sub(/.*gemma-load /, ""); path = $1
        c = $0; sub(/.*= /, "", c); sub(/ ms .*/, "", c)           # copy ms
        b = $0; sub(/.*\(/, "", b); sub(/ MiB\/s.*/, "", b)        # bandwidth
        ci[path]++; C[path, ci[path]] = c; cs[path] += c; bs[path] += b
    }
    function median(p, n,   v, j, k, t) {
        for (j = 1; j <= n; j++) v[j] = C[p, j]
        for (j = 1; j <= n; j++) for (k = j + 1; k <= n; k++)
            if (v[k] < v[j]) { t = v[j]; v[j] = v[k]; v[k] = t }
        return (n % 2) ? v[(n + 1) / 2] : (v[n / 2] + v[n / 2 + 1]) / 2
    }
    END {
        printf "%-9s %6s %12s %12s %12s\n", "path", "runs", "copy_med", "copy_mean", "bw_mean"
        for (p in ci)
            printf "%-9s %6d %9.1f ms %9.1f ms %8.1f MiB/s\n", \
                   p, ci[p], median(p, ci[p]), cs[p]/ci[p], bs[p]/ci[p]
    }
' "$LOG"

echo
echo "=== per-phase mean us/op over runs ==="
awk '
    BEGIN { np = split("mmap reg map sdma unmap dereg munmap", PH, " ") }
    /gemma-phase/ {
        sub(/.*gemma-phase /, ""); path = $1
        if (!(path in seen)) { seen[path] = 1; order[++no] = path }
        for (i = 1; i <= NF; i++) if ($i ~ /\|/) {
            split($i, xx, "|"); name = $(i-1)
            sum[path, name] += xx[2]
        }
        rc[path]++
    }
    END {
        printf "%-9s", "phase"
        for (o = 1; o <= no; o++) printf " %13s", order[o]
        printf "\n"
        for (k = 1; k <= np; k++) {
            printf "%-9s", PH[k]
            for (o = 1; o <= no; o++) {
                p = order[o]
                printf " %10.1f us", rc[p] ? sum[p, PH[k]] / rc[p] : 0
            }
            printf "\n"
        }
    }
' "$LOG"
