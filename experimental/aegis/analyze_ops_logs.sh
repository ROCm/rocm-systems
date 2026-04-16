#!/bin/bash
# Analyze /tmp/aegis_ops_probe/*.log for partial-instrumentation kernels.
# Reports per-op:
#   - number of distinct kernels patched
#   - worst and aggregate Partial-instrumentation ratios
#   - biggest kernels (by discovered site count)
LOGDIR=${1:-/tmp/aegis_ops_probe}

printf "%-22s | %8s | %8s | %-20s | %-14s | %s\n" \
    "OP" "KERNELS" "PARTIAL" "WORST_PARTIAL_RATIO" "SUM_PARTIAL" "BIGGEST_KERNEL_CFG"
printf -- "-%.0s" {1..140}; echo ""

for log in "$LOGDIR"/*.log; do
    op=$(basename "$log" .log)
    # Count kernels patched.
    kernels=$(grep -c "^\[aegisbit\] Patching kernel:" "$log" 2>/dev/null)
    partials=$(grep -c "^\[aegisbit\] Partial instrumentation:" "$log" 2>/dev/null)

    # Biggest CFG (discovered site count = VMEM + LDS).
    biggest=$(grep -oE "CFG: [0-9]+ BBs, [0-9]+ bytes, [0-9]+ VMEM \+ [0-9]+ LDS" "$log" 2>/dev/null \
        | awk '{ v=$6; l=$9; print v+l, $0 }' \
        | sort -k1 -rn | head -1 | cut -d' ' -f2-)

    # Worst (smallest coverage ratio) partial line.
    worst=""
    sum_partial="-"
    if [ "$partials" -gt 0 ]; then
        worst=$(grep -oE "Partial instrumentation: [0-9]+/[0-9]+" "$log" \
            | awk -F'[ /]' '{ printf "%.3f %d/%d\n", $3/$4, $3, $4 }' \
            | sort -g | head -1 | awk '{print $2}')
        sum_partial=$(grep -oE "Partial instrumentation: [0-9]+/[0-9]+" "$log" \
            | awk -F'[ /]' 'BEGIN{inst=0;disc=0} {inst+=$3; disc+=$4} END {printf "%d/%d", inst, disc}')
    fi

    printf "%-22s | %8s | %8s | %-20s | %-14s | %s\n" \
        "$op" "$kernels" "$partials" "${worst:--}" "$sum_partial" "$biggest"
done

echo ""
echo "=== Ops with ANY Partial instrumentation (replay candidates) ==="
for log in "$LOGDIR"/*.log; do
    op=$(basename "$log" .log)
    if grep -q "Partial instrumentation:" "$log" 2>/dev/null; then
        echo "  $op"
    fi
done
