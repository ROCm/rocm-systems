#!/bin/bash
set -e
set -u
# Test all SQ events and report results

echo "Testing SQ Event Packets After Fixes"
echo "====================================="
echo ""

pass=0
fail=0

for event in 1 3 4 24 26 27 44 45 46 47 50 54 55 58 60 61 70 71 75 102; do
    echo -n "SQ event $event: "
    result=$(python3 compare_packets.py SQ $event 2>&1 | grep -o "✓ Packets match" | head -1)
    if [ -n "$result" ]; then
        echo "PASS"
        ((pass++))
    else
        echo "FAIL"
        ((fail++))
    fi
done

echo ""
echo "====================================="
echo "Results: $pass passed, $fail failed"
echo "====================================="
