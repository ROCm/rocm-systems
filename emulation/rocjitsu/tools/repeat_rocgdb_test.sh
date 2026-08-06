#!/usr/bin/env bash
# Run one or more gdb.rocm files repeatedly, serially, and report how many
# iterations were clean. Concurrent Mirage sessions collide, so iterations never
# overlap.
#
# Usage: repeat_rocgdb_test.sh <iterations> <out-dir> <test.exp> [test.exp ...]
set -u

iterations=$1
outdir=$2
shift 2

# Everything else -- suite, gdb, venv, binaries under test -- resolves the way
# run_rocgdb_official.py already resolves it, so this adds no second source of
# truth for any path.
root=$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)

mkdir -p "$outdir"
passed=0
for i in $(seq 1 "$iterations"); do
    run="$outdir/iter-$(printf '%02d' "$i")"
    rm -rf "$run"
    python3 "$root/emulation/rocjitsu/tools/run_rocgdb_official.py" \
        --output "$run" --tests "$@" > "$run.console" 2>&1
    rc=$?
    if [ $rc -eq 0 ]; then
        passed=$((passed + 1))
        echo "iter $i: PASS"
    else
        echo "iter $i: FAIL (rc=$rc)"
        grep -h "^FAIL\|^UNRESOLVED\|^ERROR" "$run"/*/gdb.sum 2>/dev/null | sed 's/^/    /'
    fi
done
echo "clean iterations: $passed/$iterations"
[ "$passed" -eq "$iterations" ]
