#!/usr/bin/env bash
#
# Copyright (c) 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT
#
# Run one or more gdb.rocm files repeatedly, serially, and report how many
# iterations were clean. Concurrent Mirage sessions collide, so iterations never
# overlap -- and neither may two copies of this script, which the lock below
# enforces across processes.
#
# Usage: repeat_rocgdb_test.sh <iterations> <out-dir> <test.exp> [test.exp ...]
set -uo pipefail

usage() {
    echo "Usage: $(basename "$0") <iterations> <out-dir> <test.exp> [test.exp ...]" >&2
    exit 2
}

[ $# -ge 3 ] || usage
case $1 in
    '' | *[!0-9]*) usage ;;
esac
[ "$1" -ge 1 ] || usage

iterations=$1
outdir=$2
shift 2

# run_rocgdb_official.py is a sibling and resolves everything else -- suite, gdb,
# venv, binaries under test -- itself, so this knows no layout beyond its own
# directory and adds no second source of truth for any path.
here=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd) || exit 1

mkdir -p "$outdir" || exit 1

# Serialize against any other copy of this script. Overlapping runs create
# overlapping Mirage sessions, which collide and report failures that are not
# real -- the one thing a flake count must not do.
exec 9>"$outdir/.repeat.lock"
if ! flock -n 9; then
    echo "another repeat_rocgdb_test.sh holds $outdir/.repeat.lock" >&2
    exit 2
fi

passed=0
for i in $(seq 1 "$iterations"); do
    run="$outdir/iter-$(printf '%03d' "$i")"
    rm -rf "$run"
    python3 "$here/run_rocgdb_official.py" --output "$run" --tests "$@" > "$run.console" 2>&1
    rc=$?
    if [ $rc -eq 0 ]; then
        passed=$((passed + 1))
        echo "iter $i: PASS"
    else
        echo "iter $i: FAIL (rc=$rc)"
        # run_rocgdb_official.py owns the pass/fail classification; read the
        # summary it wrote rather than re-deriving one from gdb.sum.
        python3 -c 'import json,sys
try:
    result = json.load(open(sys.argv[1]))
except (OSError, ValueError):
    sys.exit(0)
for record in result.get("records", []):
    if not record.get("passed"):
        bad = record.get("bad_statuses") or {}
        detail = " ".join(f"{k}={v}" for k, v in sorted(bad.items()))
        # Bound out of the f-string: the snippet is inside a single-quoted shell
        # word, so an inner "..." subscript would have to be backslash-escaped,
        # and a backslash there is a Python syntax error -- the reporter would
        # die on exactly the iterations it exists to explain.
        name = record.get("test")
        print(f"    {name} {detail}".rstrip())' "$run/result.json"
    fi
done
echo "clean iterations: $passed/$iterations"
[ "$passed" -eq "$iterations" ]
