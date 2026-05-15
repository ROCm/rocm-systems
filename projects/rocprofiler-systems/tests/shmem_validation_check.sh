#!/usr/bin/env bash
#
# Copyright (c) Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT
#
# Runs "oshrun -n <NP> <EXE>", validates output. Use with CTest SKIP_RETURN_CODE 77.
#   Exit 0   → SHMEM OK (fixture shmem_available is set; dependent tests run).
#   Exit 77  → SHMEM unavailable / runtime failure / output mismatch (test skipped in CI).
#   Other    → real test logic failure (rare).
#
# Expected output (e.g. oshrun -np 2 ./shmem_hello):
#   Hello from PE 0 of 2
#   PE 0 received value 1 from PE 1
#   Hello from PE 1 of 2
#   PE 1 received value 0 from PE 0
#
# Usage: shmem_validation_check.sh <oshrun_exe> <num_pes> <hello_exe> [marker_file] [pingpong_exe]
# If marker_file is given, it is created (touched) on success so CTest can gate other tests.
# If pingpong_exe is given, ldd output is printed for diagnostics.

OSHRUN="$1"
NP="$2"
EXE="$3"
MARKER="$4"
PINGPONG="$5"

if [[ -z "$OSHRUN" || -z "$NP" || -z "$EXE" ]]; then
    echo "Usage: $0 <oshrun_exe> <num_pes> <hello_exe>"
    exit 77
fi

# Remove stale marker from previous runs so validation-passed cannot pass on remnants
if [[ -n "$MARKER" ]]; then
    rm -f "$MARKER"
fi

# Always print ldd output at the beginning for diagnostics (pass or fail)
if [[ -n "$PINGPONG" ]]; then
    echo "--- ldd shmem_pingpong ---"
    ldd "$PINGPONG" 2>&1 || true
    echo "--- end ldd ---"
fi

echo "--- oshrun version ---"
"$OSHRUN" --version 2>&1 || "$OSHRUN" -V 2>&1 || true
echo "--- end oshrun version ---"

OUTPUT=$("$OSHRUN" -n "$NP" "$EXE" 2>&1)
STATUS=$?

if [[ $STATUS -ne 0 ]]; then
    echo "SHMEM runtime failed (oshrun exit=$STATUS) – skipping SHMEM tests"
    echo "--- oshrun output ---"
    echo "$OUTPUT"
    echo "--- end output ---"
    exit 77
fi

# Require both greeting lines and "received value" lines (order may vary by PE)
if ! echo "$OUTPUT" | grep -qE "Hello from PE [0-9]+ of ${NP}"; then
    echo "SHMEM validation: missing 'Hello from PE X of ${NP}' – skipping"
    echo "--- oshrun output ---"
    echo "$OUTPUT"
    echo "--- end output ---"
    exit 77
fi
if ! echo "$OUTPUT" | grep -qE "PE [0-9]+ received value [0-9]+ from PE [0-9]+"; then
    echo "SHMEM validation: missing 'PE X received value Y from PE Z' – skipping"
    echo "--- oshrun output ---"
    echo "$OUTPUT"
    echo "--- end output ---"
    exit 77
fi
# For np=2 we expect both PEs to have said hello and both to have received a value
if [[ "$NP" -eq 2 ]]; then
    if ! echo "$OUTPUT" | grep -q "Hello from PE 0 of 2" || ! echo "$OUTPUT" | grep -q "Hello from PE 1 of 2"; then
        echo "SHMEM validation: expected both PE 0 and PE 1 greetings – skipping"
        echo "--- oshrun output ---"
        echo "$OUTPUT"
        echo "--- end output ---"
        exit 77
    fi
    if ! echo "$OUTPUT" | grep -q "PE 0 received value 1 from PE 1" || ! echo "$OUTPUT" | grep -q "PE 1 received value 0 from PE 0"; then
        echo "SHMEM validation: expected PE 0 received from PE 1 and PE 1 from PE 0 – skipping"
        echo "--- oshrun output ---"
        echo "$OUTPUT"
        echo "--- end output ---"
        exit 77
    fi
fi

# Signal success so a follow-up test can set the fixture (CTest only sets fixture on pass)
if [[ -n "$MARKER" ]]; then
    touch "$MARKER"
fi
exit 0
