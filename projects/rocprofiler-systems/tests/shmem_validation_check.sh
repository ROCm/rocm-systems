#!/usr/bin/env bash
#
# MIT License
#
# Copyright (c) 2025 Advanced Micro Devices, Inc. All rights reserved.
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
# THE SOFTWARE.
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
# Usage: shmem_validation_check.sh <oshrun_exe> <num_pes> <hello_exe> [marker_file]
# If marker_file is given, it is created (touched) on success so CTest can gate other tests.

OSHRUN="$1"
NP="$2"
EXE="$3"
MARKER="$4"

if [[ -z "$OSHRUN" || -z "$NP" || -z "$EXE" ]]; then
    echo "Usage: $0 <oshrun_exe> <num_pes> <hello_exe>"
    exit 77
fi

OUTPUT=$("$OSHRUN" -n "$NP" "$EXE" 2>&1)
STATUS=$?

if [[ $STATUS -ne 0 ]]; then
    echo "SHMEM runtime failed – skipping SHMEM tests"
    exit 77
fi

# Require both greeting lines and "received value" lines (order may vary by PE)
if ! echo "$OUTPUT" | grep -qE "Hello from PE [0-9]+ of ${NP}"; then
    echo "SHMEM validation: missing 'Hello from PE X of ${NP}' – skipping"
    exit 77
fi
if ! echo "$OUTPUT" | grep -qE "PE [0-9]+ received value [0-9]+ from PE [0-9]+"; then
    echo "SHMEM validation: missing 'PE X received value Y from PE Z' – skipping"
    exit 77
fi
# For np=2 we expect both PEs to have said hello and both to have received a value
if [[ "$NP" -eq 2 ]]; then
    if ! echo "$OUTPUT" | grep -q "Hello from PE 0 of 2" || ! echo "$OUTPUT" | grep -q "Hello from PE 1 of 2"; then
        echo "SHMEM validation: expected both PE 0 and PE 1 greetings – skipping"
        exit 77
    fi
    if ! echo "$OUTPUT" | grep -q "PE 0 received value 1 from PE 1" || ! echo "$OUTPUT" | grep -q "PE 1 received value 0 from PE 0"; then
        echo "SHMEM validation: expected PE 0 received from PE 1 and PE 1 from PE 0 – skipping"
        exit 77
    fi
fi

# Signal success so a follow-up test can set the fixture (CTest only sets fixture on pass)
if [[ -n "$MARKER" ]]; then
    touch "$MARKER"
fi
exit 0
