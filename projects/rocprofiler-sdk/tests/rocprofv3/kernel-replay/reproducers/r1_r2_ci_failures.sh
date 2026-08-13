#!/usr/bin/env bash
# MIT License
#
# Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
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
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
# THE SOFTWARE.
#
# Reproducers R1 and R2: the two failures actually observed in CI on mi325 / gfx94X.
# Both are intermittent, so a single run proves nothing -- these loop and report a rate
# with a 95% Wilson interval.
#
#   R1  buffered-api-tracing aborts during final output with
#         terminate called after throwing an instance of 'std::length_error'
#           what():  basic_string::_M_create
#       Observed once on Core mi325 ubuntu-22.04, after the finalize teardown order was
#       restored. The abort happens after "Outputting collected data to ...", i.e. on the
#       finalize path, which is what makes it interesting rather than a timing flake.
#
#   R2  rocprofv3-test-hip-streams-per-thread segfaults during execution, dragging its
#       three validate tests to Not Run. Measured 3/12 on TheRock gfx94X versus 1/35 on
#       Core mi325 (two-proportion z = -2.37, p = 0.018).
#
#   ./r1_r2_ci_failures.sh --build-dir /path/to/rocprofiler-sdk/build --runs 50
#
# Run it on the OS image where the failure was seen; both are environment sensitive.

set -u

BUILD_DIR=""
RUNS=30
WHICH="both"

while [ $# -gt 0 ]; do
    case "$1" in
        --build-dir) BUILD_DIR="$2"; shift 2 ;;
        --runs)      RUNS="$2"; shift 2 ;;
        --only)      WHICH="$2"; shift 2 ;;
        -h|--help)   sed -n '25,45p' "$0"; exit 0 ;;
        *) echo "unknown argument: $1" >&2; exit 2 ;;
    esac
done

if [ -z "$BUILD_DIR" ] || [ ! -d "$BUILD_DIR" ]; then
    echo "SKIP: --build-dir must point at a configured rocprofiler-sdk build tree"
    exit 77
fi
if ! command -v ctest >/dev/null 2>&1; then
    echo "SKIP: ctest not found"; exit 77
fi

wilson() {  # hits n -> "lo hi" as percentages
    python3 - "$1" "$2" <<'PY'
import math, sys
h, n = int(sys.argv[1]), int(sys.argv[2])
if n == 0:
    print("0.0 0.0"); raise SystemExit
p = h / n; z = 1.96; d = 1 + z * z / n
c = (p + z * z / (2 * n)) / d
w = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
print(f"{max(0.0, c - w) * 100:.1f} {min(1.0, c + w) * 100:.1f}")
PY
}

loop_test() {
    local label="$1" pattern="$2" grep_for="$3"
    local fails=0 matched=0 i
    echo
    echo "=== $label: $RUNS runs of '$pattern' ==="
    for i in $(seq 1 "$RUNS"); do
        out="$(cd "$BUILD_DIR" && ctest -R "$pattern" --output-on-failure 2>&1)"
        rc=$?
        if [ "$rc" -ne 0 ]; then
            fails=$((fails + 1))
            printf '  run %3d: FAILED\n' "$i"
            if [ -n "$grep_for" ] && printf '%s' "$out" | grep -q "$grep_for"; then
                matched=$((matched + 1))
                printf '%s' "$out" | grep -m2 "$grep_for" | sed 's/^/      /'
            fi
        fi
    done
    read -r lo hi <<<"$(wilson "$fails" "$RUNS")"
    echo "  failures: $fails/$RUNS = $(python3 -c "print(f'{$fails/$RUNS:.1%}')")  95% CI ${lo}%..${hi}%"
    if [ -n "$grep_for" ]; then
        echo "  of which matched the expected signature: $matched"
    fi
    [ "$fails" -gt 0 ] && echo "  REPRODUCED" || echo "  not reproduced in $RUNS runs"
}

if [ "$WHICH" = "both" ] || [ "$WHICH" = "r1" ]; then
    loop_test "R1 buffered-api-tracing finalize abort" \
              "^buffered-api-tracing$" "length_error"
fi
if [ "$WHICH" = "both" ] || [ "$WHICH" = "r2" ]; then
    loop_test "R2 hip-streams-per-thread segfault" \
              "hip-streams-per-thread" "SegFault"
fi

cat <<'NOTES'

If R1 does not reproduce under plain ctest, run the binary directly under a sanitizer --
the symptom (basic_string::_M_create throwing length_error) is a garbage length, which
points at a use-after-free or an uninitialised size on the finalize output path:

  ASAN_OPTIONS=detect_leaks=0 <build>/bin/buffered-api-tracing

and compare a build configured with -DCMAKE_CXX_FLAGS="-fsanitize=address".
NOTES
