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
# rocprofv3 kernel-replay workflows a user would actually run. Each case prints its
# command, runs it, and states what to look for. Non-zero exit means a case that should
# work did not, or a known gap reproduced.
#
#   ./rocprofv3_workflows.sh --app ./kernel-replay --args "1048576 1"

set -u

ROCPROFV3="${ROCPROFV3:-$(command -v rocprofv3 || echo rocprofv3)}"
APP=""
APP_ARGS=""
OUT="${OUT:-rocprofv3-workflows-out}"
FAILURES=0
GAPS=0

while [ $# -gt 0 ]; do
    case "$1" in
        --rocprofv3) ROCPROFV3="$2"; shift 2 ;;
        --app)       APP="$2"; shift 2 ;;
        --args)      APP_ARGS="$2"; shift 2 ;;
        --out)       OUT="$2"; shift 2 ;;
        -h|--help)   sed -n '24,32p' "$0"; exit 0 ;;
        *) echo "unknown argument: $1" >&2; exit 2 ;;
    esac
done

if ! command -v "$ROCPROFV3" >/dev/null 2>&1 && [ ! -x "$ROCPROFV3" ]; then
    echo "SKIP: rocprofv3 not found ($ROCPROFV3)"; exit 77
fi
if [ -z "$APP" ] || [ ! -x "$APP" ]; then
    echo "SKIP: --app must point at an executable (got '${APP}')"; exit 77
fi

mkdir -p "$OUT"
G1="SQ_WAVES SQ_INSTS_VALU GRBM_COUNT"
G2="SQ_WAVES SQ_INSTS_VALU GRBM_GUI_ACTIVE"
G3="SQ_WAVES SQ_INSTS_VALU SQ_INSTS_SALU"

banner() { echo; echo "=== $* ==="; }

# shellcheck disable=SC2086
run_case() {
    local name="$1"; shift
    echo "  \$ $*"
    if "$@" >"$OUT/$name.log" 2>&1; then
        echo "  ok"
        return 0
    fi
    echo "  FAILED (rc=$?), tail of log:"
    tail -8 "$OUT/$name.log" | sed 's/^/      /'
    return 1
}

banner "W1: more counters than fit one hardware pass (the headline use case)"
echo "  Three groups collected in a single application run. Without replay this needs"
echo "  three separate runs of the whole application."
# shellcheck disable=SC2086
run_case w1 "$ROCPROFV3" --pmc $G1 --pmc $G2 --pmc $G3 \
    --kernel-replay-beta-enabled --output-format json \
    -d "$OUT/w1" -o out -- "$APP" $APP_ARGS || FAILURES=$((FAILURES+1))

banner "W2: replay one hot kernel in a larger application"
echo "  --kernel-include-regex narrows replay to a single kernel; every other dispatch"
echo "  opts out through pass_count_cb returning 1 and runs single-pass."
# shellcheck disable=SC2086
run_case w2 "$ROCPROFV3" --pmc $G1 --pmc $G2 \
    --kernel-replay-beta-enabled --kernel-include-regex "vecAdd" \
    --output-format json -d "$OUT/w2" -o out -- "$APP" $APP_ARGS || FAILURES=$((FAILURES+1))

banner "W3: CSV output -- known gap, Replay_Pass is not emitted"
echo "  JSON carries the per-pass index; counter_collection.csv does not, so a CSV"
echo "  consumer cannot separate the groups and rows collapse."
# shellcheck disable=SC2086
run_case w3 "$ROCPROFV3" --pmc $G1 --pmc $G2 \
    --kernel-replay-beta-enabled --output-format csv json \
    -d "$OUT/w3" -o out -- "$APP" $APP_ARGS || FAILURES=$((FAILURES+1))
csv=$(find "$OUT/w3" -name '*counter_collection.csv' 2>/dev/null | head -1)
if [ -n "$csv" ]; then
    if head -1 "$csv" | grep -qi "replay_pass"; then
        echo "  Replay_Pass IS present in the CSV header -- the gap is closed"
    else
        echo "  Replay_Pass absent from CSV header (expected gap):"
        echo "      $(head -1 "$csv" | cut -c1-160)"
        GAPS=$((GAPS+1))
    fi
else
    echo "  no counter_collection.csv produced"
fi

banner "W4: replay combined with kernel tracing"
echo "  Replay is a callback-tracing domain, so it should compose with other services"
echo "  rather than displace them. Expect both counter records and kernel trace rows."
# shellcheck disable=SC2086
run_case w4 "$ROCPROFV3" --pmc $G1 --pmc $G2 --kernel-trace \
    --kernel-replay-beta-enabled --output-format json \
    -d "$OUT/w4" -o out -- "$APP" $APP_ARGS || FAILURES=$((FAILURES+1))

banner "W5: --stats with replay active"
echo "  Unspecified today: does a replayed dispatch count once or N times in the"
echo "  statistics table? Inspect the output and decide which is intended."
# shellcheck disable=SC2086
run_case w5 "$ROCPROFV3" --pmc $G1 --pmc $G2 --stats \
    --kernel-replay-beta-enabled --output-format json \
    -d "$OUT/w5" -o out -- "$APP" $APP_ARGS || FAILURES=$((FAILURES+1))

banner "W6: replay requested without --pmc"
echo "  Replay exists to collect counters; asking for it with no counters should be a"
echo "  clear diagnostic, not a silent no-op or a crash."
if "$ROCPROFV3" --kernel-replay-beta-enabled --output-format json \
        -d "$OUT/w6" -o out -- "$APP" $APP_ARGS >"$OUT/w6.log" 2>&1; then
    echo "  accepted with no --pmc; check whether that is intended:"
    tail -4 "$OUT/w6.log" | sed 's/^/      /'
else
    echo "  rejected, as expected. Message:"
    tail -4 "$OUT/w6.log" | sed 's/^/      /'
fi

echo
echo "=== summary ==="
echo "  failures (cases that should work but did not): $FAILURES"
echo "  known gaps reproduced:                        $GAPS"
echo "  outputs and logs under $OUT"
[ "$FAILURES" -eq 0 ] || exit 1
exit 0
