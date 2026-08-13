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
# Reproducers R4 and R7.
#
#   R4  An indefinite replay loop has no upper bound and no watchdog. The loop in the write
#       interceptor is `for(uint64_t pass = 0;; ++pass)`, exited only when the tool's
#       replay_continue_cb says stop. A tool that always continues never terminates, and it
#       holds the per-agent writer lock throughout, so every other dispatch on that GPU is
#       blocked behind it. A misbehaving or buggy tool therefore hangs the application rather
#       than being cut off.
#
#       Expected on a bounded implementation: the loop stops at some cap, or a watchdog fires
#       with a diagnostic. Observed: runs until the external timeout kills it.
#
#   R7  Replay_Pass reaches the JSON output but is not emitted in counter_collection.csv, so
#       a CSV consumer cannot tell the passes apart and N groups collapse into rows that look
#       like duplicates.
#
#   ./r4_r7_bounded_and_csv.sh --app ./your_hip_app --client ./librepro_client.so
#
set -u

APP=""
APP_ARGS=""
CLIENT=""
ROCPROFV3="${ROCPROFV3:-$(command -v rocprofv3 || echo rocprofv3)}"
TIMEOUT_S=60
OUT="${OUT:-repro-r4-r7-out}"
RC=0

while [ $# -gt 0 ]; do
    case "$1" in
        --app)       APP="$2"; shift 2 ;;
        --args)      APP_ARGS="$2"; shift 2 ;;
        --client)    CLIENT="$2"; shift 2 ;;
        --rocprofv3) ROCPROFV3="$2"; shift 2 ;;
        --timeout)   TIMEOUT_S="$2"; shift 2 ;;
        --out)       OUT="$2"; shift 2 ;;
        -h|--help)   sed -n '25,45p' "$0"; exit 0 ;;
        *) echo "unknown argument: $1" >&2; exit 2 ;;
    esac
done

if [ -z "$APP" ] || [ ! -x "$APP" ]; then
    echo "SKIP: --app must point at a HIP executable"; exit 77
fi
mkdir -p "$OUT"

echo "=== R4: indefinite replay must terminate on its own ==="
if [ -z "$CLIENT" ] || [ ! -f "$CLIENT" ]; then
    echo "  SKIP: --client must point at librepro_client.so (see build.sh)"
else
    echo "  running with KR_REPRO_MODE=indefinite under a ${TIMEOUT_S}s timeout"
    # shellcheck disable=SC2086
    KR_REPRO_MODE=indefinite LD_PRELOAD="$CLIENT" \
        timeout --signal=KILL "$TIMEOUT_S" "$APP" $APP_ARGS >"$OUT/r4.log" 2>&1
    rc=$?
    passes=$(grep -c "repro-client\] pass " "$OUT/r4.log" 2>/dev/null || echo 0)
    if [ "$rc" -eq 137 ]; then
        echo "  REPRODUCED: killed by the external timeout after $passes replay passes."
        echo "  The loop never bounded itself and held the agent writer lock the whole time."
        RC=1
    elif [ "$rc" -eq 0 ]; then
        echo "  terminated on its own after $passes passes -- a bound exists."
    else
        echo "  exited rc=$rc after $passes passes; check $OUT/r4.log"
        tail -5 "$OUT/r4.log" | sed 's/^/      /'
    fi
fi

echo
echo "=== R7: Replay_Pass missing from counter_collection.csv ==="
if ! command -v "$ROCPROFV3" >/dev/null 2>&1 && [ ! -x "$ROCPROFV3" ]; then
    echo "  SKIP: rocprofv3 not found ($ROCPROFV3)"
else
    G1="SQ_WAVES SQ_INSTS_VALU GRBM_COUNT"
    G2="SQ_WAVES SQ_INSTS_VALU GRBM_GUI_ACTIVE"
    # shellcheck disable=SC2086
    "$ROCPROFV3" --pmc $G1 --pmc $G2 --kernel-replay-beta-enabled \
        --output-format csv json -d "$OUT/r7" -o out -- "$APP" $APP_ARGS \
        >"$OUT/r7.log" 2>&1
    csv=$(find "$OUT/r7" -name '*counter_collection.csv' 2>/dev/null | head -1)
    json=$(find "$OUT/r7" -name '*results.json' 2>/dev/null | head -1)
    if [ -z "$csv" ]; then
        echo "  no counter_collection.csv produced; see $OUT/r7.log"
    else
        header=$(head -1 "$csv")
        in_json="no"
        [ -n "$json" ] && grep -q -e '"replay_pass"' -e '"n"' "$json" 2>/dev/null && in_json="yes"
        if printf '%s' "$header" | grep -qi "replay_pass"; then
            echo "  Replay_Pass IS in the CSV header -- gap closed."
        else
            echo "  REPRODUCED: present in JSON=$in_json, absent from the CSV header:"
            printf '      %s\n' "$(printf '%s' "$header" | cut -c1-150)"
            rows=$(tail -n +2 "$csv" | wc -l)
            uniq_rows=$(tail -n +2 "$csv" | sort -u | wc -l)
            echo "      $rows data rows, $uniq_rows distinct -- passes are indistinguishable"
            RC=1
        fi
    fi
fi

echo
echo "exit $RC (1 = at least one reproduced)"
exit "$RC"
