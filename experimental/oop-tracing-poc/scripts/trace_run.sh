#!/usr/bin/env bash
# Local POC: run the app with LTTng recording runtime tracepoints and dlogc draining the CP
# dispatch log of the same process. Runs inside the container. Usage: trace_run.sh <out> [iters]
set -u
OUT=${1:?out dir}; ITERS=${2:-20}
rm -rf "$OUT" && mkdir -p "$OUT"
pgrep -x lttng-sessiond >/dev/null || { lttng-sessiond --daemonize >/dev/null 2>&1; sleep 1; }
lttng create run --output="$OUT/lttng" >/dev/null
lttng enable-event -u "rocm_hip:*" >/dev/null
lttng add-context -u -t vpid -t vtid >/dev/null
lttng start >/dev/null
GO="$OUT/go"
LD_LIBRARY_PATH=/w/clr-build/hipamd/lib /w/app "$ITERS" "$GO" > "$OUT/app.out" 2>&1 &
APP=$!
until grep -q waiting "$OUT/app.out" 2>/dev/null; do sleep 0.05; done
/w/dlogc "$APP" "$OUT" > "$OUT/dlogc.out" 2> "$OUT/dlogc.err" &
DLOGC=$!
until grep -q ready "$OUT/dlogc.out" 2>/dev/null || ! kill -0 $DLOGC 2>/dev/null; do sleep 0.05; done
touch "$GO"
wait $APP; echo "app exit=$?"
wait $DLOGC; echo "dlogc exit=$?"
lttng stop >/dev/null; lttng destroy >/dev/null
cat "$OUT/app.out"; cat "$OUT/dlogc.err"
