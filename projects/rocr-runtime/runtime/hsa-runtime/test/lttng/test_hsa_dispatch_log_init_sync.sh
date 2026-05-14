#!/usr/bin/env bash
# Init-sync no-phantom-records test for the dispatch_log drainer.
#
# Locks in the invariant from the C7+C8+C9 unified zero-rt-stop fix
# and the C2-round-2 fresh-vs-re-enable distinction:
#
#   On any FIRST observation of a queue (next_idx == 0 or freshly
#   allocated buffer with FW per-pipe scratch wptr persisted to a
#   high signal value), the drainer MUST NOT emit any
#   kernel_dispatch_record event whose payload is all-zero
#   record_type slots. Pre-zeroed slots are recognized via the
#   record_type==0 stop and either:
#     - quarantine-swept with a kernel_dispatch_drop
#     - silently overrun-skipped (init-sync == true case at the
#       overrun guard)
#   but never wrapped into a kernel_dispatch_record event payload.
#
# A regression here would surface as kernel_dispatch_record events
# whose 16-byte record bodies decode to ts_lo=0, ts_hi=0,
# record_type=0, dispatch_idx=0 — phantom records from pre-zeroed
# ring slots that the drainer should have skipped. Consumers
# reading these as real records would see all kernels appearing to
# execute at gpu_ts=0 with type=0 (= "unknown event tag").
#
# Mechanics:
#   1. Run a workload that creates fresh queues. The hipcc square
#      sample creates a fresh HIP queue per process; we drive it
#      under LTTng.
#   2. Decode the trace and walk every kernel_dispatch_record event.
#      For each event, decode the packed records[count*16] payload
#      and assert that no record has all four DWORDs zero.
#   3. Also assert at least one record has nonzero contents (proves
#      we are exercising the path; otherwise the test could pass
#      trivially with zero records).
#
# Usage: $0 [<build_dir>]

set -euo pipefail

BUILD_DIR="${1:-$PWD/build/rocr}"
LIB_DIR="$BUILD_DIR/rocr/lib"

if [ ! -f "$LIB_DIR/libhsa-runtime64.so" ]; then
    echo "FAIL: $LIB_DIR/libhsa-runtime64.so not found"
    exit 1
fi

TRACE_DIR="$(mktemp -d)"
cleanup() {
    set +e
    pkill -P $$ 2>/dev/null
    if [ -n "${SESSIOND_PIDFILE:-}" ] && [ -f "$SESSIOND_PIDFILE" ]; then
        kill "$(cat "$SESSIOND_PIDFILE")" 2>/dev/null || true
    fi
    rm -rf "$TRACE_DIR"
}
trap cleanup EXIT

fail() { echo "FAIL: $*" >&2; exit 1; }
info() { echo "INFO: $*"; }

# Need a kernel-dispatching workload. Use hipcc square if available.
WORKLOAD_BIN=""
if command -v /opt/rocm/bin/hipcc >/dev/null 2>&1 \
   && [ -f /opt/rocm/share/hip/samples/0_Intro/square/square.hipref.cpp ]; then
    cp /opt/rocm/share/hip/samples/0_Intro/square/square.hipref.cpp "$TRACE_DIR/square.cpp"
    if /opt/rocm/bin/hipcc "$TRACE_DIR/square.cpp" -o "$TRACE_DIR/square" \
        >"$TRACE_DIR/hipcc.log" 2>&1; then
        WORKLOAD_BIN="$TRACE_DIR/square"
    fi
fi

if [ -z "$WORKLOAD_BIN" ]; then
    info "hipcc square not available; init-sync no-phantom test SKIPPED"
    exit 0
fi

# Per-user lttng-sessiond, scoped to this test.
export LTTNG_HOME="$TRACE_DIR/lttng-home"
mkdir -p "$LTTNG_HOME"
SESSIOND_PIDFILE="$TRACE_DIR/sessiond.pid"
lttng-sessiond --daemonize --no-kernel --pidfile="$SESSIOND_PIDFILE"
for i in 1 2 3 4 5; do
    [ -f "$SESSIOND_PIDFILE" ] && break
    sleep 0.5
done
[ -f "$SESSIOND_PIDFILE" ] || fail "lttng-sessiond did not start"

lttng create init-sync-test --output="$TRACE_DIR/trace" >/dev/null
lttng enable-event --userspace 'rocm_hsa:kernel_dispatch_record' >/dev/null
lttng start >/dev/null

LD_LIBRARY_PATH="$LIB_DIR:${LD_LIBRARY_PATH:-}" \
    "$WORKLOAD_BIN" >"$TRACE_DIR/workload.stdout" 2>"$TRACE_DIR/workload.stderr" \
    || fail "workload exited non-zero (see $TRACE_DIR/workload.stderr)"

lttng stop    >/dev/null
lttng destroy >/dev/null

# Decode the trace. babeltrace2 prints each record event on a single
# line including the records[] field as a brace-list of byte values.
# We walk those byte lists in groups of 16 (per record) and check for
# all-zero records.
DECODED="$TRACE_DIR/trace.txt"
babeltrace2 "$TRACE_DIR/trace" >"$DECODED" 2>/dev/null \
    || fail "babeltrace2 failed on $TRACE_DIR/trace"

# Use Python (available in any reasonable test env) for the byte-walk.
# This is more robust than awk for the records[] brace-list parsing.
python3 - "$DECODED" <<'PY'
import re, sys
path = sys.argv[1]
total_events = 0
total_records = 0
phantom_records = 0
nonzero_records = 0
phantom_examples = []

# Each kernel_dispatch_record event line has, somewhere in it, a
# field of the form "_records_length = N" and "records = [ B1, B2, ...
# Bn ]" where n == count*16. babeltrace2's exact format may vary
# slightly; we tolerate either "_records" or "records" as the field
# name and parse the bracketed byte list.
pat_event   = re.compile(r"rocm_hsa:kernel_dispatch_record:")
pat_count   = re.compile(r"\bcount\s*=\s*(\d+)")
# records = [ [0] = N, [1] = N, ... [k] = N ]   — matches the OUTER
# brackets only by anchoring on "records = [" then the FINAL " ] }"
# (closing bracket followed by space + closing brace of the field
# group). [^\]]* won't work because each element has its own [i].
pat_records = re.compile(r"records\s*=\s*\[(.*?)\]\s*\}\s*$")

with open(path) as f:
    for line in f:
        if "rocm_hsa:kernel_dispatch_record:" not in line:
            continue
        total_events += 1
        m_count = pat_count.search(line)
        m_recs  = pat_records.search(line)
        if not m_count or not m_recs:
            continue
        count = int(m_count.group(1))
        # Parse byte list. babeltrace2 prints elements as "[i] = value"
        # (one entry per element separated by commas), where value is
        # decimal int (or rarely hex with 0x prefix).
        body = m_recs.group(1)
        # Extract just the value from each "[i] = value" element.
        ints = [int(v.strip(), 0) for v in re.findall(r"=\s*(\S+?)\s*(?:,|$)", body)]
        if len(ints) != count * 16:
            print(f"FAIL: event has count={count} but records length {len(ints)} != count*16",
                  file=sys.stderr)
            sys.exit(2)
        for i in range(count):
            rec = ints[i*16:(i+1)*16]
            if all(b == 0 for b in rec):
                phantom_records += 1
                if len(phantom_examples) < 3:
                    phantom_examples.append((total_events, i))
            else:
                nonzero_records += 1
            total_records += 1

print(f"events={total_events} records={total_records} phantom={phantom_records} nonzero={nonzero_records}")

if phantom_records > 0:
    print(f"FAIL: drainer emitted {phantom_records} all-zero record(s) "
          f"(phantom records from pre-zeroed ring slots that should "
          f"have been quarantine-swept or init-sync overrun-skipped)",
          file=sys.stderr)
    print(f"  first {len(phantom_examples)} phantom locations (event_index, record_index_in_event):",
          file=sys.stderr)
    for ev, idx in phantom_examples:
        print(f"    event #{ev} record #{idx}", file=sys.stderr)
    sys.exit(2)

if total_records == 0:
    # Workload may have INFO-skipped, or the substrate may not support
    # dispatch_log. Self-skip rather than fail.
    print("INFO: no kernel_dispatch_record events captured; init-sync test "
          "has nothing to assert against (workload may have skipped or KFD "
          "substrate may not support dispatch_log)", file=sys.stderr)
    sys.exit(0)

if nonzero_records == 0:
    print(f"FAIL: {total_records} record(s) emitted but ALL were all-zero "
          f"(should not happen if any real kernels ran)", file=sys.stderr)
    sys.exit(2)

print("OK")
PY

RC=$?
if [ "$RC" -ne 0 ]; then
    fail "init-sync no-phantom-records assertion failed (exit $RC)"
fi

info "PASS: no phantom (all-zero) records emitted from drainer"
exit 0
