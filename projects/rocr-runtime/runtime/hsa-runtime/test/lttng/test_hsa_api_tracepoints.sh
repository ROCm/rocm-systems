#!/usr/bin/env bash
# End-to-end validation of the rocm_hsa LTTng-UST tracepoint provider.
#
# Spins up a per-user lttng-sessiond, enables the rocm_hsa:* events,
# runs a tiny HSA program against the freshly-built libhsa-runtime64.so,
# and asserts the expected events show up in the babeltrace2 output.
#
# Usage:
#   test_hsa_api_tracepoints.sh [<build_dir>]
#
# <build_dir>  defaults to $PWD/build/rocr (cmake -S projects/rocr-runtime -B build/rocr).
#
# Exits 0 on PASS, non-zero on any assertion failure.
set -euo pipefail

BUILD_DIR="${1:-$PWD/build/rocr}"
LIB_DIR="$BUILD_DIR/rocr/lib"

if [ ! -f "$LIB_DIR/libhsa-runtime64.so" ]; then
    echo "FAIL: $LIB_DIR/libhsa-runtime64.so not found"
    exit 1
fi

TRACE_DIR="$(mktemp -d)"
cleanup() {
    # Disable errexit inside the trap so a kill of an already-exited
    # daemon (or any other cleanup hiccup) doesn't override an
    # otherwise-passing test's exit status.
    set +e
    pkill -P $$ 2>/dev/null
    # Scoped sessiond kill: only this test's daemon (started below with
    # a pidfile and isolated LTTNG_HOME). Avoid host-wide pkill -f which
    # would terminate any concurrent test's lttng-sessiond.
    if [ -n "${SESSIOND_PIDFILE:-}" ] && [ -f "$SESSIOND_PIDFILE" ]; then
        kill "$(cat "$SESSIOND_PIDFILE")" 2>/dev/null || true
    fi
    rm -rf "$TRACE_DIR"
}
trap cleanup EXIT

# Tiny HSA program that exercises a handful of public APIs.
cat > "$TRACE_DIR/tiny.c" <<'EOF'
#include <hsa/hsa.h>
#include <stdio.h>
int main(void) {
    hsa_status_t st = hsa_init();
    if (st != HSA_STATUS_SUCCESS) {
        fprintf(stderr, "hsa_init failed: %d\n", st);
        return 1;
    }
    uint16_t major = 0;
    hsa_system_get_info(HSA_SYSTEM_INFO_VERSION_MAJOR, &major);
    hsa_shut_down();
    return 0;
}
EOF

cc "$TRACE_DIR/tiny.c" -o "$TRACE_DIR/tiny" \
    -I/opt/rocm/include -L"$LIB_DIR" -lhsa-runtime64 \
    -Wl,-rpath,"$LIB_DIR"

# Optional: a HIP dispatch test to exercise hsa_doorbell_ring with
# packet_type = KERNEL_DISPATCH. Built only if hipcc + a sample exist
# under /opt/rocm/share/hip/samples. If absent, the doorbell assertion
# is skipped and reported.
HIP_DISPATCH_BIN=""
if command -v /opt/rocm/bin/hipcc >/dev/null 2>&1 \
   && [ -f /opt/rocm/share/hip/samples/0_Intro/square/square.hipref.cpp ]; then
    cp /opt/rocm/share/hip/samples/0_Intro/square/square.hipref.cpp "$TRACE_DIR/square.cpp"
    if /opt/rocm/bin/hipcc "$TRACE_DIR/square.cpp" -o "$TRACE_DIR/square" \
        >"$TRACE_DIR/hipcc.log" 2>&1; then
        HIP_DISPATCH_BIN="$TRACE_DIR/square"
    else
        echo "INFO: hipcc compile failed; doorbell assertion will be skipped."
    fi
fi

fail() { echo "FAIL: $*" >&2; exit 1; }

# Start a per-user sessiond, isolated to this test's LTTNG_HOME and
# tracked by a pidfile so cleanup can target THIS test's daemon only.
export LTTNG_HOME="$TRACE_DIR/lttng-home"
mkdir -p "$LTTNG_HOME"
SESSIOND_PIDFILE="$TRACE_DIR/sessiond.pid"
lttng-sessiond --daemonize --no-kernel --pidfile="$SESSIOND_PIDFILE"
for i in 1 2 3 4 5; do
    [ -f "$SESSIOND_PIDFILE" ] && break
    sleep 0.5
done
# (cleanup trap already set above, uses SESSIOND_PIDFILE)

# ============================================================
# Run 1: API events (tiny standalone HSA program)
# ============================================================
SESSION_API="hsa-api-test-$$"
lttng create "$SESSION_API" --output="$TRACE_DIR/trace_api" >/dev/null
# Discard policy + conservative sub-buffer sizing: container hosts with
# /dev/shm limited to 64 MiB and ~224 CPUs need a much smaller channel
# than LTTng's default (524288 B x 4 sub-buffers per CPU). 4 KiB x 2 fits.
lttng enable-channel --userspace --discard --subbuf-size=4096 --num-subbuf=2 default >/dev/null
lttng enable-event --userspace -c default 'rocm_hsa:hsa_api_enter,rocm_hsa:hsa_api_exit_status,rocm_hsa:hsa_api_exit_ptr,rocm_hsa:hsa_api_exit_void' >/dev/null
lttng start "$SESSION_API" >/dev/null

LD_LIBRARY_PATH="$LIB_DIR:${LD_LIBRARY_PATH:-}" "$TRACE_DIR/tiny" || true

lttng stop "$SESSION_API" >/dev/null
lttng destroy "$SESSION_API" >/dev/null

LOG_API="$TRACE_DIR/babeltrace_api.log"
babeltrace2 "$TRACE_DIR/trace_api" > "$LOG_API" 2>/dev/null

EXPECTED_APIS=(hsa_init hsa_shut_down)
for api in "${EXPECTED_APIS[@]}"; do
    grep -q "rocm_hsa:hsa_api_enter:.*api_name = \"$api\"" "$LOG_API" \
        || fail "missing enter event for $api (api session)"
    grep -q "rocm_hsa:hsa_api_exit_status:.*api_name = \"$api\"" "$LOG_API" \
        || fail "missing exit_status event for $api (api session)"
done

N_API_EVENTS=$(wc -l < "$LOG_API")
N_ENTER=$(grep -c 'rocm_hsa:hsa_api_enter' "$LOG_API" || true)
N_EXIT=$(grep -c 'rocm_hsa:hsa_api_exit_status' "$LOG_API" || true)

# ============================================================
# Run 2: doorbell events (HIP dispatch — only if available)
# ============================================================
N_DOORBELL=0
DOORBELL_DETAIL=""
if [ -n "$HIP_DISPATCH_BIN" ]; then
    SESSION_DB="hsa-doorbell-test-$$"
    lttng create "$SESSION_DB" --output="$TRACE_DIR/trace_db" >/dev/null
    lttng enable-channel --userspace --discard --subbuf-size=4096 --num-subbuf=2 default >/dev/null
    # Only enable doorbell + intercept events to keep volume low and avoid
    # /dev/shm-sized buffer overflow.
    lttng enable-event --userspace -c default 'rocm_hsa:hsa_doorbell_ring,rocm_hsa:hsa_intercept_packets' >/dev/null
    lttng start "$SESSION_DB" >/dev/null

    LD_LIBRARY_PATH="$LIB_DIR:/opt/rocm/lib:${LD_LIBRARY_PATH:-}" "$HIP_DISPATCH_BIN" \
        > "$TRACE_DIR/dispatch.log" 2>&1 || true

    lttng stop "$SESSION_DB" >/dev/null
    lttng destroy "$SESSION_DB" >/dev/null

    LOG_DB="$TRACE_DIR/babeltrace_db.log"
    babeltrace2 "$TRACE_DIR/trace_db" > "$LOG_DB" 2>/dev/null

    grep -q 'rocm_hsa:hsa_doorbell_ring' "$LOG_DB" \
        || fail "no doorbell-ring tracepoint observed despite dispatch binary"
    grep -q 'rocm_hsa:hsa_doorbell_ring.*packet_type = 0' "$LOG_DB" \
        || fail "no KERNEL_DISPATCH-typed doorbell event observed"

    N_DOORBELL=$(grep -c 'rocm_hsa:hsa_doorbell_ring' "$LOG_DB" || true)
    DOORBELL_DETAIL=$(grep 'rocm_hsa:hsa_doorbell_ring.*packet_type = 0' "$LOG_DB" | head -1)
fi

# ============================================================
# Summary
# ============================================================
echo "PASS hsa_api_tracepoints: api_events=$N_API_EVENTS enter=$N_ENTER exit_status=$N_EXIT doorbell=$N_DOORBELL"
if [ -n "$DOORBELL_DETAIL" ]; then
    echo "  doorbell sample: $DOORBELL_DETAIL"
fi
