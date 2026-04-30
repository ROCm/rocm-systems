#!/usr/bin/env bash
# HSA LTTng invariant tests.
#
# Asserts semantic invariants beyond the symbol smoke checks in
# test_hsa_api_tracepoints.sh:
#
#   I1. Enter/exit balance: count(hsa_api_enter) ==
#         sum(_exit_status, _ptr, _void, _u64, _i64). Catches missed exit
#         emits (especially for pure-void no-return wrappers).
#
#   I2. 64-bit return preservation: after submitting N packets with
#         hsa_queue_add_write_index_relaxed, the next
#         hsa_queue_load_write_index_relaxed must produce a
#         hsa_api_exit_u64 event whose retval matches the expected value
#         exactly. Also requires at least one hsa_api_exit_i64 from a
#         signal_load call, proving _i64 path works.
#
#   I3. (Schema v2 era only — doorbell corr_id uniqueness — removed in
#       schema v3 since corr_id is no longer carried as an event field.
#       The HIP invariants test now asserts vtid-based dispatch-chain
#       affinity, which serves the same purpose.)
#
# Usage:
#   test_hsa_invariants.sh [<build_dir>]
#
# <build_dir>  defaults to $PWD/build/rocr.
set -euo pipefail

BUILD_DIR="${1:-$PWD/build/rocr}"
LIB_DIR="$BUILD_DIR/rocr/lib"
if [ ! -f "$LIB_DIR/libhsa-runtime64.so" ]; then
    echo "FAIL: $LIB_DIR/libhsa-runtime64.so not found"
    exit 1
fi

WORK="$(mktemp -d)"
SESSION="hsa-inv-$$"
SESSIOND_PIDFILE="$WORK/sessiond.pid"

# Per-test isolated LTTng state directory so lttng-sessiond's runtime
# socket / per-user state is scoped to this test's WORK dir and cannot
# collide with other tests' sessiond instances on the same host.
export LTTNG_HOME="$WORK/lttng-home"
mkdir -p "$LTTNG_HOME"

cleanup() {
    set +e
    lttng destroy "$SESSION" >/dev/null 2>&1
    # Kill ONLY the sessiond instance this test started, captured by
    # --pidfile. Do NOT use `pkill -f 'lttng-sessiond'` — that would
    # match every lttng-sessiond on the host (other concurrent tests).
    if [ -s "$SESSIOND_PIDFILE" ]; then
        kill "$(cat "$SESSIOND_PIDFILE")" 2>/dev/null
    fi
    rm -rf "$WORK"
}
trap cleanup EXIT

# Test program: exercise the queue-load and queue-add index paths with
# known values so I2 can hard-assert exact retval preservation.
#
# Sequence:
#   1. hsa_init
#   2. find a GPU agent + a queue-capable AQL queue
#   3. create the queue (size=64, type=multi)
#   4. hsa_queue_load_write_index_relaxed -> expect 0 (queue is empty)
#   5. hsa_queue_add_write_index_relaxed(queue, 5) -> expect 0 (previous)
#   6. hsa_queue_load_write_index_relaxed -> expect 5
#   7. hsa_signal_load_relaxed -> drives the exit_i64 path
#   8. cleanup
cat > "$WORK/tiny.c" <<'EOF'
#include <hsa/hsa.h>
#include <stdio.h>

static hsa_status_t find_gpu_iter(hsa_agent_t agent, void* data) {
    hsa_device_type_t dev_type;
    if (hsa_agent_get_info(agent, HSA_AGENT_INFO_DEVICE, &dev_type) == HSA_STATUS_SUCCESS
        && dev_type == HSA_DEVICE_TYPE_GPU) {
        *(hsa_agent_t*)data = agent;
        return HSA_STATUS_INFO_BREAK;
    }
    return HSA_STATUS_SUCCESS;
}

int main(void) {
    hsa_status_t st = hsa_init();
    if (st != HSA_STATUS_SUCCESS) {
        fprintf(stderr, "hsa_init failed: %d\n", st);
        return 1;
    }
    hsa_agent_t gpu = {0};
    hsa_iterate_agents(find_gpu_iter, &gpu);
    if (gpu.handle == 0) {
        fprintf(stderr, "no GPU agent found\n");
        hsa_shut_down();
        return 2;
    }
    hsa_queue_t* q = NULL;
    st = hsa_queue_create(gpu, 64, HSA_QUEUE_TYPE_MULTI,
                          NULL, NULL, 0, 0, &q);
    if (st != HSA_STATUS_SUCCESS || q == NULL) {
        fprintf(stderr, "hsa_queue_create failed: %d\n", st);
        hsa_shut_down();
        return 3;
    }
    /* Sequence: should produce exit_u64 events with retval 0, 0, 5. */
    uint64_t r1 = hsa_queue_load_write_index_relaxed(q);  /* expect 0 */
    uint64_t r2 = hsa_queue_add_write_index_relaxed(q, 5); /* expect 0 (previous) */
    uint64_t r3 = hsa_queue_load_write_index_relaxed(q);  /* expect 5 */
    fprintf(stderr, "expected 0,0,5; got %lu,%lu,%lu\n",
            (unsigned long)r1, (unsigned long)r2, (unsigned long)r3);
    /* Drive exit_i64 path. */
    hsa_signal_t sig;
    if (hsa_signal_create(0, 0, NULL, &sig) == HSA_STATUS_SUCCESS) {
        hsa_signal_value_t v = hsa_signal_load_relaxed(sig);
        (void)v;
        hsa_signal_destroy(sig);
    }
    hsa_queue_destroy(q);
    hsa_shut_down();
    return 0;
}
EOF

cc "$WORK/tiny.c" -o "$WORK/tiny" \
    -I/opt/rocm/include -L"$LIB_DIR" -lhsa-runtime64 \
    -Wl,-rpath,"$LIB_DIR"

lttng-sessiond --daemonize --no-kernel --pidfile="$SESSIOND_PIDFILE"
# Wait for the pidfile to appear (--daemonize forks before writing it).
for _ in $(seq 1 20); do
    [ -s "$SESSIOND_PIDFILE" ] && break
    sleep 0.1
done
lttng create "$SESSION" --output="$WORK/trace" >/dev/null
lttng enable-channel --userspace --discard --subbuf-size=8192 --num-subbuf=2 default >/dev/null
# Per schema v3 the events themselves no longer carry corr_id / parent_corr_id
# / tid; per-event identity comes from channel-context vpid + vtid plus the
# CTF event-header timestamp. Attach those contexts to the channel.
lttng add-context --userspace --channel=default --type=vpid --type=vtid >/dev/null
lttng enable-event --userspace -c default \
    'rocm_hsa:hsa_api_enter,rocm_hsa:hsa_api_exit_status,rocm_hsa:hsa_api_exit_ptr,rocm_hsa:hsa_api_exit_void,rocm_hsa:hsa_api_exit_u64,rocm_hsa:hsa_api_exit_i64,rocm_hsa:hsa_doorbell_ring' >/dev/null
lttng start >/dev/null

LD_LIBRARY_PATH="$LIB_DIR:${LD_LIBRARY_PATH:-}" "$WORK/tiny" || true

lttng stop >/dev/null
lttng destroy "$SESSION" >/dev/null

LOG="$WORK/babeltrace.log"
babeltrace2 "$WORK/trace" > "$LOG" 2>&1

FAIL=0

# ---- I1: enter/exit balance --------------------------------------------------
N_ENTER=$(grep -c 'rocm_hsa:hsa_api_enter:' "$LOG" || true)
N_EXIT=$(grep -cE 'rocm_hsa:hsa_api_exit_(status|ptr|void|u64|i64):' "$LOG" || true)
if [ "$N_ENTER" -ne "$N_EXIT" ]; then
    echo "  I1 FAIL: enter=$N_ENTER, exit=$N_EXIT" >&2
    FAIL=1
else
    echo "  I1 OK: enter==exit ($N_ENTER each)"
fi

# ---- I2: 64-bit return preservation (HARD ASSERTION) ------------------------
# After C4, queue-load/add index ops emit hsa_api_exit_u64 with the actual
# uint64 retval. The test program issues a known sequence and we assert
# the captured retval values match the expected sequence (0, 0, 5).
N_I64=$(grep -c 'rocm_hsa:hsa_api_exit_i64:' "$LOG" || true)
if [ "$N_I64" -lt 1 ]; then
    echo "  I2 FAIL: hsa_signal_load_relaxed must emit hsa_api_exit_i64; got $N_I64" >&2
    FAIL=1
else
    echo "  I2a OK: $N_I64 exit_i64 events (signal load preserved as int64)"
fi

# Pull the retvals from exit_u64 events for hsa_queue_{load,add}_write_index_relaxed.
# Order by event timestamp (which is babeltrace2's natural output order).
EXPECTED="0 0 5"
ACTUAL=$(grep -E 'rocm_hsa:hsa_api_exit_u64:.*api_name = "(hsa_queue_load_write_index_relaxed|hsa_queue_add_write_index_relaxed)"' "$LOG" \
    | sed -n 's/.*retval = \([0-9]\+\).*/\1/p' \
    | tr '\n' ' ' | sed 's/ $//')
if [ -z "$ACTUAL" ]; then
    echo "  I2b FAIL: no hsa_api_exit_u64 events captured for queue index ops" >&2
    FAIL=1
elif [ "$ACTUAL" != "$EXPECTED" ]; then
    echo "  I2b FAIL: queue-index retval sequence mismatch — expected '$EXPECTED', got '$ACTUAL'" >&2
    FAIL=1
else
    echo "  I2b OK: queue-index retvals match expected sequence ($EXPECTED)"
fi

# ---- I3: vpid + vtid context propagation (HARD ASSERTION) -------------------
# Per schema v3 every event carries vpid + vtid via channel context;
# without this the consumer's per-thread LIFO walk is impossible.
SAMPLE_LINE=$(grep -E 'rocm_hsa:hsa_api_enter:' "$LOG" | head -1 || true)
if [ -z "$SAMPLE_LINE" ]; then
    echo "  I3 FAIL: no hsa_api_enter event found in trace" >&2
    FAIL=1
elif echo "$SAMPLE_LINE" | grep -qE 'vpid = [0-9]+' && \
     echo "$SAMPLE_LINE" | grep -qE 'vtid = [0-9]+'; then
    echo "  I3 OK: hsa_api_enter carries vpid + vtid channel context"
else
    echo "  I3 FAIL: hsa_api_enter missing vpid or vtid context" >&2
    echo "          line: $SAMPLE_LINE" >&2
    FAIL=1
fi

if [ "$FAIL" -ne 0 ]; then
    echo "FAIL: see $LOG" >&2
    exit 1
fi
echo "PASS"
