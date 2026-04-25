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
    pkill -P $$ 2>/dev/null || true
    pkill -f "lttng-sessiond.*--no-kernel" 2>/dev/null || true
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

# Start a per-user sessiond, isolated session.
SESSION="hsa-api-test-$$"
lttng-sessiond --daemonize --no-kernel
trap 'pkill -f "lttng-sessiond.*--no-kernel" 2>/dev/null || true; rm -rf "$TRACE_DIR"' EXIT

lttng create "$SESSION" --output="$TRACE_DIR/trace" >/dev/null
# Per the channel-mode commitment: discard policy.
# Sub-buffer sizing: per Phase 0 finding #1, container hosts with /dev/shm
# limited to 64 MiB and ~224 CPUs need a much smaller channel than
# LTTng's default (524288 B x 4 sub-buffers per CPU). 4 KiB x 2 fits.
lttng enable-channel --userspace --discard --subbuf-size=4096 --num-subbuf=2 default >/dev/null
lttng enable-event --userspace -c default 'rocm_hsa:hsa_api_enter,rocm_hsa:hsa_api_exit_status,rocm_hsa:hsa_api_exit_ptr,rocm_hsa:hsa_api_exit_void,rocm_hsa:hsa_doorbell_ring,rocm_hsa:hsa_intercept_packets' >/dev/null
lttng start "$SESSION" >/dev/null

# Run the tiny program. If no GPU is reachable, we still expect at least
# hsa_init events; that is enough to verify the tracepoint plumbing.
LD_LIBRARY_PATH="$LIB_DIR:${LD_LIBRARY_PATH:-}" "$TRACE_DIR/tiny" || true

lttng stop "$SESSION" >/dev/null
lttng destroy "$SESSION" >/dev/null

LOG="$TRACE_DIR/babeltrace.log"
babeltrace2 "$TRACE_DIR/trace" > "$LOG"

# Assertions: hsa_init must produce at least one enter and one exit_status.
EXPECTED_APIS=(hsa_init hsa_shut_down)

fail() { echo "FAIL: $*" >&2; exit 1; }

for api in "${EXPECTED_APIS[@]}"; do
    grep -q "rocm_hsa:hsa_api_enter:.*api_name = \"$api\"" "$LOG" \
        || fail "missing enter event for $api"
    grep -q "rocm_hsa:hsa_api_exit_status:.*api_name = \"$api\"" "$LOG" \
        || fail "missing exit_status event for $api"
done

# Print summary.
N_EVENTS=$(wc -l < "$LOG")
N_ENTER=$(grep -c 'rocm_hsa:hsa_api_enter' "$LOG" || true)
N_EXIT=$(grep -c 'rocm_hsa:hsa_api_exit_status' "$LOG" || true)
N_DOORBELL=$(grep -c 'rocm_hsa:hsa_doorbell_ring' "$LOG" || true)

echo "PASS hsa_api_tracepoints: $N_EVENTS events, $N_ENTER enter, $N_EXIT exit_status, $N_DOORBELL doorbell"
