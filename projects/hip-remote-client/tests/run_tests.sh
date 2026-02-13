#!/bin/bash
#
# Run remote HIP client tests
#
# Usage:
#   ./run_tests.sh                    # Uses TF_WORKER_HOST/PORT from env
#   ./run_tests.sh loopback           # Starts local worker, runs tests
#   ./run_tests.sh remote <host>      # Connects to remote host:50051
#   ./run_tests.sh tunnel <host>      # Creates SSH tunnel then runs tests
#
# Exit codes:
#   0 = All tests passed
#   1 = Some tests failed
#   2 = Worker not available
#

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BUILD_DIR="${SCRIPT_DIR}/../build"
WORKER_BUILD_DIR="${SCRIPT_DIR}/../../hip-remote-worker/build"

# Default settings
WORKER_PORT="${TF_WORKER_PORT:-50051}"
WORKER_HOST="${TF_WORKER_HOST:-localhost}"
MODE="${1:-env}"
TUNNEL_PID=""
LOCAL_WORKER_PID=""

cleanup() {
    if [ -n "$TUNNEL_PID" ]; then
        echo "Cleaning up SSH tunnel (PID $TUNNEL_PID)..."
        kill $TUNNEL_PID 2>/dev/null || true
    fi
    if [ -n "$LOCAL_WORKER_PID" ]; then
        echo "Cleaning up local worker (PID $LOCAL_WORKER_PID)..."
        kill $LOCAL_WORKER_PID 2>/dev/null || true
    fi
}
trap cleanup EXIT

check_worker() {
    local host=$1
    local port=$2
    nc -z "$host" "$port" 2>/dev/null
}

start_local_worker() {
    if [ ! -x "$WORKER_BUILD_DIR/hip-worker" ]; then
        echo "ERROR: Worker not found at $WORKER_BUILD_DIR/hip-worker"
        echo "Build the worker first: cd $WORKER_BUILD_DIR && cmake .. && make"
        exit 2
    fi

    echo "Starting local worker on port $WORKER_PORT..."
    "$WORKER_BUILD_DIR/hip-worker" -p "$WORKER_PORT" &
    LOCAL_WORKER_PID=$!
    sleep 2

    if ! kill -0 $LOCAL_WORKER_PID 2>/dev/null; then
        echo "ERROR: Worker failed to start"
        exit 2
    fi
    echo "Worker started (PID $LOCAL_WORKER_PID)"
}

create_tunnel() {
    local remote_host=$1
    local local_port="${2:-50052}"
    local remote_port="${3:-50051}"

    echo "Creating SSH tunnel: localhost:$local_port -> $remote_host:$remote_port"
    ssh -L "$local_port:localhost:$remote_port" "$remote_host" "sleep 300" &
    TUNNEL_PID=$!
    sleep 2

    if ! check_worker localhost "$local_port"; then
        echo "ERROR: Tunnel failed to establish"
        exit 2
    fi

    WORKER_HOST="localhost"
    WORKER_PORT="$local_port"
    echo "Tunnel established"
}

run_tests() {
    local failed=0

    echo ""
    echo "========================================"
    echo "Running HIP Remote Client Tests"
    echo "Worker: $WORKER_HOST:$WORKER_PORT"
    echo "========================================"
    echo ""

    export TF_WORKER_HOST="$WORKER_HOST"
    export TF_WORKER_PORT="$WORKER_PORT"

    # Basic tests
    echo "--- Basic Tests ---"
    if "$BUILD_DIR/hip_remote_test_basic"; then
        echo "Basic tests: PASSED"
    else
        echo "Basic tests: FAILED"
        failed=1
    fi
    echo ""

    # Extended tests
    if [ -x "$BUILD_DIR/hip_remote_test_extended" ]; then
        echo "--- Extended Tests ---"
        if "$BUILD_DIR/hip_remote_test_extended"; then
            echo "Extended tests: PASSED"
        else
            echo "Extended tests: FAILED"
            failed=1
        fi
        echo ""
    fi

    # Phase 2 tests
    if [ -x "$BUILD_DIR/hip_remote_test_phase2" ]; then
        echo "--- Phase 2 Tests ---"
        if "$BUILD_DIR/hip_remote_test_phase2"; then
            echo "Phase 2 tests: PASSED"
        else
            echo "Phase 2 tests: FAILED"
            failed=1
        fi
        echo ""
    fi

    echo "========================================"
    if [ $failed -eq 0 ]; then
        echo "ALL TESTS PASSED"
    else
        echo "SOME TESTS FAILED"
    fi
    echo "========================================"

    return $failed
}

# Main
case "$MODE" in
    env)
        # Use environment variables
        if [ -z "$TF_WORKER_HOST" ]; then
            echo "NOTE: TF_WORKER_HOST not set, using localhost"
        fi
        ;;

    loopback)
        # Start local worker
        WORKER_HOST="localhost"
        start_local_worker
        ;;

    remote)
        # Direct connection to remote host
        if [ -z "$2" ]; then
            echo "Usage: $0 remote <host> [port]"
            exit 1
        fi
        WORKER_HOST="$2"
        WORKER_PORT="${3:-50051}"
        ;;

    tunnel)
        # SSH tunnel to remote host
        if [ -z "$2" ]; then
            echo "Usage: $0 tunnel <host> [local_port] [remote_port]"
            exit 1
        fi
        create_tunnel "$2" "${3:-50052}" "${4:-50051}"
        ;;

    *)
        echo "Usage: $0 [env|loopback|remote <host>|tunnel <host>]"
        echo ""
        echo "Modes:"
        echo "  env      - Use TF_WORKER_HOST/PORT environment variables (default)"
        echo "  loopback - Start local worker and run tests"
        echo "  remote   - Connect directly to remote host"
        echo "  tunnel   - Create SSH tunnel then run tests"
        exit 1
        ;;
esac

# Check if tests exist
if [ ! -x "$BUILD_DIR/hip_remote_test_basic" ]; then
    echo "ERROR: Tests not built. Run: cd $BUILD_DIR && cmake .. -DBUILD_TESTING=ON && make"
    exit 2
fi

# Check worker connectivity (except for loopback which we just started)
if [ "$MODE" != "loopback" ]; then
    if ! check_worker "$WORKER_HOST" "$WORKER_PORT"; then
        echo "ERROR: Cannot connect to worker at $WORKER_HOST:$WORKER_PORT"
        echo ""
        echo "Options:"
        echo "  1. Start a local worker: $0 loopback"
        echo "  2. Use SSH tunnel: $0 tunnel <remote-host>"
        echo "  3. Set TF_WORKER_HOST and TF_WORKER_PORT"
        exit 2
    fi
fi

run_tests
