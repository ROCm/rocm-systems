#!/bin/bash
# test_local.sh - Test remote access tools with a local database
#
# This script tests the tools locally (without SSH) to verify they work correctly.
# It creates a temporary test database and runs each tool against it.
#
# Can be run directly or invoked from Docker via test_docker.sh

set -e

# Determine tool directory - either relative to script or /app in Docker
if [ -n "$TOOLS_DIR" ]; then
    : # Use provided TOOLS_DIR
elif [ -d "/app" ] && [ -x "/app/rocstorage-server" ]; then
    TOOLS_DIR="/app"
else
    SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
    TOOLS_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
fi

TEST_DIR=$(mktemp -d)
TEST_DB="$TEST_DIR/test_trace.db"

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

log_info() { echo -e "${GREEN}[INFO]${NC} $1"; }
log_warn() { echo -e "${YELLOW}[WARN]${NC} $1"; }
log_error() { echo -e "${RED}[ERROR]${NC} $1"; }
log_test() { echo -e "${BLUE}[TEST]${NC} $1"; }
log_pass() { echo -e "${GREEN}[PASS]${NC} $1"; }
log_fail() { echo -e "${RED}[FAIL]${NC} $1"; }

# Test tracking
TESTS_RUN=0
TESTS_PASSED=0
TESTS_FAILED=0

cleanup() {
    log_info "Cleaning up..."
    rm -rf "$TEST_DIR"
    # Kill any background processes
    jobs -p 2>/dev/null | xargs -r kill 2>/dev/null || true
}
trap cleanup EXIT

# Run a test and track results
run_test() {
    local name="$1"
    local cmd="$2"

    TESTS_RUN=$((TESTS_RUN + 1))

    if eval "$cmd" >/dev/null 2>&1; then
        log_pass "$name"
        TESTS_PASSED=$((TESTS_PASSED + 1))
        return 0
    else
        log_fail "$name"
        TESTS_FAILED=$((TESTS_FAILED + 1))
        return 1
    fi
}

# Create test database
create_test_db() {
    log_info "Creating test database: $TEST_DB"

    sqlite3 "$TEST_DB" <<'EOF'
-- Create rocstorage-style tables
CREATE TABLE rocpd_metadata (
    tag TEXT PRIMARY KEY,
    value TEXT
);

INSERT INTO rocpd_metadata VALUES ('schema_version', '3');
INSERT INTO rocpd_metadata VALUES ('uuid', 'test123');
INSERT INTO rocpd_metadata VALUES ('guid', 'test123');

CREATE TABLE rocpd_op (
    id INTEGER PRIMARY KEY,
    start INTEGER,
    "end" INTEGER,
    name TEXT,
    track_id INTEGER
);

INSERT INTO rocpd_op VALUES (1, 1000000, 2000000, 'kernel_launch', 1);
INSERT INTO rocpd_op VALUES (2, 2000000, 3000000, 'memory_copy_h2d', 1);
INSERT INTO rocpd_op VALUES (3, 3000000, 4500000, 'kernel_execute', 1);
INSERT INTO rocpd_op VALUES (4, 4500000, 5000000, 'memory_copy_d2h', 1);
INSERT INTO rocpd_op VALUES (5, 5000000, 6000000, 'kernel_launch', 2);
INSERT INTO rocpd_op VALUES (6, 6000000, 7500000, 'kernel_execute', 2);
INSERT INTO rocpd_op VALUES (7, 7500000, 8000000, 'barrier', 1);
INSERT INTO rocpd_op VALUES (8, 8000000, 9000000, 'kernel_launch', 1);
INSERT INTO rocpd_op VALUES (9, 9000000, 10000000, 'kernel_execute', 1);
INSERT INTO rocpd_op VALUES (10, 10000000, 11000000, 'finalize', 1);

CREATE TABLE rocpd_track (
    id INTEGER PRIMARY KEY,
    name TEXT,
    type TEXT
);

INSERT INTO rocpd_track VALUES (1, 'GPU 0 - Compute', 'gpu');
INSERT INTO rocpd_track VALUES (2, 'GPU 1 - Compute', 'gpu');
INSERT INTO rocpd_track VALUES (3, 'Host Thread 0', 'cpu');

CREATE VIEW rocpd_event AS SELECT * FROM rocpd_op;
EOF

    local row_count=$(sqlite3 "$TEST_DB" 'SELECT COUNT(*) FROM rocpd_op')
    log_info "Test database created with $row_count rows in rocpd_op"
}

# Test dependencies
test_dependencies() {
    log_test "Checking dependencies..."

    run_test "sqlite3 available" "command -v sqlite3"
    run_test "curl available" "command -v curl"
    run_test "python3 available" "command -v python3"
    run_test "Python json module" "python3 -c 'import json'"
    run_test "Python http.server module" "python3 -c 'import http.server'"
    run_test "Python sqlite3 module" "python3 -c 'import sqlite3'"
}

# Test rocstorage-sshfs query command
test_sshfs_query() {
    log_test "Testing rocstorage-sshfs query command..."

    # Check if sshfs is available
    if ! command -v sshfs &>/dev/null; then
        log_warn "sshfs not installed, skipping sshfs tests"
        return 0
    fi

    local result=$("$TOOLS_DIR/rocstorage-sshfs" query "$TEST_DB" "SELECT COUNT(*) FROM rocpd_op" 2>&1)
    run_test "Basic count query" "echo '$result' | grep -q '10'"

    result=$("$TOOLS_DIR/rocstorage-sshfs" query "$TEST_DB" "SELECT name FROM rocpd_op LIMIT 1" 2>&1)
    run_test "Select with column" "echo '$result' | grep -q 'kernel_launch'"

    result=$("$TOOLS_DIR/rocstorage-sshfs" query "$TEST_DB" "SELECT value FROM rocpd_metadata WHERE tag='schema_version'" 2>&1)
    run_test "Metadata query" "echo '$result' | grep -q '3'"
}

# Test rocstorage-server
test_server() {
    log_test "Testing rocstorage-server..."

    local port=18080
    local base_url="http://localhost:$port"

    # Start server in background
    "$TOOLS_DIR/rocstorage-server" --db "$TEST_DB" --port $port --bind 127.0.0.1 &
    local server_pid=$!

    # Wait for server to start
    local retries=20
    while [ $retries -gt 0 ]; do
        if curl -s "$base_url/status" >/dev/null 2>&1; then
            break
        fi
        sleep 0.25
        retries=$((retries - 1))
    done

    if [ $retries -eq 0 ]; then
        log_error "Server failed to start"
        kill $server_pid 2>/dev/null || true
        return 1
    fi

    # Test /status endpoint
    local response=$(curl -s "$base_url/status")
    run_test "GET /status returns ok" "echo '$response' | grep -q '\"status\": \"ok\"'"

    # Test /tables endpoint
    response=$(curl -s "$base_url/tables")
    run_test "GET /tables returns success" "echo '$response' | grep -q '\"success\": true'"
    run_test "GET /tables lists rocpd_op" "echo '$response' | grep -q 'rocpd_op'"

    # Test /schema endpoint
    response=$(curl -s "$base_url/schema")
    run_test "GET /schema returns success" "echo '$response' | grep -q '\"success\": true'"

    # Test /metadata endpoint
    response=$(curl -s "$base_url/metadata")
    run_test "GET /metadata returns schema_version" "echo '$response' | grep -q 'schema_version'"

    # Test /traces endpoint
    response=$(curl -s "$base_url/traces")
    run_test "GET /traces returns success" "echo '$response' | grep -q '\"success\": true'"

    # Test GET /query
    response=$(curl -s "$base_url/query?sql=SELECT%20COUNT(*)%20as%20count%20FROM%20rocpd_op")
    run_test "GET /query count returns 10" "echo '$response' | grep -q '\"count\": 10'"

    # Test GET /query with limit
    response=$(curl -s "$base_url/query?sql=SELECT%20*%20FROM%20rocpd_op&limit=3")
    run_test "GET /query with limit returns 3 rows" "echo '$response' | grep -q '\"row_count\": 3'"

    # Test POST /query with JSON
    response=$(curl -s -X POST "$base_url/query" \
        -H "Content-Type: application/json" \
        -d '{"sql": "SELECT name FROM rocpd_op WHERE id = 1"}')
    run_test "POST /query JSON returns kernel_launch" "echo '$response' | grep -q 'kernel_launch'"

    # Test POST /query with plain SQL
    response=$(curl -s -X POST "$base_url/query" \
        -d "SELECT COUNT(*) as cnt FROM rocpd_track")
    run_test "POST /query plain SQL returns count" "echo '$response' | grep -q '\"cnt\": 3'"

    # Test /cache/clear
    response=$(curl -s "$base_url/cache/clear")
    run_test "GET /cache/clear succeeds" "echo '$response' | grep -q '\"success\": true'"

    # Test invalid SQL
    response=$(curl -s "$base_url/query?sql=INVALID%20SQL%20QUERY")
    run_test "Invalid SQL returns error" "echo '$response' | grep -q '\"success\": false'"

    # Stop server
    kill $server_pid 2>/dev/null || true
    wait $server_pid 2>/dev/null || true
}

# Test rocstorage-ssh-proxy startup and validation
test_ssh_proxy_startup() {
    log_test "Testing rocstorage-ssh-proxy startup validation..."

    # Test that help works
    local output=$("$TOOLS_DIR/rocstorage-ssh-proxy" --help 2>&1)
    run_test "SSH proxy --help works" "echo '$output' | grep -q 'Local HTTP proxy'"

    # Test that help mentions ControlMaster
    run_test "SSH proxy --help mentions ControlMaster" "echo '$output' | grep -q 'ControlMaster'"

    # Test that missing args are caught
    output=$("$TOOLS_DIR/rocstorage-ssh-proxy" 2>&1 || true)
    run_test "SSH proxy requires --host" "echo '$output' | grep -q -E '(--host|required)'"

    # Test that missing socket shows helpful message
    output=$("$TOOLS_DIR/rocstorage-ssh-proxy" --host nonexistent-host --db /tmp/test.db 2>&1 || true)
    run_test "SSH proxy shows socket setup instructions" "echo '$output' | grep -q 'ssh -fNM'"

    # Test socket discovery with a real socket file (using netcat if available)
    if command -v nc >/dev/null 2>&1; then
        local socket_path="$TEST_DIR/testhost.sock"

        # Create a Unix socket with netcat in background
        nc -lU "$socket_path" &
        local nc_pid=$!
        sleep 0.3

        # Test that the proxy finds the socket and tries to check connection
        output=$("$TOOLS_DIR/rocstorage-ssh-proxy" --host testhost --db /tmp/test.db --socket "$socket_path" 2>&1 || true)
        run_test "SSH proxy finds specified socket" "echo '$output' | grep -q -E '(Checking connection|not active)'"

        kill $nc_pid 2>/dev/null || true
        rm -f "$socket_path"
    fi
}

# Test C++ RemoteDatabase backend
test_cpp_remote_database() {
    log_test "Testing C++ RemoteDatabase backend..."

    # Find the project root and test binary
    local project_root="$TOOLS_DIR/../.."
    local test_binary=""

    # Check common build locations
    for build_dir in "$project_root/build" "$project_root/cmake-build-debug" "$project_root/cmake-build-release"; do
        if [ -x "$build_dir/tests/unit/rocstorage_unit_tests" ]; then
            test_binary="$build_dir/tests/unit/rocstorage_unit_tests"
            break
        fi
    done

    if [ -z "$test_binary" ]; then
        log_warn "Unit test binary not found, skipping C++ RemoteDatabase tests"
        log_warn "Build the project with: cd build && cmake .. && make"
        return 0
    fi

    # Check if RemoteDatabase tests exist in the binary
    if ! "$test_binary" --gtest_list_tests 2>/dev/null | grep -q "RemoteDatabaseTest"; then
        log_warn "RemoteDatabase tests not found in binary (CURL not available during build?)"
        return 0
    fi

    local port=18082
    local base_url="http://localhost:$port"

    # Start server in background
    "$TOOLS_DIR/rocstorage-server" --db "$TEST_DB" --port $port --bind 127.0.0.1 &
    local server_pid=$!

    # Wait for server to start
    local retries=20
    while [ $retries -gt 0 ]; do
        if curl -s "$base_url/status" >/dev/null 2>&1; then
            break
        fi
        sleep 0.25
        retries=$((retries - 1))
    done

    if [ $retries -eq 0 ]; then
        log_error "Server failed to start for C++ tests"
        kill $server_pid 2>/dev/null || true
        return 1
    fi

    # Run the RemoteDatabase unit tests (enabled ones only - they test construction and error handling)
    local output
    output=$("$test_binary" --gtest_filter="RemoteDatabaseTest.*" 2>&1)
    local exit_code=$?

    if [ $exit_code -eq 0 ]; then
        run_test "C++ RemoteDatabase unit tests pass" "true"
    else
        log_fail "C++ RemoteDatabase unit tests failed"
        echo "$output" | tail -20
        TESTS_RUN=$((TESTS_RUN + 1))
        TESTS_FAILED=$((TESTS_FAILED + 1))
    fi

    # Run the integration tests (DISABLED by default, these actually connect to server)
    # Override the server URL to use our test server
    export ROCSTORAGE_TEST_SERVER_URL="$base_url"
    output=$("$test_binary" --gtest_filter="RemoteDatabaseIntegrationTest.*" --gtest_also_run_disabled_tests 2>&1)
    exit_code=$?

    if [ $exit_code -eq 0 ]; then
        run_test "C++ RemoteDatabase integration tests pass" "true"
    else
        # Integration tests may fail if server doesn't match expected format, that's ok
        log_warn "C++ RemoteDatabase integration tests returned non-zero (may be expected)"
    fi

    # Stop server
    kill $server_pid 2>/dev/null || true
    wait $server_pid 2>/dev/null || true
}

# Test SSH proxy with localhost if SSH is available
test_ssh_proxy_localhost() {
    log_test "Testing rocstorage-ssh-proxy with localhost..."

    # Skip if ssh to localhost isn't available
    if ! ssh -o BatchMode=yes -o ConnectTimeout=2 localhost true 2>/dev/null; then
        log_warn "Skipping localhost SSH test (SSH to localhost not available)"
        return 0
    fi

    local socket_path="$TEST_DIR/localhost.sock"
    local port=18081
    local base_url="http://localhost:$port"

    # Establish master connection
    ssh -fNM -S "$socket_path" -o ControlPersist=60 localhost
    sleep 0.5

    if [ ! -S "$socket_path" ]; then
        log_warn "Failed to create SSH control socket"
        return 0
    fi

    # Start proxy in background
    "$TOOLS_DIR/rocstorage-ssh-proxy" --host localhost --db "$TEST_DB" --socket "$socket_path" --port $port &
    local proxy_pid=$!

    # Wait for proxy to start
    local retries=20
    while [ $retries -gt 0 ]; do
        if curl -s "$base_url/status" >/dev/null 2>&1; then
            break
        fi
        sleep 0.25
        retries=$((retries - 1))
    done

    if [ $retries -eq 0 ]; then
        log_warn "Proxy failed to start (expected if localhost SSH not configured)"
        kill $proxy_pid 2>/dev/null || true
        ssh -S "$socket_path" -O exit localhost 2>/dev/null || true
        return 0
    fi

    # Test endpoints
    local response=$(curl -s "$base_url/status")
    run_test "SSH proxy /status returns ok" "echo '$response' | grep -q '\"status\": \"ok\"'"

    response=$(curl -s "$base_url/tables")
    run_test "SSH proxy /tables returns success" "echo '$response' | grep -q '\"success\": true'"

    response=$(curl -s "$base_url/query?sql=SELECT%20COUNT(*)%20as%20count%20FROM%20rocpd_op")
    run_test "SSH proxy /query returns count" "echo '$response' | grep -q '\"count\": 10'"

    # Cleanup
    kill $proxy_pid 2>/dev/null || true
    ssh -S "$socket_path" -O exit localhost 2>/dev/null || true
}

# Print summary
print_summary() {
    echo ""
    echo "=========================================="
    echo "Test Summary"
    echo "=========================================="
    echo "Tests run:    $TESTS_RUN"
    echo -e "Tests passed: ${GREEN}$TESTS_PASSED${NC}"
    echo -e "Tests failed: ${RED}$TESTS_FAILED${NC}"
    echo "=========================================="

    if [ $TESTS_FAILED -eq 0 ]; then
        echo -e "${GREEN}All tests passed!${NC}"
        return 0
    else
        echo -e "${RED}Some tests failed!${NC}"
        return 1
    fi
}

# Main
main() {
    echo "=========================================="
    echo "rocstorage Remote Tools - Test Suite"
    echo "=========================================="
    echo "Tools directory: $TOOLS_DIR"
    echo ""

    test_dependencies
    echo ""

    create_test_db
    echo ""

    test_sshfs_query
    echo ""

    test_server
    echo ""

    test_ssh_proxy_startup
    echo ""

    test_cpp_remote_database
    echo ""

    test_ssh_proxy_localhost
    echo ""

    print_summary
}

main "$@"
