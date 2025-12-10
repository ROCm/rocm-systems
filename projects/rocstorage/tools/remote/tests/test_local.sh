#!/bin/bash
# test_local.sh - Test remote access tools with a local database
#
# This script tests the tools locally (without SSH) to verify they work correctly.
# It creates a temporary test database and runs each tool against it.

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TEST_DIR=$(mktemp -d)
TEST_DB="$TEST_DIR/test_trace.db"

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

log_info() { echo -e "${GREEN}[INFO]${NC} $1"; }
log_warn() { echo -e "${YELLOW}[WARN]${NC} $1"; }
log_error() { echo -e "${RED}[ERROR]${NC} $1"; }
log_test() { echo -e "${YELLOW}[TEST]${NC} $1"; }

cleanup() {
    log_info "Cleaning up..."
    rm -rf "$TEST_DIR"
    # Kill any background processes
    jobs -p | xargs -r kill 2>/dev/null || true
}
trap cleanup EXIT

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

CREATE TABLE rocpd_op (
    id INTEGER PRIMARY KEY,
    start INTEGER,
    "end" INTEGER,
    name TEXT
);

INSERT INTO rocpd_op VALUES (1, 1000, 2000, 'kernel_launch');
INSERT INTO rocpd_op VALUES (2, 2000, 3000, 'memory_copy');
INSERT INTO rocpd_op VALUES (3, 3000, 4000, 'kernel_launch');
INSERT INTO rocpd_op VALUES (4, 4000, 5000, 'barrier');
INSERT INTO rocpd_op VALUES (5, 5000, 6000, 'kernel_launch');

CREATE TABLE rocpd_track (
    id INTEGER PRIMARY KEY,
    name TEXT
);

INSERT INTO rocpd_track VALUES (1, 'GPU 0');
INSERT INTO rocpd_track VALUES (2, 'GPU 1');
INSERT INTO rocpd_track VALUES (3, 'Host');

CREATE VIEW rocpd_event AS SELECT * FROM rocpd_op;
EOF

    log_info "Test database created with $(sqlite3 "$TEST_DB" 'SELECT COUNT(*) FROM rocpd_op') rows in rocpd_op"
}

# Test rocstorage-server
test_server() {
    log_test "Testing rocstorage-server..."

    local port=18080
    local pid

    # Start server in background
    "$SCRIPT_DIR/rocstorage-server" --db "$TEST_DB" --port $port &
    pid=$!

    # Wait for server to start
    sleep 2

    if ! kill -0 $pid 2>/dev/null; then
        log_error "Server failed to start"
        return 1
    fi

    # Test endpoints
    log_info "Testing /status endpoint..."
    local status=$(curl -s "http://localhost:$port/status")
    if echo "$status" | grep -q '"status": "ok"'; then
        log_info "  /status: OK"
    else
        log_error "  /status: FAILED"
        echo "$status"
    fi

    log_info "Testing /tables endpoint..."
    local tables=$(curl -s "http://localhost:$port/tables")
    if echo "$tables" | grep -q '"success": true'; then
        log_info "  /tables: OK"
    else
        log_error "  /tables: FAILED"
        echo "$tables"
    fi

    log_info "Testing /query endpoint (GET)..."
    local query_result=$(curl -s "http://localhost:$port/query?sql=SELECT%20COUNT(*)%20as%20count%20FROM%20rocpd_op")
    if echo "$query_result" | grep -q '"count": 5'; then
        log_info "  /query (GET): OK"
    else
        log_error "  /query (GET): FAILED"
        echo "$query_result"
    fi

    log_info "Testing /query endpoint (POST)..."
    local post_result=$(curl -s -X POST "http://localhost:$port/query" \
        -H "Content-Type: application/json" \
        -d '{"sql": "SELECT name FROM rocpd_op WHERE id = 1"}')
    if echo "$post_result" | grep -q '"kernel_launch"'; then
        log_info "  /query (POST): OK"
    else
        log_error "  /query (POST): FAILED"
        echo "$post_result"
    fi

    log_info "Testing /metadata endpoint..."
    local metadata=$(curl -s "http://localhost:$port/metadata")
    if echo "$metadata" | grep -q '"schema_version"'; then
        log_info "  /metadata: OK"
    else
        log_error "  /metadata: FAILED"
        echo "$metadata"
    fi

    # Stop server
    kill $pid 2>/dev/null || true
    wait $pid 2>/dev/null || true

    log_info "rocstorage-server tests completed"
}

# Test rocstorage-sshfs (query command only, without actual SSHFS mount)
test_sshfs_query() {
    log_test "Testing rocstorage-sshfs query command..."

    local result=$("$SCRIPT_DIR/rocstorage-sshfs" query "$TEST_DB" "SELECT COUNT(*) FROM rocpd_op")
    if echo "$result" | grep -q "5"; then
        log_info "  query: OK"
    else
        log_error "  query: FAILED"
        echo "$result"
    fi

    log_info "rocstorage-sshfs query tests completed"
}

# Test Python dependencies
test_python_deps() {
    log_test "Checking Python dependencies..."

    if python3 -c "import json, http.server, sqlite3, threading" 2>/dev/null; then
        log_info "  Python standard library: OK"
    else
        log_error "  Python standard library: FAILED"
        return 1
    fi
}

# Test shell dependencies
test_shell_deps() {
    log_test "Checking shell dependencies..."

    local deps=("sqlite3" "curl")
    local missing=()

    for dep in "${deps[@]}"; do
        if command -v "$dep" &>/dev/null; then
            log_info "  $dep: OK"
        else
            log_error "  $dep: MISSING"
            missing+=("$dep")
        fi
    done

    if [ ${#missing[@]} -ne 0 ]; then
        log_error "Missing dependencies: ${missing[*]}"
        return 1
    fi
}

# Main
main() {
    echo "=========================================="
    echo "rocstorage Remote Tools - Local Test"
    echo "=========================================="
    echo ""

    test_shell_deps || exit 1
    test_python_deps || exit 1

    echo ""
    create_test_db

    echo ""
    test_sshfs_query

    echo ""
    test_server

    echo ""
    echo "=========================================="
    log_info "All tests completed successfully!"
    echo "=========================================="
}

main "$@"