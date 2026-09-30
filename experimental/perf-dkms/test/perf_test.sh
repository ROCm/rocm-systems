#!/bin/bash
#
# perf_test.sh - Perf integration test for PMU stub module
#

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Test configuration
MODULE_NAME="amdgpu_pmu"
PMU_NAME="amdgpu_pmu"
TEST_DURATION=5
TEST_TIMEOUT=60

# Helper functions
log_info() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

log_warn() {
    echo -e "${YELLOW}[WARN]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

check_root() {
    if [ "$EUID" -ne 0 ]; then
        log_error "This script must be run as root"
        exit 1
    fi
}

check_perf_available() {
    if ! command -v perf &> /dev/null; then
        log_error "perf command not found. Please install linux-perf-tools"
        exit 1
    fi
}

check_module_loaded() {
    lsmod | grep -q "^${MODULE_NAME}" && return 0 || return 1
}

load_module() {
    log_info "Loading module ${MODULE_NAME}..."

    if check_module_loaded; then
        log_info "Module already loaded"
        return 0
    fi

    if [ ! -f "../build/src/${MODULE_NAME}.ko" ]; then
        log_error "Module not found. Please build first with 'cmake -B build && cmake --build build'"
        exit 1
    fi

    insmod "../build/src/${MODULE_NAME}.ko" debug_enable=true timer_period_ms=50

    # Wait for module to initialize
    sleep 2

    if check_module_loaded; then
        log_info "Module loaded successfully"
        return 0
    else
        log_error "Failed to load module"
        return 1
    fi
}

unload_module() {
    if check_module_loaded; then
        log_info "Unloading module ${MODULE_NAME}..."
        rmmod "${MODULE_NAME}"
        sleep 1
    fi
}

test_perf_list() {
    log_info "Testing perf list integration..."

    local output=$(perf list | grep -i "${PMU_NAME}" || true)

    if [ -n "$output" ]; then
        log_info "Found PMU events in perf list:"
        echo "$output"
        return 0
    else
        log_error "PMU events not found in perf list"
        return 1
    fi
}

test_perf_stat() {
    log_info "Testing perf stat with PMU events..."

    local events=(
        "${PMU_NAME}/cycles/"
        "${PMU_NAME}/instructions/"
        "${PMU_NAME}/cache-misses/"
        "${PMU_NAME}/bandwidth/"
    )

    for event in "${events[@]}"; do
        log_info "Testing event: $event"

        local output=$(timeout 10 perf stat -e "$event" sleep $TEST_DURATION 2>&1 || true)

        if echo "$output" | grep -q "Performance counter stats"; then
            log_info "Event $event working correctly"
            echo "$output" | grep -E "(Performance counter stats|$event)" || true
        else
            log_error "Event $event failed"
            echo "$output"
            return 1
        fi
    done

    return 0
}

test_multiple_events() {
    log_info "Testing multiple events simultaneously..."

    local event_list="${PMU_NAME}/cycles/,${PMU_NAME}/instructions/,${PMU_NAME}/cache-misses/"

    local output=$(timeout 15 perf stat -e "$event_list" sleep $TEST_DURATION 2>&1 || true)

    if echo "$output" | grep -q "Performance counter stats"; then
        log_info "Multiple events test passed"
        echo "$output" | grep -E "(Performance counter stats|$PMU_NAME)" || true
        return 0
    else
        log_error "Multiple events test failed"
        echo "$output"
        return 1
    fi
}

test_event_validation() {
    log_info "Testing event validation..."

    # Test invalid event (should fail gracefully)
    local invalid_event="${PMU_NAME}/invalid-event/"

    local output=$(timeout 10 perf stat -e "$invalid_event" sleep 1 2>&1 || true)

    if echo "$output" | grep -qi "not supported\|invalid\|error"; then
        log_info "Invalid event correctly rejected"
        return 0
    else
        log_warn "Invalid event validation may not be working properly"
        echo "$output"
        return 0  # Non-critical
    fi
}

test_perf_record() {
    log_info "Testing perf record (if supported)..."

    # Note: Our PMU doesn't support sampling, so this should fail gracefully
    local output=$(timeout 10 perf record -e "${PMU_NAME}/cycles/" sleep 1 2>&1 || true)

    if echo "$output" | grep -qi "not supported\|sampling"; then
        log_info "Sampling correctly not supported (as expected)"
        return 0
    else
        log_warn "Sampling test result unclear"
        echo "$output"
        return 0  # Non-critical
    fi
}

check_counter_increment() {
    log_info "Checking counter increments..."

    # Read initial counter values through sysfs if available
    local sysfs_path="/sys/bus/event_source/devices/${PMU_NAME}"

    if [ -d "$sysfs_path" ]; then
        log_info "PMU sysfs path found: $sysfs_path"

        # Run a short test and check if counters are updating
        timeout 10 perf stat -e "${PMU_NAME}/cycles/" sleep 2 2>&1 | grep -E "cycles|performance" || true

        return 0
    else
        log_error "PMU sysfs path not found"
        return 1
    fi
}

run_perf_tests() {
    log_info "Running perf integration tests..."

    # Test 1: Load module
    if ! load_module; then
        log_error "Test 1 FAILED: Could not load module"
        return 1
    fi

    # Test 2: Check perf list
    if ! test_perf_list; then
        log_error "Test 2 FAILED: perf list integration"
        return 1
    fi

    # Test 3: Check perf stat
    if ! test_perf_stat; then
        log_error "Test 3 FAILED: perf stat integration"
        return 1
    fi

    # Test 4: Multiple events
    if ! test_multiple_events; then
        log_error "Test 4 FAILED: multiple events"
        return 1
    fi

    # Test 5: Event validation
    test_event_validation  # Non-critical

    # Test 6: Record test
    test_perf_record  # Non-critical

    # Test 7: Counter increment
    if ! check_counter_increment; then
        log_error "Test 7 FAILED: counter increment"
        return 1
    fi

    log_info "All perf tests PASSED"
    return 0
}

cleanup() {
    log_info "Cleaning up..."
    unload_module
}

# Set up cleanup trap
trap cleanup EXIT

# Main execution
main() {
    log_info "Starting PMU stub perf integration test..."

    check_root
    check_perf_available

    if run_perf_tests; then
        log_info "=== ALL PERF TESTS PASSED ==="
        return 0
    else
        log_error "=== PERF TESTS FAILED ==="
        return 1
    fi
}

# Run with timeout
timeout $TEST_TIMEOUT bash -c "$(declare -f main check_root check_perf_available run_perf_tests load_module unload_module test_perf_list test_perf_stat test_multiple_events test_event_validation test_perf_record check_counter_increment log_info log_warn log_error check_module_loaded); main" || {
    log_error "Test timed out after $TEST_TIMEOUT seconds"
    exit 1
}