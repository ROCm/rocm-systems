#!/bin/bash
#
# load_test.sh - Basic load/unload test for PMU stub module
#

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Test configuration
MODULE_NAME="pmu_stub"
TEST_TIMEOUT=30

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

check_module_built() {
    if [ ! -f "../src/${MODULE_NAME}.ko" ]; then
        log_error "Module not found. Please build first with 'make'"
        exit 1
    fi
}

check_module_loaded() {
    lsmod | grep -q "^${MODULE_NAME}" && return 0 || return 1
}

load_module() {
    log_info "Loading module ${MODULE_NAME}..."

    if check_module_loaded; then
        log_warn "Module already loaded, unloading first..."
        unload_module
    fi

    insmod "../src/${MODULE_NAME}.ko" debug_enable=true timer_period_ms=100

    # Wait a bit for module to initialize
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
    log_info "Unloading module ${MODULE_NAME}..."

    if check_module_loaded; then
        rmmod "${MODULE_NAME}"
        sleep 1

        if check_module_loaded; then
            log_error "Failed to unload module"
            return 1
        else
            log_info "Module unloaded successfully"
            return 0
        fi
    else
        log_warn "Module not loaded"
        return 0
    fi
}

check_sysfs() {
    log_info "Checking sysfs interface..."

    local sysfs_path="/sys/bus/event_source/devices/pmu_stub"

    if [ -d "$sysfs_path" ]; then
        log_info "Found sysfs directory: $sysfs_path"

        # Check for events directory
        if [ -d "$sysfs_path/events" ]; then
            log_info "Found events directory"
            ls -la "$sysfs_path/events/" || true
        else
            log_warn "Events directory not found"
        fi

        # Check for format directory
        if [ -d "$sysfs_path/format" ]; then
            log_info "Found format directory"
            ls -la "$sysfs_path/format/" || true
        else
            log_warn "Format directory not found"
        fi

        return 0
    else
        log_error "Sysfs directory not found: $sysfs_path"
        return 1
    fi
}

check_dmesg() {
    log_info "Checking kernel messages..."

    local recent_messages=$(dmesg | tail -20 | grep -i "pmu_stub" || true)

    if [ -n "$recent_messages" ]; then
        echo "$recent_messages"
    else
        log_warn "No recent kernel messages from pmu_stub"
    fi
}

run_basic_tests() {
    log_info "Running basic functionality tests..."

    # Test 1: Load module
    if ! load_module; then
        log_error "Test 1 FAILED: Could not load module"
        return 1
    fi

    # Test 2: Check sysfs interface
    if ! check_sysfs; then
        log_error "Test 2 FAILED: Sysfs interface not working"
        unload_module
        return 1
    fi

    # Test 3: Check module info
    log_info "Module information:"
    modinfo "${MODULE_NAME}" 2>/dev/null || log_warn "Could not get module info"

    # Test 4: Check module parameters
    log_info "Module parameters:"
    if [ -d "/sys/module/${MODULE_NAME}/parameters" ]; then
        ls -la "/sys/module/${MODULE_NAME}/parameters/" || true
    fi

    # Test 5: Unload module
    if ! unload_module; then
        log_error "Test 5 FAILED: Could not unload module"
        return 1
    fi

    # Test 6: Reload test
    log_info "Testing reload..."
    if ! load_module; then
        log_error "Test 6 FAILED: Could not reload module"
        return 1
    fi

    if ! unload_module; then
        log_error "Test 6 FAILED: Could not unload module after reload"
        return 1
    fi

    log_info "All basic tests PASSED"
    return 0
}

# Main execution
main() {
    log_info "Starting PMU stub module load test..."

    check_root
    check_module_built

    if run_basic_tests; then
        log_info "=== ALL TESTS PASSED ==="
        check_dmesg
        exit 0
    else
        log_error "=== TESTS FAILED ==="
        check_dmesg
        exit 1
    fi
}

# Run with timeout
timeout $TEST_TIMEOUT bash -c "$(declare -f main check_root check_module_built run_basic_tests load_module unload_module check_sysfs check_dmesg log_info log_warn log_error check_module_loaded); main" || {
    log_error "Test timed out after $TEST_TIMEOUT seconds"
    exit 1
}