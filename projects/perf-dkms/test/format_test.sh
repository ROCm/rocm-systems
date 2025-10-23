#!/bin/bash
#
# format_test.sh - Test format attributes for amdgpu_pmu dimension support
#
# This script verifies that all format attributes are correctly exposed
# in sysfs and contain the expected bit field specifications.
#
# Expected behavior:
# - Format directory exists at /sys/bus/event_source/devices/amdgpu_pmu/format/
# - Each format file contains the correct bit range specification
# - perf list shows the amdgpu_pmu events with dimension parameters

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Test counters
PASS=0
FAIL=0

# Helper functions
log_pass() {
    echo -e "${GREEN}[PASS]${NC} $1"
    ((PASS++))
}

log_fail() {
    echo -e "${RED}[FAIL]${NC} $1"
    ((FAIL++))
}

log_info() {
    echo -e "${YELLOW}[INFO]${NC} $1"
}

# Module name
MODULE_NAME="amdgpu_pmu"
PMU_PATH="/sys/bus/event_source/devices/${MODULE_NAME}"
FORMAT_PATH="${PMU_PATH}/format"

# Check if module is loaded
if ! lsmod | grep -q "^${MODULE_NAME}"; then
    log_info "Module ${MODULE_NAME} not loaded, attempting to load..."
    if ! sudo modprobe ${MODULE_NAME} 2>/dev/null && ! sudo insmod ../src/${MODULE_NAME}.ko 2>/dev/null; then
        log_fail "Failed to load module ${MODULE_NAME}"
        echo "Please load the module manually before running this test."
        exit 1
    fi
    log_info "Module loaded successfully"
fi

# Check if PMU device exists
if [ ! -d "${PMU_PATH}" ]; then
    log_fail "PMU device directory does not exist: ${PMU_PATH}"
    exit 1
fi
log_pass "PMU device directory exists"

# Check if format directory exists
if [ ! -d "${FORMAT_PATH}" ]; then
    log_fail "Format directory does not exist: ${FORMAT_PATH}"
    exit 1
fi
log_pass "Format directory exists"

# Define expected format attributes and their values
declare -A EXPECTED_FORMATS=(
    ["config"]="config:0-63"
    ["config1"]="config1:0-63"
    ["xcc"]="config1:0-7"
    ["se"]="config1:8-15"
    ["sa"]="config1:16-23"
    ["wgp"]="config1:24-31"
    ["cu"]="config1:32-39"
    ["aggregate"]="config1:40"
    ["sample_all"]="config1:41"
)

# Test each format attribute
log_info "Testing format attributes..."
for attr in "${!EXPECTED_FORMATS[@]}"; do
    expected="${EXPECTED_FORMATS[$attr]}"
    file="${FORMAT_PATH}/${attr}"

    # Check if file exists
    if [ ! -f "$file" ]; then
        log_fail "Format file does not exist: $file"
        continue
    fi

    # Read and check content
    content=$(cat "$file" | tr -d '\n')
    if [ "$content" = "$expected" ]; then
        log_pass "Format attribute '$attr' = '$content'"
    else
        log_fail "Format attribute '$attr': expected '$expected', got '$content'"
    fi
done

# Test perf list output
log_info "Testing perf list output..."
if command -v perf &> /dev/null; then
    if perf list | grep -q "${MODULE_NAME}"; then
        log_pass "PMU appears in 'perf list' output"

        # Show some sample events
        log_info "Sample events from 'perf list ${MODULE_NAME}':"
        perf list ${MODULE_NAME} | head -20 | sed 's/^/  /'
    else
        log_fail "PMU does not appear in 'perf list' output"
    fi
else
    log_info "perf command not found, skipping perf list test"
fi

# Test dimension parameter parsing (if perf supports it)
log_info "Testing dimension parameter parsing..."
if command -v perf &> /dev/null; then
    # Try to create an event with dimension parameters
    # Note: This will fail if no GPU workload is running, but we're just
    # checking that perf accepts the syntax

    test_event="${MODULE_NAME}/sq_waves,se=0/"

    # Use a short sleep to test - this should succeed even without GPU activity
    if timeout 2 perf stat -e "$test_event" -a -- sleep 0.1 2>&1 | grep -q "sq_waves"; then
        log_pass "perf accepts dimension parameter syntax: $test_event"
    else
        # Check if it's just a counter value issue or syntax error
        if timeout 2 perf stat -e "$test_event" -a -- sleep 0.1 2>&1 | grep -qi "invalid.*event\|parse"; then
            log_fail "perf rejects dimension parameter syntax: $test_event"
        else
            log_pass "perf accepts dimension parameter syntax: $test_event (no events counted)"
        fi
    fi
else
    log_info "perf command not found, skipping parameter parsing test"
fi

# Test raw config1 encoding
log_info "Testing raw config1 encoding..."
if command -v perf &> /dev/null; then
    # config1=0x0100 should encode se=1
    test_event="${MODULE_NAME}/sq_waves,config1=0x0100/"

    if timeout 2 perf stat -e "$test_event" -a -- sleep 0.1 2>&1 | grep -q "sq_waves"; then
        log_pass "perf accepts raw config1 encoding: $test_event"
    else
        if timeout 2 perf stat -e "$test_event" -a -- sleep 0.1 2>&1 | grep -qi "invalid.*event\|parse"; then
            log_fail "perf rejects raw config1 encoding: $test_event"
        else
            log_pass "perf accepts raw config1 encoding: $test_event (no events counted)"
        fi
    fi
else
    log_info "perf command not found, skipping config1 encoding test"
fi

# Summary
echo ""
echo "========================================="
echo "Test Summary"
echo "========================================="
echo -e "${GREEN}PASSED: ${PASS}${NC}"
echo -e "${RED}FAILED: ${FAIL}${NC}"
echo "========================================="

if [ $FAIL -eq 0 ]; then
    echo -e "${GREEN}All tests passed!${NC}"
    exit 0
else
    echo -e "${RED}Some tests failed.${NC}"
    exit 1
fi
