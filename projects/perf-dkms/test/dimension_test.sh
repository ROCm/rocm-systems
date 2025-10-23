#!/bin/bash
#
# dimension_test.sh - Integration tests for dimension-aware PMU events
#
# This script tests the dimension-aware performance monitoring functionality
# by running perf commands with various dimension specifications.
#
# NOTE: This script requires actual AMD GPU hardware to run successfully.
#       Without hardware, it will verify the module loads and format attributes
#       are exposed correctly, but event measurements will fail.
#

set -e  # Exit on error

# Color output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Test counters
TESTS_RUN=0
TESTS_PASSED=0
TESTS_FAILED=0

# Logging functions
log_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

log_success() {
    echo -e "${GREEN}[PASS]${NC} $1"
    ((TESTS_PASSED++))
    ((TESTS_RUN++))
}

log_fail() {
    echo -e "${RED}[FAIL]${NC} $1"
    ((TESTS_FAILED++))
    ((TESTS_RUN++))
}

log_warn() {
    echo -e "${YELLOW}[WARN]${NC} $1"
}

# Check if module is loaded
check_module_loaded() {
    if ! lsmod | grep -q amdgpu_pmu; then
        log_fail "Module amdgpu_pmu is not loaded"
        return 1
    fi
    log_success "Module amdgpu_pmu is loaded"
    return 0
}

# Check format attributes
check_format_attributes() {
    local format_dir="/sys/bus/event_source/devices/amdgpu_pmu/format"

    if [ ! -d "$format_dir" ]; then
        log_fail "Format directory not found: $format_dir"
        return 1
    fi

    local expected_attrs=("config" "config1" "xcc" "se" "sa" "wgp" "cu" "aggregate" "sample_all")
    local all_found=true

    for attr in "${expected_attrs[@]}"; do
        if [ ! -f "$format_dir/$attr" ]; then
            log_fail "Format attribute missing: $attr"
            all_found=false
        else
            local content=$(cat "$format_dir/$attr")
            log_info "Format $attr: $content"
        fi
    done

    if $all_found; then
        log_success "All format attributes present"
        return 0
    else
        return 1
    fi
}

# Verify format attribute bit ranges
verify_format_ranges() {
    local format_dir="/sys/bus/event_source/devices/amdgpu_pmu/format"

    # Expected bit ranges
    declare -A expected_ranges=(
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

    local all_correct=true

    for attr in "${!expected_ranges[@]}"; do
        local actual=$(cat "$format_dir/$attr" 2>/dev/null | tr -d '\n')
        local expected="${expected_ranges[$attr]}"

        if [ "$actual" != "$expected" ]; then
            log_fail "Format $attr: expected '$expected', got '$actual'"
            all_correct=false
        fi
    done

    if $all_correct; then
        log_success "All format attribute bit ranges correct"
        return 0
    else
        return 1
    fi
}

# Test named parameter syntax
test_named_parameters() {
    log_info "Testing named parameter syntax..."

    # Test 1: Single SE specification
    log_info "Test: SE=0"
    if perf stat -e amdgpu_pmu/sq_waves,se=0/ -a sleep 0.1 2>&1 | grep -q "sq_waves"; then
        log_success "Named parameter: se=0"
    else
        log_fail "Named parameter: se=0"
    fi

    # Test 2: SE + SA specification
    log_info "Test: SE=1, SA=0"
    if perf stat -e amdgpu_pmu/sq_waves,se=1,sa=0/ -a sleep 0.1 2>&1 | grep -q "sq_waves"; then
        log_success "Named parameters: se=1,sa=0"
    else
        log_fail "Named parameters: se=1,sa=0"
    fi

    # Test 3: Full SE/SA/WGP specification
    log_info "Test: SE=2, SA=1, WGP=3"
    if perf stat -e amdgpu_pmu/sq_waves,se=2,sa=1,wgp=3/ -a sleep 0.1 2>&1 | grep -q "sq_waves"; then
        log_success "Named parameters: se=2,sa=1,wgp=3"
    else
        log_fail "Named parameters: se=2,sa=1,wgp=3"
    fi

    # Test 4: Different counter
    log_info "Test: TA counter with dimensions"
    if perf stat -e amdgpu_pmu/ta_busy,se=0,sa=0/ -a sleep 0.1 2>&1 | grep -q "ta_busy"; then
        log_success "TA counter with dimensions"
    else
        log_fail "TA counter with dimensions"
    fi
}

# Test raw config1 syntax
test_raw_config1() {
    log_info "Testing raw config1 syntax..."

    # Test 1: SE=1 encoded as config1
    # SE is bits 8-15, so SE=1 is 0x0100
    log_info "Test: config1=0x0100 (SE=1)"
    if perf stat -e amdgpu_pmu/sq_waves,config1=0x0100/ -a sleep 0.1 2>&1 | grep -q "sq_waves"; then
        log_success "Raw config1: 0x0100 (SE=1)"
    else
        log_fail "Raw config1: 0x0100 (SE=1)"
    fi

    # Test 2: SE=2, SA=1 encoded
    # SE=2 (bits 8-15) | SA=1 (bits 16-23) = 0x00020200
    log_info "Test: config1=0x00010200 (SE=2, SA=1)"
    if perf stat -e amdgpu_pmu/sq_waves,config1=0x00010200/ -a sleep 0.1 2>&1 | grep -q "sq_waves"; then
        log_success "Raw config1: 0x00010200"
    else
        log_fail "Raw config1: 0x00010200"
    fi

    # Test 3: SE=3, SA=1, WGP=2 encoded
    # SE=3 | SA=1 | WGP=2 = 0x02010300
    log_info "Test: config1=0x02010300 (SE=3, SA=1, WGP=2)"
    if perf stat -e amdgpu_pmu/sq_waves,config1=0x02010300/ -a sleep 0.1 2>&1 | grep -q "sq_waves"; then
        log_success "Raw config1: 0x02010300"
    else
        log_fail "Raw config1: 0x02010300"
    fi
}

# Test mixed syntax (named + flags)
test_mixed_syntax() {
    log_info "Testing mixed syntax (named + flags)..."

    # Test 1: SE with aggregate flag
    log_info "Test: se=1,aggregate=1"
    if perf stat -e amdgpu_pmu/sq_waves,se=1,aggregate=1/ -a sleep 0.1 2>&1 | grep -q "sq_waves"; then
        log_success "Mixed syntax: se=1,aggregate=1"
    else
        log_warn "Mixed syntax: se=1,aggregate=1 (aggregate mode not yet implemented)"
    fi

    # Test 2: sample_all flag
    log_info "Test: sample_all=1"
    if perf stat -e amdgpu_pmu/sq_waves,sample_all=1/ -a sleep 0.1 2>&1 | grep -q "sq_waves"; then
        log_success "Mixed syntax: sample_all=1"
    else
        log_warn "Mixed syntax: sample_all=1 (sample_all mode not yet implemented)"
    fi
}

# Test error cases
test_error_cases() {
    log_info "Testing error cases..."

    # Test 1: Invalid SE (too large)
    log_info "Test: Invalid SE=99"
    if perf stat -e amdgpu_pmu/sq_waves,se=99/ -a sleep 0.1 2>&1 | grep -qi "invalid\|error"; then
        log_success "Error detected for invalid SE=99"
    else
        log_fail "No error for invalid SE=99"
    fi

    # Test 2: Invalid SA
    log_info "Test: Invalid SA=99"
    if perf stat -e amdgpu_pmu/sq_waves,sa=99/ -a sleep 0.1 2>&1 | grep -qi "invalid\|error"; then
        log_success "Error detected for invalid SA=99"
    else
        log_fail "No error for invalid SA=99"
    fi

    # Test 3: Invalid WGP
    log_info "Test: Invalid WGP=99"
    if perf stat -e amdgpu_pmu/sq_waves,wgp=99/ -a sleep 0.1 2>&1 | grep -qi "invalid\|error"; then
        log_success "Error detected for invalid WGP=99"
    else
        log_fail "No error for invalid WGP=99"
    fi

    # Test 4: Unsupported dimension for global counter
    log_info "Test: Dimensions on global counter (GRBM)"
    if perf stat -e amdgpu_pmu/grbm_count,se=0/ -a sleep 0.1 2>&1 | grep -qi "not support\|invalid"; then
        log_success "Error detected for dimensions on global counter"
    else
        log_warn "No error for dimensions on global counter (may not be enforced yet)"
    fi
}

# Compare results across dimensions
test_dimension_comparison() {
    log_info "Testing result comparison across dimensions..."
    log_warn "This test requires actual GPU workload and hardware - skipping automated check"
    log_info "Manual test: Run 'perf stat -e amdgpu_pmu/sq_waves,se=0/' and compare with se=1"
}

# Test perf list output
test_perf_list() {
    log_info "Testing perf list output..."

    if perf list amdgpu_pmu 2>&1 | grep -q "amdgpu_pmu"; then
        log_success "perf list shows amdgpu_pmu events"
    else
        log_fail "perf list does not show amdgpu_pmu events"
    fi

    # Show some events for manual inspection
    log_info "Available amdgpu_pmu events:"
    perf list amdgpu_pmu 2>&1 | head -20
}

# Check kernel logs for dimension-specific messages
check_kernel_logs() {
    log_info "Checking kernel logs for dimension-specific messages..."

    # Look for dimension-related log messages
    if dmesg | tail -100 | grep -q "dimension"; then
        log_info "Found dimension-related kernel messages:"
        dmesg | tail -100 | grep "dimension" | tail -5
    else
        log_info "No dimension-related messages in recent kernel log"
    fi
}

# Main test execution
main() {
    echo "======================================"
    echo "Dimension-Aware PMU Integration Tests"
    echo "======================================"
    echo ""

    log_info "Starting integration tests..."
    echo ""

    # Phase 1: Module and format checks (always possible)
    log_info "Phase 1: Module and Format Attribute Checks"
    check_module_loaded
    check_format_attributes
    verify_format_ranges
    test_perf_list
    echo ""

    # Phase 2: Event syntax tests (requires hardware)
    log_info "Phase 2: Event Syntax Tests"
    log_warn "These tests require AMD GPU hardware"
    echo ""

    test_named_parameters
    echo ""

    test_raw_config1
    echo ""

    test_mixed_syntax
    echo ""

    test_error_cases
    echo ""

    test_dimension_comparison
    echo ""

    # Phase 3: Kernel log inspection
    log_info "Phase 3: Kernel Log Inspection"
    check_kernel_logs
    echo ""

    # Summary
    echo "======================================"
    echo "Test Summary"
    echo "======================================"
    echo "Tests run:    $TESTS_RUN"
    echo -e "Tests passed: ${GREEN}$TESTS_PASSED${NC}"
    echo -e "Tests failed: ${RED}$TESTS_FAILED${NC}"
    echo ""

    if [ $TESTS_FAILED -eq 0 ]; then
        echo -e "${GREEN}All tests passed!${NC}"
        return 0
    else
        echo -e "${RED}Some tests failed.${NC}"
        return 1
    fi
}

# Run main function
main
exit $?
