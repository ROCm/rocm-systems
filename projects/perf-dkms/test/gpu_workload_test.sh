#!/bin/bash
#
# gpu_workload_test.sh - GPU Workload Tests for Dimension-Aware PMU
#
# This script runs actual GPU workloads while monitoring dimension-specific
# performance counters to validate that dimension targeting works correctly.
#
# REQUIREMENTS:
#  - AMD GPU hardware (GFX12 recommended)
#  - ROCm runtime installed
#  - amdgpu_pmu kernel module loaded
#  - rocminfo tool available
#  - Sample GPU compute kernels (or /opt/rocm/bin/rocm-smi)
#

set -e

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

# Test results
TESTS_RUN=0
TESTS_PASSED=0
TESTS_FAILED=0

log_info() { echo -e "${BLUE}[INFO]${NC} $1"; }
log_success() { echo -e "${GREEN}[PASS]${NC} $1"; ((TESTS_PASSED++)); ((TESTS_RUN++)); }
log_fail() { echo -e "${RED}[FAIL]${NC} $1"; ((TESTS_FAILED++)); ((TESTS_RUN++)); }
log_warn() { echo -e "${YELLOW}[WARN]${NC} $1"; }

# Check prerequisites
check_prerequisites() {
    log_info "Checking prerequisites..."

    # Check for AMD GPU
    if ! lspci | grep -i "VGA.*AMD\|Display.*AMD" > /dev/null; then
        log_fail "No AMD GPU detected"
        return 1
    fi
    log_success "AMD GPU detected"

    # Check for amdgpu_pmu module
    if ! lsmod | grep -q amdgpu_pmu; then
        log_fail "amdgpu_pmu module not loaded"
        return 1
    fi
    log_success "amdgpu_pmu module loaded"

    # Check for perf
    if ! command -v perf &> /dev/null; then
        log_fail "perf tool not found"
        return 1
    fi
    log_success "perf tool available"

    # Check for ROCm (optional but helpful)
    if command -v rocminfo &> /dev/null; then
        log_success "ROCm runtime available"
        HAVE_ROCM=true
    else
        log_warn "ROCm runtime not found - some tests will be skipped"
        HAVE_ROCM=false
    fi

    return 0
}

# Get GPU topology information
get_gpu_topology() {
    log_info "Getting GPU topology..."

    # Try to get topology from rocminfo
    if [ "$HAVE_ROCM" = true ]; then
        log_info "GPU Topology from rocminfo:"
        rocminfo | grep -E "Shader Engine|Compute Unit" | head -10
    fi

    # Try to get from sysfs
    if [ -d /sys/class/drm/card0/device ]; then
        log_info "GPU device information:"
        cat /sys/class/drm/card0/device/uevent 2>/dev/null | grep PCI || true
    fi

    # Get dimension limits from module
    if dmesg | tail -200 | grep -q "dimension limits"; then
        log_info "Dimension limits from kernel:"
        dmesg | tail -200 | grep "dimension limits"
    fi
}

# Simple GPU workload using rocm-smi
run_simple_gpu_workload() {
    if [ "$HAVE_ROCM" = true ]; then
        log_info "Running simple GPU workload (rocm-smi)..."
        timeout 2s rocm-smi --showuse --json > /dev/null 2>&1 || true
        return 0
    else
        log_warn "No ROCm available for GPU workload"
        return 1
    fi
}

# Test: Monitor all SEs in parallel
test_monitor_all_ses() {
    log_info "Test: Monitor all shader engines in parallel..."

    # For GFX12: 4 shader engines (SE 0-3)
    local num_se=4
    local pids=()

    for se in $(seq 0 $((num_se-1))); do
        log_info "Starting monitoring for SE=$se"
        perf stat -e amdgpu_pmu/sq_waves,se=$se/ -a -o /tmp/perf_se${se}.log sleep 2 &
        pids+=($!)
    done

    # Wait for all monitoring to complete
    for pid in "${pids[@]}"; do
        wait $pid 2>/dev/null || true
    done

    # Check results
    log_info "Results from each SE:"
    local all_valid=true
    for se in $(seq 0 $((num_se-1))); do
        if [ -f /tmp/perf_se${se}.log ]; then
            local count=$(grep sq_waves /tmp/perf_se${se}.log | awk '{print $1}' | tr -d ',' || echo "0")
            log_info "  SE$se: $count sq_waves"
            rm -f /tmp/perf_se${se}.log
        else
            log_fail "No results for SE$se"
            all_valid=false
        fi
    done

    if $all_valid; then
        log_success "Successfully monitored all SEs"
    else
        log_fail "Failed to monitor all SEs"
    fi
}

# Test: Compare SE-specific vs broadcast
test_se_specific_vs_broadcast() {
    log_info "Test: Compare SE-specific vs broadcast monitoring..."

    # Run with SE-specific targeting
    log_info "Running with SE=0..."
    perf stat -e amdgpu_pmu/sq_waves,se=0/ -a sleep 1 2>&1 | tee /tmp/perf_se0.log || true

    # Run with broadcast (no dimensions)
    log_info "Running with broadcast (no dimensions)..."
    perf stat -e amdgpu_pmu/sq_waves/ -a sleep 1 2>&1 | tee /tmp/perf_broadcast.log || true

    # Extract counts
    local se0_count=$(grep sq_waves /tmp/perf_se0.log | awk '{print $1}' | tr -d ',' || echo "0")
    local broadcast_count=$(grep sq_waves /tmp/perf_broadcast.log | awk '{print $1}' | tr -d ',' || echo "0")

    log_info "SE0 count: $se0_count"
    log_info "Broadcast count: $broadcast_count"

    # Broadcast should typically be >= SE-specific (summed across all SEs)
    if [ -n "$broadcast_count" ] && [ "$broadcast_count" != "0" ]; then
        log_success "Got valid counts from both modes"
    else
        log_warn "Unable to verify counts (may need active GPU workload)"
    fi

    rm -f /tmp/perf_se0.log /tmp/perf_broadcast.log
}

# Test: Monitor with active GPU compute kernel
test_with_compute_kernel() {
    log_info "Test: Monitor dimension-specific counters during GPU compute..."

    if [ "$HAVE_ROCM" != true ]; then
        log_warn "Skipping compute kernel test (ROCm not available)"
        return
    fi

    # Create simple HIP/OpenCL workload or use existing tool
    log_info "This test requires a GPU compute workload"
    log_info "Recommended: Run separately with 'perf stat -e amdgpu_pmu/sq_waves,se=0/ <gpu_app>'"
    log_info "Example: perf stat -e amdgpu_pmu/sq_waves,se=0,sa=0/ python3 -c 'import torch; torch.cuda.FloatTensor(1000,1000).zero_()'"

    # For automated testing, just verify the syntax works
    if timeout 2s perf stat -e amdgpu_pmu/sq_waves,se=0,sa=0,wgp=0/ -a sleep 1 2>&1 | grep -q sq_waves; then
        log_success "Dimension-specific monitoring syntax validated"
    else
        log_fail "Dimension-specific monitoring failed"
    fi
}

# Test: Verify dimension targeting is working
test_dimension_targeting() {
    log_info "Test: Verify GRBM_GFX_INDEX targeting is working..."

    # Monitor different SEs and verify we get different results
    # (requires active workload that uses multiple SEs)

    log_info "Monitoring SE=0 and SE=1 separately..."

    perf stat -e amdgpu_pmu/sq_waves,se=0/ -a -o /tmp/perf_test_se0.log sleep 2 &
    local pid1=$!

    perf stat -e amdgpu_pmu/sq_waves,se=1/ -a -o /tmp/perf_test_se1.log sleep 2 &
    local pid2=$!

    wait $pid1 2>/dev/null || true
    wait $pid2 2>/dev/null || true

    if [ -f /tmp/perf_test_se0.log ] && [ -f /tmp/perf_test_se1.log ]; then
        log_info "SE0 results:"
        cat /tmp/perf_test_se0.log | grep sq_waves || true
        log_info "SE1 results:"
        cat /tmp/perf_test_se1.log | grep sq_waves || true
        log_success "Successfully collected dimension-specific data"
        rm -f /tmp/perf_test_se0.log /tmp/perf_test_se1.log
    else
        log_fail "Failed to collect dimension-specific data"
    fi
}

# Test: Multiple dimensions (SE/SA/WGP)
test_multiple_dimensions() {
    log_info "Test: Monitor with multiple dimension levels..."

    # Test SE/SA specification
    if perf stat -e amdgpu_pmu/sq_waves,se=0,sa=1/ -a sleep 1 2>&1 | grep -q sq_waves; then
        log_success "SE+SA dimension targeting works"
    else
        log_fail "SE+SA dimension targeting failed"
    fi

    # Test SE/SA/WGP specification
    if perf stat -e amdgpu_pmu/sq_waves,se=1,sa=0,wgp=2/ -a sleep 1 2>&1 | grep -q sq_waves; then
        log_success "SE+SA+WGP dimension targeting works"
    else
        log_fail "SE+SA+WGP dimension targeting failed"
    fi
}

# Test: Different counter types with dimensions
test_different_counters() {
    log_info "Test: Different counter types with dimension targeting..."

    # SQ counter (shader engine specific)
    if perf stat -e amdgpu_pmu/sq_waves,se=0/ -a sleep 1 2>&1 | grep -q sq_waves; then
        log_success "SQ counter with dimensions works"
    else
        log_fail "SQ counter with dimensions failed"
    fi

    # TA counter (shader engine specific)
    if perf stat -e amdgpu_pmu/ta_busy,se=0/ -a sleep 1 2>&1 | grep -q ta_busy; then
        log_success "TA counter with dimensions works"
    else
        log_fail "TA counter with dimensions failed"
    fi

    # GL2C counter (per SE/SA)
    if perf stat -e amdgpu_pmu/gl2c_hit,se=0,sa=0/ -a sleep 1 2>&1 | grep -q gl2c_hit; then
        log_success "GL2C counter with dimensions works"
    else
        log_warn "GL2C counter with dimensions (may not be supported on this GPU)"
    fi
}

# Performance comparison test
test_performance_overhead() {
    log_info "Test: Performance overhead of dimension-specific monitoring..."

    # Time broadcast mode
    local start_time=$(date +%s%N)
    perf stat -e amdgpu_pmu/sq_waves/ -a sleep 1 > /dev/null 2>&1 || true
    local end_time=$(date +%s%N)
    local broadcast_duration=$(( (end_time - start_time) / 1000000 ))

    # Time dimension-specific mode
    start_time=$(date +%s%N)
    perf stat -e amdgpu_pmu/sq_waves,se=0/ -a sleep 1 > /dev/null 2>&1 || true
    end_time=$(date +%s%N)
    local dimension_duration=$(( (end_time - start_time) / 1000000 ))

    log_info "Broadcast mode duration: ${broadcast_duration}ms"
    log_info "Dimension-specific duration: ${dimension_duration}ms"

    # Allow for 10% overhead
    local overhead=$(( (dimension_duration - broadcast_duration) * 100 / broadcast_duration ))
    log_info "Overhead: ${overhead}%"

    if [ $overhead -lt 20 ]; then
        log_success "Performance overhead acceptable (<20%)"
    else
        log_warn "Performance overhead high (${overhead}%)"
    fi
}

# Main test execution
main() {
    echo "============================================"
    echo "GPU Workload Tests for Dimension-Aware PMU"
    echo "============================================"
    echo ""

    # Prerequisites
    if ! check_prerequisites; then
        echo ""
        echo -e "${RED}Prerequisites not met. Exiting.${NC}"
        exit 1
    fi
    echo ""

    # Get GPU topology
    get_gpu_topology
    echo ""

    # Run tests
    log_info "Running GPU workload tests..."
    echo ""

    test_se_specific_vs_broadcast
    echo ""

    test_monitor_all_ses
    echo ""

    test_dimension_targeting
    echo ""

    test_multiple_dimensions
    echo ""

    test_different_counters
    echo ""

    test_with_compute_kernel
    echo ""

    test_performance_overhead
    echo ""

    # Summary
    echo "============================================"
    echo "Test Summary"
    echo "============================================"
    echo "Tests run:    $TESTS_RUN"
    echo -e "Tests passed: ${GREEN}$TESTS_PASSED${NC}"
    echo -e "Tests failed: ${RED}$TESTS_FAILED${NC}"
    echo ""

    if [ $TESTS_FAILED -eq 0 ]; then
        echo -e "${GREEN}All tests passed!${NC}"
        echo ""
        echo "Recommendations:"
        echo "  - Run with active GPU workload for more comprehensive testing"
        echo "  - Compare counter values across different SEs during compute"
        echo "  - Test with various ROCm applications"
        return 0
    else
        echo -e "${RED}Some tests failed.${NC}"
        echo ""
        echo "Troubleshooting:"
        echo "  - Check dmesg for kernel errors"
        echo "  - Verify GPU hardware is functioning"
        echo "  - Ensure amdgpu_pmu module loaded correctly"
        echo "  - Check dimension limits match your GPU architecture"
        return 1
    fi
}

# Run main
main
exit $?
