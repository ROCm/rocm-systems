#!/bin/bash
###############################################################################
# GPU Parallel Test Execution Benchmark Script
#
# This script compares sequential vs parallel test execution and measures
# the performance improvement from GPU-aware scheduling.
#
# Usage:
#   ./benchmark_parallel_execution.sh [test_filter]
#
# Example:
#   ./benchmark_parallel_execution.sh "ParallelExecution.BasicSweep"
#   ./benchmark_parallel_execution.sh "AllReduce.*"
###############################################################################

set -e

# Color output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Default test filter
TEST_FILTER="${1:-ParallelExecution.BasicSweep}"
TEST_BINARY="${2:-./rccl-UnitTests}"

# Check if test binary exists
if [ ! -f "$TEST_BINARY" ]; then
    echo -e "${RED}Error: Test binary not found: $TEST_BINARY${NC}"
    echo "Please build the tests first or specify the correct path:"
    echo "  ./benchmark_parallel_execution.sh <test_filter> <path_to_binary>"
    exit 1
fi

echo -e "${BLUE}============================================${NC}"
echo -e "${BLUE}RCCL GPU Parallel Execution Benchmark${NC}"
echo -e "${BLUE}============================================${NC}"
echo ""
echo "Test filter: $TEST_FILTER"
echo "Test binary: $TEST_BINARY"
echo ""

# Create temporary log files
BASELINE_LOG=$(mktemp /tmp/rccl_baseline.XXXXXX.log)
PARALLEL_LOG=$(mktemp /tmp/rccl_parallel.XXXXXX.log)

cleanup() {
    rm -f "$BASELINE_LOG" "$PARALLEL_LOG"
}
trap cleanup EXIT

# Function to run tests and capture output
run_test() {
    local mode=$1
    local log_file=$2
    local env_vars=$3

    echo -e "${YELLOW}Running in $mode mode...${NC}"

    # Run with time measurement
    if eval "$env_vars /usr/bin/time -v $TEST_BINARY --gtest_filter=\"$TEST_FILTER\" > $log_file 2>&1"; then
        echo -e "${GREEN}✓ Tests passed${NC}"
        return 0
    else
        echo -e "${RED}✗ Tests failed${NC}"
        echo "See log: $log_file"
        return 1
    fi
}

# Run baseline (sequential) tests
echo ""
echo -e "${BLUE}[1/2] Running Sequential Execution (Baseline)${NC}"
echo "--------------------------------------------"
if run_test "SEQUENTIAL" "$BASELINE_LOG" "UT_PARALLEL_TESTS=0"; then
    BASELINE_SUCCESS=1
else
    BASELINE_SUCCESS=0
fi

echo ""

# Run parallel tests
echo -e "${BLUE}[2/2] Running Parallel Execution${NC}"
echo "--------------------------------------------"
if run_test "PARALLEL" "$PARALLEL_LOG" "UT_PARALLEL_TESTS=1 UT_PARALLEL_VERBOSE=1"; then
    PARALLEL_SUCCESS=1
else
    PARALLEL_SUCCESS=0
fi

echo ""
echo -e "${BLUE}============================================${NC}"
echo -e "${BLUE}Results${NC}"
echo -e "${BLUE}============================================${NC}"
echo ""

# Extract timing information
extract_time() {
    local log_file=$1
    # Extract wall clock time from /usr/bin/time output
    local time_str=$(grep "Elapsed (wall clock) time" "$log_file" | awk '{print $NF}')

    # Convert time to seconds (format: H:MM:SS or M:SS.ss)
    local hours=0
    local minutes=0
    local seconds=0

    if [[ $time_str =~ ^([0-9]+):([0-9]+):([0-9.]+)$ ]]; then
        hours=${BASH_REMATCH[1]}
        minutes=${BASH_REMATCH[2]}
        seconds=${BASH_REMATCH[3]}
    elif [[ $time_str =~ ^([0-9]+):([0-9.]+)$ ]]; then
        minutes=${BASH_REMATCH[1]}
        seconds=${BASH_REMATCH[2]}
    else
        seconds=$time_str
    fi

    # Convert to total seconds
    echo "$hours * 3600 + $minutes * 60 + $seconds" | bc
}

# Extract GPU utilization from parallel run
extract_gpu_util() {
    local log_file=$1
    grep "Average GPU utilization" "$log_file" | awk '{print $4}' | tr -d '%' || echo "N/A"
}

# Extract test counts
extract_test_count() {
    local log_file=$1
    grep "Total tests executed" "$log_file" | awk '{print $4}' || echo "N/A"
}

if [ $BASELINE_SUCCESS -eq 1 ]; then
    BASELINE_TIME=$(extract_time "$BASELINE_LOG")
    echo -e "${GREEN}Sequential Execution:${NC}"
    echo "  Wall time: ${BASELINE_TIME}s"
    echo "  Avg GPU utilization: ~20-30% (estimated)"
else
    echo -e "${RED}Sequential execution failed - see $BASELINE_LOG${NC}"
fi

echo ""

if [ $PARALLEL_SUCCESS -eq 1 ]; then
    PARALLEL_TIME=$(extract_time "$PARALLEL_LOG")
    PARALLEL_GPU_UTIL=$(extract_gpu_util "$PARALLEL_LOG")
    PARALLEL_TESTS=$(extract_test_count "$PARALLEL_LOG")

    echo -e "${GREEN}Parallel Execution:${NC}"
    echo "  Wall time: ${PARALLEL_TIME}s"
    echo "  Avg GPU utilization: ${PARALLEL_GPU_UTIL}%"
    echo "  Tests executed: $PARALLEL_TESTS"
else
    echo -e "${RED}Parallel execution failed - see $PARALLEL_LOG${NC}"
fi

echo ""

# Calculate speedup
if [ $BASELINE_SUCCESS -eq 1 ] && [ $PARALLEL_SUCCESS -eq 1 ]; then
    SPEEDUP=$(echo "scale=2; $BASELINE_TIME / $PARALLEL_TIME" | bc)

    echo -e "${BLUE}Performance Improvement:${NC}"
    echo "  Speedup: ${SPEEDUP}x"

    # Show percentage improvement
    IMPROVEMENT=$(echo "scale=1; ($BASELINE_TIME - $PARALLEL_TIME) / $BASELINE_TIME * 100" | bc)
    echo "  Time saved: ${IMPROVEMENT}%"

    echo ""

    if (( $(echo "$SPEEDUP > 2.0" | bc -l) )); then
        echo -e "${GREEN}⚡ Excellent speedup!${NC}"
    elif (( $(echo "$SPEEDUP > 1.5" | bc -l) )); then
        echo -e "${GREEN}✓ Good speedup${NC}"
    elif (( $(echo "$SPEEDUP > 1.1" | bc -l) )); then
        echo -e "${YELLOW}⚠ Modest speedup - check test distribution${NC}"
    else
        echo -e "${YELLOW}⚠ Limited speedup - tests may be mostly 8-GPU${NC}"
    fi
fi

echo ""
echo -e "${BLUE}============================================${NC}"
echo -e "${BLUE}Detailed Logs${NC}"
echo -e "${BLUE}============================================${NC}"
echo ""
echo "Sequential log: $BASELINE_LOG"
echo "Parallel log:   $PARALLEL_LOG"
echo ""
echo "To view scheduling details:"
echo "  grep -A 20 'GPU Scheduling Statistics' $PARALLEL_LOG"
echo ""
echo "To view GPU allocations:"
echo "  grep 'Allocated GPUs\\|Released GPUs' $PARALLEL_LOG"
echo ""

# Optional: Show summary if verbose
if [ "${VERBOSE:-0}" = "1" ]; then
    echo -e "${BLUE}GPU Scheduling Statistics:${NC}"
    grep -A 10 "GPU Scheduling Statistics" "$PARALLEL_LOG" || echo "Statistics not found"
    echo ""
fi
