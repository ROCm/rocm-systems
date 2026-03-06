#!/bin/bash
# RCCL Smoke Test - Optimized for 5-10 minute runtime
# Focuses on critical collective operations and data types
#
# Usage:
#   ./smoke_test.sh                                    - Run both Phase 1 and Phase 2
#   ./smoke_test.sh --phase2                           - Run Phase 2 only
#   ./smoke_test.sh --test-path /path/to/rccl-UnitTests - Specify test executable path
#   ./smoke_test.sh --phase2 --test-path /path/to/exe  - Combine options

set -e

# Parse command line arguments
PHASE2_ONLY=false
CUSTOM_TEST_PATH=""

while [[ $# -gt 0 ]]; do
    case $1 in
        --phase2|--phase2-only)
            PHASE2_ONLY=true
            shift
            ;;
        --test-path)
            CUSTOM_TEST_PATH="$2"
            shift 2
            ;;
        *)
            echo "Unknown option: $1"
            echo "Usage: $0 [--phase2] [--test-path /path/to/rccl-UnitTests]"
            exit 1
            ;;
    esac
done

# Color output for better visibility
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${BLUE}========================================${NC}"
echo -e "${BLUE}RCCL Smoke Test Configuration${NC}"
echo -e "${BLUE}========================================${NC}"

# Detect number of GPUs available using rocm-smi
NUM_GPUS_AVAILABLE=$(rocm-smi --showid 2>/dev/null | grep "GPU\[.*\].*Device Name" | wc -l)

# Fallback to 8 if detection failed
if [ -z "$NUM_GPUS_AVAILABLE" ] || [ "$NUM_GPUS_AVAILABLE" -eq 0 ]; then
    NUM_GPUS_AVAILABLE=8
fi

echo -e "${YELLOW}Detected $NUM_GPUS_AVAILABLE GPUs on system${NC}"
echo ""

# Test Configuration (Phase 1)
export UT_MIN_GPUS=2                    # Skip single GPU (no communication)
export UT_MAX_GPUS=$NUM_GPUS_AVAILABLE  # Test up to max available GPUs
export UT_POW2_GPUS=1                   # Only test power-of-2: 2, 4, 8, etc.
export UT_PROCESS_MASK=3
export UT_MAX_RANKS_PER_GPU=1           # One rank per GPU

# Critical data types for ML workloads
export UT_DATATYPES="ncclFloat32,ncclFloat16,ncclBfloat16,ncclInt32"

# Most common reduction operations
export UT_REDOPS="Sum,Max"

# Show configuration
export UT_SHOW_NAMES=1
export UT_SHOW_TIMING=1
export UT_VERBOSE=0

# HSA configuration
export HSA_NO_SCRATCH_RECLAIM=1   # Disable scratch memory reclaim for stability

echo -e "${YELLOW}Environment Variables:${NC}"
echo "  UT_MIN_GPUS        = $UT_MIN_GPUS"
echo "  UT_MAX_GPUS        = $UT_MAX_GPUS"
echo "  UT_POW2_GPUS       = $UT_POW2_GPUS"
echo "  UT_DATATYPES       = $UT_DATATYPES"
echo "  UT_REDOPS          = $UT_REDOPS"
echo "  UT_PROCESS_MASK    = $UT_PROCESS_MASK (2=multi-process only, production mode)"
echo ""
echo -e "${YELLOW}HSA Configuration:${NC}"
echo "  HSA_NO_SCRATCH_RECLAIM = $HSA_NO_SCRATCH_RECLAIM"
echo ""

# Find the test executable
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

if [ -n "$CUSTOM_TEST_PATH" ]; then
    # Use user-provided path
    UNIT_TEST_PATH="$CUSTOM_TEST_PATH"
    if [ ! -f "$UNIT_TEST_PATH" ]; then
        echo -e "\033[0;31mError: Specified test executable not found: $UNIT_TEST_PATH${NC}"
        exit 1
    fi
else
    # Assume rccl-UnitTests is in the same directory as this script
    UNIT_TEST_PATH="$SCRIPT_DIR/rccl-UnitTests"
    if [ ! -f "$UNIT_TEST_PATH" ]; then
        echo -e "\033[0;31mError: Cannot find rccl-UnitTests executable${NC}"
        echo "Expected location: $UNIT_TEST_PATH"
        echo ""
        echo "Please specify the path with --test-path:"
        echo "  $0 --test-path /path/to/rccl-UnitTests"
        exit 1
    fi
fi

echo -e "${GREEN}Using test executable: $UNIT_TEST_PATH${NC}"
echo ""

# Create log directory
LOG_DIR="/tmp/rccl_smoke_test_logs"
mkdir -p "$LOG_DIR"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
PHASE1_LOG="$LOG_DIR/phase1_${TIMESTAMP}.log"
PHASE2_LOG="$LOG_DIR/phase2_${TIMESTAMP}.log"

echo -e "${YELLOW}Logs will be saved to: $LOG_DIR${NC}"
echo ""

# Function to parse logs and find the failed/crashed test
find_failed_test() {
    local log_file=$1

    # Look for the currently running test when crash happened
    # Pattern: "[ RUN      ] TestSuite.TestName" without corresponding PASSED/FAILED
    local running_test=$(tac "$log_file" | grep -m1 "^\[ RUN      \]" | awk '{print $3}')

    if [ -n "$running_test" ]; then
        # Check if this test actually failed or just was running
        if grep -q "^\[  FAILED  \] $running_test" "$log_file" 2>/dev/null; then
            echo "$running_test"
            return 0
        elif grep -q "^\[       OK \] $running_test" "$log_file" 2>/dev/null; then
            # Test passed, look for actual failed tests
            local failed=$(grep "^\[  FAILED  \]" "$log_file" | head -1 | awk '{print $3}')
            echo "$failed"
            return 0
        else
            # Test was running but didn't finish (crash)
            echo "$running_test"
            return 0
        fi
    fi

    # Fallback: look for any failed test
    local failed=$(grep "^\[  FAILED  \]" "$log_file" | head -1 | awk '{print $3}')
    echo "$failed"
}

# Function to run tests in bulk and handle crashes
run_tests_with_monitoring() {
    local phase_name=$1
    local test_filter=$2
    local log_file=$3

    echo -e "${BLUE}Running tests in bulk...${NC}"
    echo ""

    # Run all tests in bulk and capture output
    $UNIT_TEST_PATH --gtest_filter="$test_filter" 2>&1 | tee "$log_file"
    local exit_code=${PIPESTATUS[0]}

    if [ $exit_code -ne 0 ]; then
        echo ""
        echo -e "\033[0;31m========================================${NC}"
        echo -e "\033[0;31mTESTS FAILED/CRASHED${NC}"
        echo -e "\033[0;31mExit code: $exit_code${NC}"
        echo -e "\033[0;31m========================================${NC}"

        # Parse log to find which test failed/crashed
        local failed_test=$(find_failed_test "$log_file")

        if [ -z "$failed_test" ]; then
            echo -e "\033[0;31mCould not determine which test failed from log${NC}"
            echo -e "\033[0;31m========================================${NC}"
            echo -e "\033[0;31mFull Phase Log:${NC}"
            echo -e "\033[0;31m========================================${NC}"
            cat "$log_file"
            echo -e "\033[0;31m========================================${NC}"
            return $exit_code
        fi

        echo ""
        echo -e "\033[0;31mFailed/Crashed test: $failed_test${NC}"
        echo ""
        echo -e "${YELLOW}Rerunning failed test with NCCL_DEBUG=INFO for diagnostics...${NC}"
        echo ""

        # Rerun the failed test with debug logging
        local debug_log="${log_file%.log}_${failed_test//\./_}_debug.log"
        export NCCL_DEBUG=INFO

        $UNIT_TEST_PATH --gtest_filter="$failed_test" 2>&1 | tee "$debug_log"
        local debug_exit=$?

        unset NCCL_DEBUG

        echo ""
        echo -e "\033[0;31m========================================${NC}"
        echo -e "\033[0;31mDebug rerun complete (exit code: $debug_exit)${NC}"
        echo -e "\033[0;31mFailed/Crashed test: $failed_test${NC}"
        echo -e "\033[0;31m========================================${NC}"
        echo ""
        echo -e "\033[0;31m========================================${NC}"
        echo -e "\033[0;31mFull Phase Log:${NC}"
        echo -e "\033[0;31m========================================${NC}"
        cat "$log_file"
        echo -e "\033[0;31m========================================${NC}"
        echo ""
        echo -e "\033[0;31m========================================${NC}"
        echo -e "\033[0;31mDebug Log (with NCCL_DEBUG=INFO):${NC}"
        echo -e "\033[0;31m========================================${NC}"
        cat "$debug_log"
        echo -e "\033[0;31m========================================${NC}"

        return $exit_code
    fi

    return 0
}


# Comprehensive collective operations testing
# All critical collective operations with OutOfPlace variant
# This provides full coverage of collective communication primitives

SMOKE_TESTS="AllReduce.OutOfPlace:\
AllReduce.InPlace:\
ReduceScatter.OutOfPlace:\
AllGather.OutOfPlace:\
Broadcast.OutOfPlace:\
Reduce.OutOfPlace:\
Scatter.OutOfPlace:\
Gather.OutOfPlace:\
AlltoAll.OutOfPlace:\
AlltoAllv.OutOfPlace:\
SendRecv.SinglePairs:\
GroupCall.Identical"

echo -e "${BLUE}Running Comprehensive Collective Tests:${NC}"
echo "  - AllReduce (Out-of-place + In-place) - gradient sync"
echo "  - ReduceScatter + AllGather - ring algorithm"
echo "  - Broadcast + Reduce - root-based collectives"
echo "  - Scatter + Gather - data distribution"
echo "  - AllToAll + AllToAllV - tensor parallelism"
echo "  - SendRecv - point-to-point"
echo "  - GroupCall - pipeline parallelism"
echo ""
echo -e "${BLUE}========================================${NC}"
echo ""

# Record start time
START_TIME=$(date +%s)

# PHASE 1: Critical collectives on 2, 4, 8 GPUs with both SP and MP
PHASE1_EXIT_CODE=0
if [ "$PHASE2_ONLY" = false ]; then
    echo -e "${BLUE}PHASE 1: Critical Collectives (2/4/8 GPUs, SP+MP)${NC}"
    echo ""

    # Run tests one-by-one with crash monitoring
    run_tests_with_monitoring "Phase 1" "$SMOKE_TESTS" "$PHASE1_LOG"
    PHASE1_EXIT_CODE=$?

    PHASE1_TIME=$(date +%s)
    PHASE1_ELAPSED=$((PHASE1_TIME - START_TIME))
    PHASE1_MIN=$((PHASE1_ELAPSED / 60))
    PHASE1_SEC=$((PHASE1_ELAPSED % 60))

    echo ""
    if [ $PHASE1_EXIT_CODE -eq 0 ]; then
        echo -e "${GREEN}Phase 1 Complete: ${PHASE1_MIN}m ${PHASE1_SEC}s${NC}"
    else
        echo -e "\033[0;31mPhase 1 FAILED: ${PHASE1_MIN}m ${PHASE1_SEC}s (exit code: $PHASE1_EXIT_CODE)\033[0m"
        exit $PHASE1_EXIT_CODE
    fi
    echo ""
else
    echo -e "${YELLOW}Skipping Phase 1 (--phase2 flag set)${NC}"
    echo ""
    PHASE1_TIME=$START_TIME
fi

# PHASE 2: All other tests on 8 GPUs MP only (no duplication)
echo -e "${BLUE}========================================${NC}"
echo -e "${BLUE}PHASE 2: Remaining Tests (8 GPUs, MP only)${NC}"
echo -e "${BLUE}========================================${NC}"
echo ""
echo -e "${YELLOW}Testing additional features on $NUM_GPUS_AVAILABLE GPUs (MP only):${NC}"
echo "  - HIP Graph variants"
echo "  - Managed Memory variants"
echo "  - Advanced GroupCall patterns"
echo "  - NonBlocking operations"
echo "  - Infrastructure tests"
echo ""

# Reconfigure for Phase 2: Max GPUs only, MP only, all data types
export UT_MIN_GPUS=$NUM_GPUS_AVAILABLE
export UT_MAX_GPUS=$NUM_GPUS_AVAILABLE
export UT_PROCESS_MASK=2  # Multi-process only
unset UT_DATATYPES  # Use all default data types for Phase 2

# Exclude Phase 1 tests to avoid duplication
# gtest syntax: -Test1:Test2:Test3 (only one minus sign at the start)
PHASE2_EXCLUSIONS="-AllReduce.OutOfPlace:\
AllReduce.InPlace:\
ReduceScatter.OutOfPlace:\
AllGather.OutOfPlace:\
Broadcast.OutOfPlace:\
Reduce.OutOfPlace:\
Scatter.OutOfPlace:\
Gather.OutOfPlace:\
AlltoAll.OutOfPlace:\
AlltoAllv.OutOfPlace:\
SendRecv.SinglePairs:\
GroupCall.Identical"

# Run all remaining tests one-by-one with crash monitoring
run_tests_with_monitoring "Phase 2" "$PHASE2_EXCLUSIONS" "$PHASE2_LOG"
PHASE2_EXIT_CODE=$?

# Record end time
END_TIME=$(date +%s)
PHASE2_ELAPSED=$((END_TIME - PHASE1_TIME))
PHASE2_MIN=$((PHASE2_ELAPSED / 60))
PHASE2_SEC=$((PHASE2_ELAPSED % 60))

TOTAL_ELAPSED=$((END_TIME - START_TIME))
TOTAL_MIN=$((TOTAL_ELAPSED / 60))
TOTAL_SEC=$((TOTAL_ELAPSED % 60))

echo ""
echo -e "${BLUE}========================================${NC}"

# Check Phase 2 results
if [ $PHASE2_EXIT_CODE -eq 0 ]; then
    echo -e "${GREEN}Smoke Test Completed Successfully!${NC}"
    if [ "$PHASE2_ONLY" = false ]; then
        echo -e "${GREEN}Phase 1 Runtime: ${PHASE1_MIN}m ${PHASE1_SEC}s (Critical collectives)${NC}"
    fi
    echo -e "${GREEN}Phase 2 Runtime: ${PHASE2_MIN}m ${PHASE2_SEC}s (Remaining tests)${NC}"
    echo -e "${GREEN}Total Runtime:   ${TOTAL_MIN}m ${TOTAL_SEC}s${NC}"
else
    echo -e "\033[0;31mPhase 2 FAILED (exit code: $PHASE2_EXIT_CODE)\033[0m"
    if [ "$PHASE2_ONLY" = false ]; then
        echo -e "${GREEN}Phase 1 Runtime: ${PHASE1_MIN}m ${PHASE1_SEC}s (PASSED)${NC}"
    fi
    echo -e "\033[0;31mPhase 2 Runtime: ${PHASE2_MIN}m ${PHASE2_SEC}s (FAILED)${NC}"
    echo -e "${YELLOW}Total Runtime:   ${TOTAL_MIN}m ${TOTAL_SEC}s${NC}"
fi

echo -e "${BLUE}========================================${NC}"

# Exit with appropriate code
if [ $PHASE1_EXIT_CODE -ne 0 ]; then
    exit $PHASE1_EXIT_CODE
elif [ $PHASE2_EXIT_CODE -ne 0 ]; then
    exit $PHASE2_EXIT_CODE
else
    exit 0
fi
