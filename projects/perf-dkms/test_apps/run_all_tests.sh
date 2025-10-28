#!/bin/bash

# Script to run all perf-dkms test applications
# This runs each test sequentially and reports results

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Get the directory where the script is located
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BUILD_DIR="${SCRIPT_DIR}/build"

# Check if build directory exists
if [ ! -d "${BUILD_DIR}" ]; then
    echo -e "${RED}Error: Build directory not found at ${BUILD_DIR}${NC}"
    echo "Please build the tests first:"
    echo "  cd ${SCRIPT_DIR}"
    echo "  mkdir build && cd build"
    echo "  cmake .."
    echo "  make"
    exit 1
fi

# List of all test executables
TESTS=(
    "valu_heavy/valu_heavy"
    "salu_heavy/salu_heavy"
    "lds_basic/lds_basic"
    "lds_conflicts/lds_conflicts"
    "smem/smem"
    "buffer_read/buffer_read"
    "buffer_write/buffer_write"
    "flat_memory/flat_memory"
    "mem_sizes/mem_sizes"
    "cache_hit/cache_hit"
    "cache_miss/cache_miss"
    "wave_modes/wave_modes"
    "waits_stalls/waits_stalls"
    "basic_activity/basic_activity"
)

# Test descriptions
declare -A TEST_DESCRIPTIONS=(
    ["valu_heavy"]="VALU Heavy - FMA Operations"
    ["salu_heavy"]="SALU Heavy - Scalar Operations"
    ["lds_basic"]="LDS Basic - Sequential Access"
    ["lds_conflicts"]="LDS Conflicts - Bank Conflicts"
    ["smem"]="SMEM - Scalar Memory (Constant Memory)"
    ["buffer_read"]="Buffer Read - Coalesced Memory Reads"
    ["buffer_write"]="Buffer Write - Coalesced Memory Writes"
    ["flat_memory"]="FLAT Memory - Generic Pointer Operations"
    ["mem_sizes"]="Memory Sizes - 32B/64B/128B Transactions"
    ["cache_hit"]="Cache Hit - L2 Cache Hits"
    ["cache_miss"]="Cache Miss - L2 Cache Misses"
    ["wave_modes"]="Wave Modes - Wave32/Wave64 Operations"
    ["waits_stalls"]="Waits/Stalls - Memory Dependencies"
    ["basic_activity"]="Basic Activity - Baseline GPU Activity"
)

# Counters
TOTAL_TESTS=${#TESTS[@]}
PASSED_TESTS=0
FAILED_TESTS=0
declare -a FAILED_TEST_NAMES

echo -e "${BLUE}================================================================================${NC}"
echo -e "${BLUE}  AMD GPU Performance Counter Test Suite${NC}"
echo -e "${BLUE}================================================================================${NC}"
echo ""
echo "Running ${TOTAL_TESTS} tests from: ${BUILD_DIR}"
echo ""

# Run each test
TEST_NUM=1
for TEST_PATH in "${TESTS[@]}"; do
    TEST_EXECUTABLE="${BUILD_DIR}/${TEST_PATH}"
    TEST_NAME=$(basename "${TEST_PATH}")
    TEST_DESC="${TEST_DESCRIPTIONS[${TEST_NAME}]}"

    echo -e "${BLUE}[${TEST_NUM}/${TOTAL_TESTS}]${NC} Running: ${YELLOW}${TEST_NAME}${NC}"
    echo -e "      Description: ${TEST_DESC}"

    if [ ! -f "${TEST_EXECUTABLE}" ]; then
        echo -e "${RED}      ERROR: Executable not found: ${TEST_EXECUTABLE}${NC}"
        FAILED_TESTS=$((FAILED_TESTS + 1))
        FAILED_TEST_NAMES+=("${TEST_NAME} (not found)")
        echo ""
        TEST_NUM=$((TEST_NUM + 1))
        continue
    fi

    # Run the test
    if "${TEST_EXECUTABLE}" > /dev/null 2>&1; then
        echo -e "${GREEN}      PASSED${NC}"
        PASSED_TESTS=$((PASSED_TESTS + 1))
    else
        EXIT_CODE=$?
        echo -e "${RED}      FAILED (exit code: ${EXIT_CODE})${NC}"
        FAILED_TESTS=$((FAILED_TESTS + 1))
        FAILED_TEST_NAMES+=("${TEST_NAME}")
    fi

    echo ""
    TEST_NUM=$((TEST_NUM + 1))
done

# Print summary
echo -e "${BLUE}================================================================================${NC}"
echo -e "${BLUE}  Test Summary${NC}"
echo -e "${BLUE}================================================================================${NC}"
echo ""
echo "Total tests:  ${TOTAL_TESTS}"
echo -e "Passed:       ${GREEN}${PASSED_TESTS}${NC}"
echo -e "Failed:       ${RED}${FAILED_TESTS}${NC}"
echo ""

if [ ${FAILED_TESTS} -gt 0 ]; then
    echo -e "${RED}Failed tests:${NC}"
    for FAILED_TEST in "${FAILED_TEST_NAMES[@]}"; do
        echo "  - ${FAILED_TEST}"
    done
    echo ""
    echo -e "${RED}Some tests failed!${NC}"
    exit 1
else
    echo -e "${GREEN}All tests passed successfully!${NC}"
    exit 0
fi
