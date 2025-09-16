#!/bin/bash
"""
Automated Test Runner for AQL Packet Comparison

This script runs both dumper programs with various configurations and
compares their outputs to validate aql_c compatibility with aqlprofile_v2.
"""

set -e  # Exit on any error

# Configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
AQL_C_DUMPER="$SCRIPT_DIR/aql_c_dumper"
AQLPROFILE_V2_DUMPER="$SCRIPT_DIR/aqlprofile_v2_dumper"
PACKET_COMPARE="$SCRIPT_DIR/packet_compare.py"
TEST_RESULTS_DIR="$SCRIPT_DIR/test_results"
REFERENCE_RESULTS_DIR="$SCRIPT_DIR/reference_results"

# Test configurations
declare -a ARCHITECTURES=("gfx942" "gfx1101" "gfx1102" "gfx1200")
declare -a SINGLE_COUNTER_TESTS=(
    "CPC:0:123"
    "GRBM:0:456"
    "SQ:0:789"
    "TA:0:100"
    "TD:0:200"
)
declare -a MULTI_COUNTER_TESTS=(
    "CPC:0:123,GRBM:0:456"
    "SQ:0:789,TA:0:100,TD:0:200"
    "CPC:0:123,CPC:1:124,CPC:2:125"
)

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Global counters
TOTAL_TESTS=0
PASSED_TESTS=0
FAILED_TESTS=0

# Usage information
usage() {
    echo "Usage: $0 [OPTIONS]"
    echo "Options:"
    echo "  --arch ARCH         Test specific architecture (default: all)"
    echo "  --test-suite SUITE  Test suite: basic, full, stress (default: basic)"
    echo "  --parallel N        Run tests in parallel (default: 1)"
    echo "  --clean             Clean test results before running"
    echo "  --verbose           Enable verbose output"
    echo "  --help              Show this help"
    echo ""
    echo "Test Suites:"
    echo "  basic    - Single counter tests per architecture"
    echo "  full     - Single and multi-counter tests"
    echo "  stress   - Maximum configuration stress tests"
    echo ""
    echo "Examples:"
    echo "  $0 --arch gfx942 --test-suite basic"
    echo "  $0 --test-suite full --parallel 4"
    echo "  $0 --clean --test-suite stress --verbose"
}

# Logging functions
log_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

log_success() {
    echo -e "${GREEN}[PASS]${NC} $1"
}

log_warning() {
    echo -e "${YELLOW}[WARN]${NC} $1"
}

log_error() {
    echo -e "${RED}[FAIL]${NC} $1"
}

log_test_start() {
    echo -e "${BLUE}[TEST]${NC} $1"
}

# Check prerequisites
check_prerequisites() {
    log_info "Checking prerequisites..."

    if [[ ! -x "$AQL_C_DUMPER" ]]; then
        log_error "AQL_C dumper not found or not executable: $AQL_C_DUMPER"
        log_info "Please build the dumper programs first:"
        log_info "  make aql_c_dumper"
        exit 1
    fi

    if [[ ! -x "$AQLPROFILE_V2_DUMPER" ]]; then
        log_error "AQLProfile v2 dumper not found or not executable: $AQLPROFILE_V2_DUMPER"
        log_info "Please build the dumper programs first:"
        log_info "  make aqlprofile_v2_dumper"
        exit 1
    fi

    if [[ ! -x "$PACKET_COMPARE" ]]; then
        log_error "Packet compare script not found or not executable: $PACKET_COMPARE"
        chmod +x "$PACKET_COMPARE" 2>/dev/null || true
        if [[ ! -x "$PACKET_COMPARE" ]]; then
            exit 1
        fi
    fi

    # Check Python availability
    if ! command -v python3 &> /dev/null; then
        log_error "Python 3 is required for packet comparison"
        exit 1
    fi

    log_success "All prerequisites satisfied"
}

# Setup test directories
setup_test_dirs() {
    log_info "Setting up test directories..."

    if [[ "$CLEAN_TESTS" == "1" ]]; then
        rm -rf "$TEST_RESULTS_DIR" "$REFERENCE_RESULTS_DIR"
    fi

    mkdir -p "$TEST_RESULTS_DIR" "$REFERENCE_RESULTS_DIR"

    log_success "Test directories ready"
}

# Generate test name
generate_test_name() {
    local arch="$1"
    local counters="$2"
    local test_type="$3"

    # Replace special characters for filename
    local clean_counters=$(echo "$counters" | tr ':,' '_')
    echo "${test_type}_${arch}_${clean_counters}"
}

# Run single test case
run_test_case() {
    local arch="$1"
    local counters="$2"
    local test_type="$3"

    local test_name=$(generate_test_name "$arch" "$counters" "$test_type")
    local aql_c_output="$TEST_RESULTS_DIR/${test_name}_aqlc.json"
    local aqlprofile_output="$REFERENCE_RESULTS_DIR/${test_name}_aqlv2.json"

    log_test_start "Running $test_name"

    TOTAL_TESTS=$((TOTAL_TESTS + 1))

    # Convert comma-separated counters to multiple --counter arguments
    local counter_args=""
    IFS=',' read -ra COUNTER_ARRAY <<< "$counters"
    for counter in "${COUNTER_ARRAY[@]}"; do
        counter_args="$counter_args --counter $counter"
    done

    # Run AQL_C dumper
    if [[ "$VERBOSE" == "1" ]]; then
        log_info "Running AQL_C dumper: $AQL_C_DUMPER --arch $arch $counter_args --output $aql_c_output"
    fi

    if ! $AQL_C_DUMPER --arch "$arch" $counter_args --output "$aql_c_output" 2>/dev/null; then
        log_error "AQL_C dumper failed for $test_name"
        FAILED_TESTS=$((FAILED_TESTS + 1))
        return 1
    fi

    # Run AQLProfile v2 dumper
    if [[ "$VERBOSE" == "1" ]]; then
        log_info "Running AQLProfile v2 dumper: $AQLPROFILE_V2_DUMPER --arch $arch $counter_args --output $aqlprofile_output"
    fi

    if ! $AQLPROFILE_V2_DUMPER --arch "$arch" $counter_args --output "$aqlprofile_output" 2>/dev/null; then
        log_error "AQLProfile v2 dumper failed for $test_name"
        FAILED_TESTS=$((FAILED_TESTS + 1))
        return 1
    fi

    # Compare outputs
    if [[ "$VERBOSE" == "1" ]]; then
        log_info "Comparing outputs: $PACKET_COMPARE $aql_c_output $aqlprofile_output"
    fi

    if $PACKET_COMPARE "$aql_c_output" "$aqlprofile_output" > "/dev/null" 2>&1; then
        log_success "Test $test_name PASSED"
        PASSED_TESTS=$((PASSED_TESTS + 1))
        return 0
    else
        log_error "Test $test_name FAILED - differences found"
        if [[ "$VERBOSE" == "1" ]]; then
            log_info "Running detailed comparison..."
            $PACKET_COMPARE "$aql_c_output" "$aqlprofile_output" --detailed || true
        fi
        FAILED_TESTS=$((FAILED_TESTS + 1))
        return 1
    fi
}

# Run test suite for a specific architecture
run_architecture_tests() {
    local arch="$1"
    local test_suite="$2"

    log_info "Running $test_suite test suite for $arch"

    case "$test_suite" in
        "basic")
            # Single counter tests
            for counter in "${SINGLE_COUNTER_TESTS[@]}"; do
                run_test_case "$arch" "$counter" "single"
            done
            ;;

        "full")
            # Single counter tests
            for counter in "${SINGLE_COUNTER_TESTS[@]}"; do
                run_test_case "$arch" "$counter" "single"
            done

            # Multi-counter tests
            for counters in "${MULTI_COUNTER_TESTS[@]}"; do
                run_test_case "$arch" "$counters" "multi"
            done
            ;;

        "stress")
            # All basic and full tests
            run_architecture_tests "$arch" "full"

            # Stress tests with maximum counters
            local max_counters="CPC:0:123,CPC:1:124,GRBM:0:456,SQ:0:789,TA:0:100,TD:0:200,TCP:0:300,TCC:0:400"
            run_test_case "$arch" "$max_counters" "stress"

            # Edge case tests
            run_test_case "$arch" "CPC:3:999" "edge"  # High instance/event IDs
            ;;

        *)
            log_error "Unknown test suite: $test_suite"
            exit 1
            ;;
    esac
}

# Main test execution
run_tests() {
    local target_arch="$1"
    local test_suite="$2"

    log_info "Starting test execution"
    log_info "Test suite: $test_suite"
    log_info "Target architecture: ${target_arch:-all}"
    log_info "Parallel jobs: $PARALLEL_JOBS"

    if [[ -n "$target_arch" ]]; then
        # Test specific architecture
        run_architecture_tests "$target_arch" "$test_suite"
    else
        # Test all architectures
        for arch in "${ARCHITECTURES[@]}"; do
            if [[ "$PARALLEL_JOBS" -gt 1 ]]; then
                # Run in background for parallel execution
                run_architecture_tests "$arch" "$test_suite" &

                # Limit concurrent jobs
                while [[ $(jobs -r | wc -l) -ge $PARALLEL_JOBS ]]; do
                    sleep 0.1
                done
            else
                run_architecture_tests "$arch" "$test_suite"
            fi
        done

        # Wait for all background jobs to complete
        wait
    fi
}

# Generate test report
generate_report() {
    echo ""
    echo "=============================================="
    echo "           TEST EXECUTION SUMMARY"
    echo "=============================================="
    echo "Total tests run: $TOTAL_TESTS"
    echo "Passed: $PASSED_TESTS"
    echo "Failed: $FAILED_TESTS"
    echo "Success rate: $(( PASSED_TESTS * 100 / TOTAL_TESTS ))%"
    echo ""

    if [[ $FAILED_TESTS -gt 0 ]]; then
        echo "RESULT: FAILED - Some tests showed differences"
        echo ""
        echo "Next steps:"
        echo "1. Review failed test outputs in $TEST_RESULTS_DIR"
        echo "2. Run packet_compare.py with --detailed flag for analysis"
        echo "3. Investigate root causes of differences"
        echo "4. Update aql_c implementation if needed"
        echo ""
        return 1
    else
        echo "RESULT: SUCCESS - All tests passed!"
        echo ""
        echo "aql_c implementation appears to be compatible with aqlprofile_v2"
        return 0
    fi
}

# Parse command line arguments
ARCH=""
TEST_SUITE="basic"
PARALLEL_JOBS=1
CLEAN_TESTS=0
VERBOSE=0

while [[ $# -gt 0 ]]; do
    case $1 in
        --arch)
            ARCH="$2"
            shift 2
            ;;
        --test-suite)
            TEST_SUITE="$2"
            shift 2
            ;;
        --parallel)
            PARALLEL_JOBS="$2"
            shift 2
            ;;
        --clean)
            CLEAN_TESTS=1
            shift
            ;;
        --verbose)
            VERBOSE=1
            shift
            ;;
        --help)
            usage
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            usage
            exit 1
            ;;
    esac
done

# Validate arguments
if [[ -n "$ARCH" ]]; then
    if [[ ! " ${ARCHITECTURES[@]} " =~ " ${ARCH} " ]]; then
        log_error "Unknown architecture: $ARCH"
        log_info "Supported architectures: ${ARCHITECTURES[*]}"
        exit 1
    fi
fi

if [[ ! "$TEST_SUITE" =~ ^(basic|full|stress)$ ]]; then
    log_error "Unknown test suite: $TEST_SUITE"
    log_info "Supported test suites: basic, full, stress"
    exit 1
fi

if ! [[ "$PARALLEL_JOBS" =~ ^[0-9]+$ ]] || [[ "$PARALLEL_JOBS" -lt 1 ]] || [[ "$PARALLEL_JOBS" -gt 16 ]]; then
    log_error "Invalid parallel jobs count: $PARALLEL_JOBS (must be 1-16)"
    exit 1
fi

# Main execution
main() {
    log_info "AQL Packet Comparison Test Runner"
    log_info "=================================="

    check_prerequisites
    setup_test_dirs
    run_tests "$ARCH" "$TEST_SUITE"

    if generate_report; then
        exit 0
    else
        exit 1
    fi
}

# Run main function
main "$@"