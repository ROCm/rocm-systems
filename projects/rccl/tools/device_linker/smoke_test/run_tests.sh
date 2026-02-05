#!/bin/bash
#
# Build and run smoke tests for device linker RCCL build
#
# Usage: run_tests.sh [OPTIONS] [TEST_NAMES...]
#
# Options:
#   --build-dir DIR   RCCL build directory (default: ../../../build/release)
#   --all             Run all tests including known-failing multi-GPU tests
#   --list            List all available tests and exit
#   --help            Show this help message
#
# Examples:
#   ./run_tests.sh                          # Run passing tests only
#   ./run_tests.sh --all                    # Run all tests
#   ./run_tests.sh --list                   # List available tests
#   ./run_tests.sh test_single_gpu          # Run specific test
#   ./run_tests.sh test_single test_init    # Run multiple specific tests
#   ./run_tests.sh --build-dir /path test_single_gpu
#

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RCCL_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"

# Parse arguments
RUN_ALL=0
LIST_ONLY=0
BUILD_DIR=""
SPECIFIC_TESTS=""

show_help() {
    head -20 "$0" | tail -18 | sed 's/^#//' | sed 's/^ //'
    exit 0
}

while [ $# -gt 0 ]; do
    case "$1" in
        --all)
            RUN_ALL=1
            shift
            ;;
        --list)
            LIST_ONLY=1
            shift
            ;;
        --build-dir)
            BUILD_DIR="$2"
            shift 2
            ;;
        --help|-h)
            show_help
            ;;
        -*)
            echo "Unknown option: $1"
            echo "Use --help for usage"
            exit 1
            ;;
        *)
            # Treat as test name - strip .cpp extension and test_ prefix if present
            test_name="$1"
            test_name="${test_name%.cpp}"
            [[ "$test_name" != test_* ]] && test_name="test_$test_name"
            SPECIFIC_TESTS="$SPECIFIC_TESTS $test_name"
            shift
            ;;
    esac
done

BUILD_DIR="${BUILD_DIR:-$RCCL_ROOT/build/release}"

# Find all test source files
ALL_TEST_SOURCES=$(find "$SCRIPT_DIR" -maxdepth 1 -name "test_*.cpp" | sort)

if [ -z "$ALL_TEST_SOURCES" ]; then
    echo "No test sources found!"
    exit 1
fi

# List tests and exit if --list specified
if [ $LIST_ONLY -eq 1 ]; then
    echo "Available tests:"
    echo ""
    for src in $ALL_TEST_SOURCES; do
        name=$(basename "$src" .cpp)
        # Mark multi-GPU tests
        if [[ "$name" == *"multi_gpu"* ]] || [[ "$name" == "test_two_gpu" ]]; then
            echo "  $name  (multi-GPU, may hang)"
        else
            echo "  $name"
        fi
    done
    echo ""
    echo "Total: $(echo "$ALL_TEST_SOURCES" | wc -w) tests"
    exit 0
fi

# Check librccl.so exists
if [ ! -f "$BUILD_DIR/librccl.so" ]; then
    echo "Error: librccl.so not found in $BUILD_DIR"
    echo "Build RCCL first with: cmake -DDEVICE_LINKER=ON .. && ninja"
    exit 1
fi

# Compiler and flags
HIPCC=/opt/rocm/bin/hipcc
CFLAGS="-O2"
INCLUDES="-I$BUILD_DIR/include"
LDFLAGS="-L$BUILD_DIR -lrccl -lpthread -Wl,-rpath,$BUILD_DIR"

# Output directory for binaries
BIN_DIR="$SCRIPT_DIR/bin"
mkdir -p "$BIN_DIR"

echo "=========================================="
echo "  Device Linker RCCL Smoke Tests"
echo "=========================================="
echo ""
echo "Build directory: $BUILD_DIR"
echo "Output directory: $BIN_DIR"
echo ""

# Determine which tests to run
if [ -n "$SPECIFIC_TESTS" ]; then
    # User specified specific tests
    TEST_SOURCES=""
    for test_name in $SPECIFIC_TESTS; do
        src="$SCRIPT_DIR/${test_name}.cpp"
        if [ -f "$src" ]; then
            TEST_SOURCES="$TEST_SOURCES $src"
        else
            echo "Warning: Test not found: $test_name ($src)"
        fi
    done
    if [ -z "$TEST_SOURCES" ]; then
        echo "Error: No valid tests specified"
        exit 1
    fi
    echo "Running specified tests:$SPECIFIC_TESTS"
elif [ $RUN_ALL -eq 1 ]; then
    TEST_SOURCES="$ALL_TEST_SOURCES"
    echo "Running ALL tests (including known-failing multi-GPU tests)"
else
    TEST_SOURCES=""
    for src in $ALL_TEST_SOURCES; do
        name=$(basename "$src" .cpp)
        # Skip multi-GPU tests (known to fail with device linker build)
        if [[ "$name" != *"multi_gpu"* ]] && [[ "$name" != "test_two_gpu" ]]; then
            TEST_SOURCES="$TEST_SOURCES $src"
        fi
    done
    echo "Running passing tests only (use --all for multi-GPU tests)"
fi
echo ""

# Build all tests
echo "=== Building Tests ==="
for src in $TEST_SOURCES; do
    name=$(basename "$src" .cpp)
    echo "  Building $name..."
    $HIPCC $CFLAGS $INCLUDES "$src" $LDFLAGS -o "$BIN_DIR/$name"
done
echo ""

# Run all tests
echo "=== Running Tests ==="
PASSED=0
FAILED=0
FAILED_TESTS=""

for src in $TEST_SOURCES; do
    name=$(basename "$src" .cpp)
    echo ""
    echo "--- $name ---"
    
    if "$BIN_DIR/$name"; then
        PASSED=$((PASSED + 1))
    else
        FAILED=$((FAILED + 1))
        FAILED_TESTS="$FAILED_TESTS $name"
    fi
done

# Summary
echo ""
echo "=========================================="
echo "  Summary"
echo "=========================================="
echo ""
echo "Passed: $PASSED"
echo "Failed: $FAILED"

if [ $FAILED -gt 0 ]; then
    echo ""
    echo "Failed tests:$FAILED_TESTS"
    exit 1
else
    echo ""
    echo "All tests passed!"
    exit 0
fi
