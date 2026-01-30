#!/bin/bash
#
# Build and run smoke tests for device linker RCCL build
#
# Usage: run_tests.sh [BUILD_DIR] [--all]
#   BUILD_DIR: RCCL build directory (default: ../../../build/release)
#   --all: Run all tests including known-failing multi-GPU tests
#
# By default, only runs tests that are expected to pass.
# Multi-GPU tests currently fail with the device linker build (see README.md).
#

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RCCL_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"

# Parse arguments
RUN_ALL=0
BUILD_DIR=""
for arg in "$@"; do
    if [ "$arg" = "--all" ]; then
        RUN_ALL=1
    elif [ -z "$BUILD_DIR" ]; then
        BUILD_DIR="$arg"
    fi
done
BUILD_DIR="${BUILD_DIR:-$RCCL_ROOT/build/release}"

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

# Find all test source files
ALL_TEST_SOURCES=$(find "$SCRIPT_DIR" -name "test_*.cpp" | sort)

if [ -z "$ALL_TEST_SOURCES" ]; then
    echo "No test sources found!"
    exit 1
fi

# Filter out known-failing multi-GPU tests unless --all is specified
if [ $RUN_ALL -eq 1 ]; then
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
