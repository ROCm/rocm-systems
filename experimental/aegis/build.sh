#!/bin/bash
# AegisBit Build Script
# Usage: ./build.sh [clean|configure|build|test|e2e|test-all|all]

# this needs to be cleaned up and not use hardcoded paths, amd-staging probably not needed anymore
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BUILD_DIR="${SCRIPT_DIR}/build"

# Prefer amd-staging, fall back to upstream
if [ -d "${SCRIPT_DIR}/../llvm-project-amd-staging/build" ]; then
    LLVM_BUILD_DIR="${SCRIPT_DIR}/../llvm-project-amd-staging/build"
elif [ -d "${SCRIPT_DIR}/../llvm-project/build" ]; then
    LLVM_BUILD_DIR="${SCRIPT_DIR}/../llvm-project/build"
else
    echo "Error: No LLVM build found."
    echo "Looked for: llvm-project-amd-staging/build and llvm-project/build"
    exit 1
fi

# Parse command line
CMD="${1:-all}"

case "$CMD" in
    clean)
        echo "Cleaning build directory..."
        rm -rf "$BUILD_DIR"
        echo "Done."
        ;;

    configure)
        echo "Configuring build..."
        mkdir -p "$BUILD_DIR"
        cd "$BUILD_DIR"
        cmake -DLLVM_DIR="$LLVM_BUILD_DIR/lib/cmake/llvm" ..
        echo "Configuration complete."
        ;;

    build)
        if [ ! -f "$BUILD_DIR/Makefile" ] && [ ! -f "$BUILD_DIR/build.ninja" ]; then
            echo "Build not configured. Running configure first..."
            "$0" configure
        fi
        echo "Building..."
        cmake --build "$BUILD_DIR" -j16
        echo "Build complete."
        ;;

    test)
        "$0" build
        echo ""
        echo "Running C++ unit/integration tests..."
        cd "$BUILD_DIR"
        ctest --output-on-failure
        echo ""
        echo "C++ tests complete."
        ;;

    e2e)
        "$0" build
        echo ""
        echo "Running E2E tests..."
        python3 "$SCRIPT_DIR/test/run_e2e.py" "$@"
        echo ""
        echo "E2E tests complete."
        ;;

    test-all)
        "$0" test
        echo ""
        "$0" e2e
        ;;

    all)
        "$0" build
        "$0" test
        ;;

    *)
        echo "Usage: $0 [clean|configure|build|test|e2e|test-all|all]"
        echo ""
        echo "Commands:"
        echo "  clean     - Remove build directory"
        echo "  configure - Run cmake configuration"
        echo "  build     - Build the project"
        echo "  test      - Build and run C++ unit/integration tests"
        echo "  e2e       - Build and run E2E Python tests (GPU required)"
        echo "  test-all  - Run both C++ and E2E tests"
        echo "  all       - Build and run C++ tests (default)"
        exit 1
        ;;
esac
