#!/usr/bin/env bash
# Integration test for simdojo find_package() support.
# Builds and installs simdojo as both STATIC and SHARED, then configures
# an out-of-tree consumer project against each installed package.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
SIMDOJO_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"
WORK_DIR="${SCRIPT_DIR}/_build"

pass=0
fail=0

run_variant() {
    local variant="$1"        # "static" or "shared"
    local shared_flag="$2"    # "OFF" or "ON"

    local install_dir="${WORK_DIR}/install-${variant}"
    local build_lib="${WORK_DIR}/build-simdojo-${variant}"
    local build_test="${WORK_DIR}/build-test-${variant}"

    echo "==== ${variant} ===="

    # Build and install simdojo.
    cmake -S "$SIMDOJO_DIR" -B "$build_lib" \
        -G Ninja \
        -DCMAKE_BUILD_TYPE=Release \
        -DBUILD_SHARED_LIBS="${shared_flag}" \
        -DCMAKE_INSTALL_PREFIX="$install_dir" \
        > /dev/null
    cmake --build "$build_lib" --target install > /dev/null

    # Build the integration test against the installed package.
    cmake -S "$SCRIPT_DIR" -B "$build_test" \
        -G Ninja \
        -DCMAKE_BUILD_TYPE=Release \
        -DCMAKE_PREFIX_PATH="$install_dir" \
        > /dev/null
    cmake --build "$build_test" > /dev/null

    # Run it.
    if ctest --test-dir "$build_test" --output-on-failure; then
        echo "  PASS (${variant})"
        pass=$((pass + 1))
    else
        echo "  FAIL (${variant})"
        fail=$((fail + 1))
    fi
    echo
}

run_variant "static" "OFF"
run_variant "shared" "ON"

echo "==== Results: ${pass} passed, ${fail} failed ===="
[ "$fail" -eq 0 ]
