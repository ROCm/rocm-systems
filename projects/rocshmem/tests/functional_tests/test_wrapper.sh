#!/bin/bash
###############################################################################
# Copyright (c) Advanced Micro Devices, Inc. All rights reserved.
#
# SPDX-License-Identifier: MIT
###############################################################################

# CTest wrapper for rocshmem functional tests
# This script handles conditional test skipping based on backend type
# and other runtime conditions, matching the behavior of driver.sh

# CTest SKIP return code
SKIP_CODE=125

# Extract test name (first argument)
TEST_NAME=$1
shift

# Get backend type from environment or rocshmem_info
if [[ -n "$ROCSHMEM_BACKEND_TYPE" ]]; then
    BACKEND="$ROCSHMEM_BACKEND_TYPE"
elif [[ -n "$TEST" ]]; then
    # For compatibility with driver.sh TEST variable (e.g., "ro", "gda")
    BACKEND="$TEST"
else
    # Try to detect from rocshmem_info if available
    ROCSHMEM_INFO="$(dirname "$1")/rocshmem_info"
    if [[ ! -x "$ROCSHMEM_INFO" ]]; then
        # Try alternate locations
        ROCSHMEM_INFO="$(dirname "$1")/../../tools/rocshmem_info"
    fi
    if [[ -x "$ROCSHMEM_INFO" ]]; then
        BACKEND=$("$ROCSHMEM_INFO" | grep -i "backend" | awk '{print tolower($2)}')
    else
        BACKEND="default"
    fi
fi

echo "Test: $TEST_NAME (Backend: $BACKEND)"

# Apply skip conditions based on backend type and known issues
# These match the skip logic from driver.sh

# AIROCSHMEM-120: RO get tests abort
if [[ "$BACKEND" == ro* ]]; then
    case "$TEST_NAME" in
        get_*|getnbi_*|defaultctxget_*|defaultctxgetnbi_*|teamctxget_*|teamctxgetnbi_*|wgget_*|wggetnbi_*|waveget_*|wavegetnbi_*)
            echo "Skip: $TEST_NAME (AIROCSHMEM-120: RO get tests abort)"
            exit $SKIP_CODE
            ;;
    esac
fi

# AIROCSHMEM-162: GDA _g not implemented
if [[ "$BACKEND" == gda* ]]; then
    case "$TEST_NAME" in
        g_*|defaultctxg_*|flood_g_*)
            echo "Skip: $TEST_NAME (AIROCSHMEM-162: GDA _g not implemented)"
            exit $SKIP_CODE
            ;;
    esac
fi

# AIROCSHMEM-211: RO AMO operations abort
if [[ "$BACKEND" == ro* ]]; then
    case "$TEST_NAME" in
        amo_add_*|amo_fadd_*|amo_inc_*|amo_finc_*)
            echo "Skip: $TEST_NAME (AIROCSHMEM-211: RO amo abort)"
            exit $SKIP_CODE
            ;;
    esac
fi

# AIROCSHMEM-217: RO putmem_signal_on_stream sometimes abort
if [[ "$BACKEND" == ro* ]]; then
    case "$TEST_NAME" in
        putmem_signal_on_stream_*)
            echo "Skip: $TEST_NAME (AIROCSHMEM-217: RO sometimes abort)"
            exit $SKIP_CODE
            ;;
    esac
fi

# AIROCSHMEM-324: RO flood tests fail in UCX
if [[ "$BACKEND" == ro* ]]; then
    case "$TEST_NAME" in
        flood_*)
            echo "Skip: $TEST_NAME (AIROCSHMEM-324: RO flood tests fail in UCX)"
            exit $SKIP_CODE
            ;;
    esac
fi

# AIROCSHMEM-418: fence tests not supported on RO
if [[ "$BACKEND" == ro* ]]; then
    case "$TEST_NAME" in
        fence_*)
            echo "Skip: $TEST_NAME (AIROCSHMEM-418: fence tests not supported on RO)"
            exit $SKIP_CODE
            ;;
    esac
fi

# Check GPU availability
if command -v amd-smi >/dev/null && amd-smi version 2>&1 >/dev/null; then
    NUM_GPUS=$(amd-smi list | grep GPU | wc -l)
elif command -v rocm-smi >/dev/null && rocm-smi --version 2>&1 >/dev/null; then
    NUM_GPUS=$(rocm-smi --showserial | grep GPU | wc -l)
else
    NUM_GPUS=0
fi
NUM_GPUS=$((NUM_GPUS > 0 ? NUM_GPUS : 8))

# Extract number of ranks from test name (format: testname_n<ranks>_w<wg>_z<threads>)
if [[ "$TEST_NAME" =~ _n([0-9]+)_ ]]; then
    NUM_RANKS=${BASH_REMATCH[1]}

    # Skip if not enough GPUs (unless using hostfile)
    if [[ -z "$HOSTFILE" ]] && [[ $NUM_GPUS -lt $NUM_RANKS ]]; then
        echo "Skip: $TEST_NAME ($NUM_RANKS ranks required but only $NUM_GPUS GPUs available)"
        exit $SKIP_CODE
    fi
fi

# Setup log directory and file (matching driver.sh behavior)
# Use environment variable LOG_DIR if set, otherwise use current directory
LOG_DIR=${ROCSHMEM_TEST_LOG_DIR:-${LOG_DIR:-.}}
mkdir -p "$LOG_DIR"

LOG_FILE="$LOG_DIR/$TEST_NAME.log"

# Print command for debugging (matching driver.sh)
echo "# $@" > "$LOG_FILE"

# Execute the actual test command and capture output
"$@" >> "$LOG_FILE" 2>&1
TEST_EXIT_CODE=$?

# If test failed, show the log content (for CTest output)
if [ $TEST_EXIT_CODE -ne 0 ]; then
    echo "Test failed - see log: $LOG_FILE"
    cat "$LOG_FILE"
fi

exit $TEST_EXIT_CODE
