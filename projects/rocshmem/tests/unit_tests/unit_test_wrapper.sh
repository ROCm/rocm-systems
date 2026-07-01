#!/bin/bash
###############################################################################
# Copyright (c) Advanced Micro Devices, Inc. All rights reserved.
#
# SPDX-License-Identifier: MIT
###############################################################################

# CTest wrapper for rocshmem unit tests
# This script handles conditional test skipping based on GPU availability

# CTest SKIP return code
SKIP_CODE=125

# Extract test name and required ranks
# Usage: unit_test_wrapper.sh <test_name> <required_ranks> mpirun -np <ranks> ... <test_executable> <args>
TEST_NAME=$1
REQUIRED_RANKS=$2
shift 2

# Check GPU availability
if command -v amd-smi >/dev/null && amd-smi version 2>&1 >/dev/null; then
    NUM_GPUS=$(amd-smi list | grep GPU | wc -l)
elif command -v rocm-smi >/dev/null && rocm-smi --version 2>&1 >/dev/null; then
    NUM_GPUS=$(rocm-smi --showserial | grep GPU | wc -l)
else
    NUM_GPUS=0
fi

# Default to 8 GPUs if detection fails (optimistic)
NUM_GPUS=$((NUM_GPUS > 0 ? NUM_GPUS : 8))

echo "Unit Test: $TEST_NAME (Requires: $REQUIRED_RANKS GPUs, Available: $NUM_GPUS)"

# Skip if not enough GPUs (unless using hostfile)
if [[ -z "$HOSTFILE" ]] && [[ $NUM_GPUS -lt $REQUIRED_RANKS ]]; then
    echo "Skip: $TEST_NAME ($REQUIRED_RANKS GPUs required but only $NUM_GPUS available)"
    exit $SKIP_CODE
fi

# Execute the actual test command
"$@"
TEST_EXIT_CODE=$?

exit $TEST_EXIT_CODE
