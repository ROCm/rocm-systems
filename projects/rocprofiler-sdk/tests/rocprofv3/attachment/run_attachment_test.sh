#!/bin/bash

# MIT License
#
# Copyright (c) 2025 Advanced Micro Devices, Inc. All rights reserved.
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
# THE SOFTWARE.

set -e

# Arguments
TEST_APP=$1
ROCPROFV3=$2
OUTPUT_DIR=$3
LOG_LEVEL=$4

# Set environment variables required for attachment
export ROCPROFILER_REGISTER_ATTACHMENT_QUEUES_ENABLED=1

# Clean up any existing output
rm -rf ${OUTPUT_DIR}/attachment-output
mkdir -p ${OUTPUT_DIR}/attachment-output

echo "Starting attachment test..."

# Start the test application in the background
echo "Launching test application: ${TEST_APP}"
${TEST_APP} &
APP_PID=$!

# Wait a moment for the application to start
sleep 1

# Check if the application is still running
if ! kill -0 $APP_PID 2>/dev/null; then
    echo "Test application failed to start or exited early"
    exit 1
fi

echo "Test application started with PID: $APP_PID"

if [ ! -f "${ROCPROFV3}" ]; then
    echo "Error: rocprofv3 not found at ${ROCPROFV3}"
    kill $APP_PID 2>/dev/null
    exit 1
fi

echo "Attaching profiler to PID $APP_PID for 5 seconds..."

# Output the command and environment for debugging
echo "===== COMMAND TO EXECUTE ====="
echo "${ROCPROFV3} --attach $APP_PID --attach-duration-msec 5000 --hsa-core-trace --hip-trace --kernel-trace --memory-copy-trace -f csv -d ${OUTPUT_DIR}/attachment-output -o out"
echo ""
echo "===== ENVIRONMENT VARIABLES ====="
env | sort
echo "===== END ENVIRONMENT ====="
echo ""

# Run rocprofv3 with --attach option
${ROCPROFV3} --attach $APP_PID --attach-duration-msec 5000 --hsa-core-trace --hip-trace --kernel-trace --memory-copy-trace -f csv -d ${OUTPUT_DIR}/attachment-output -o out &
ATTACH_PID=$!

# Wait for the attach process to complete
wait $ATTACH_PID
ATTACH_EXIT_CODE=$?

if [ $ATTACH_EXIT_CODE -ne 0 ]; then
    echo "rocprofv3_attach failed with exit code $ATTACH_EXIT_CODE"
    kill $APP_PID 2>/dev/null
    exit 1
fi

echo "Profiler detached successfully"

# Wait for the application to finish
echo "Waiting for application to complete..."
wait $APP_PID
APP_EXIT_CODE=$?

if [ $APP_EXIT_CODE -ne 0 ]; then
    echo "Test application failed with exit code $APP_EXIT_CODE"
    exit 1
fi

echo "Test application completed successfully"

# Files should be created directly in the expected location with the specified output name
echo "Checking for generated output files..."
ls -la ${OUTPUT_DIR}/attachment-output/

# Check if output files were created
if [ ! -f "${OUTPUT_DIR}/attachment-output/out_kernel_trace.csv" ]; then
    echo "Error: Expected output file out_kernel_trace.csv not found"
    exit 1
fi

echo "Attachment test completed successfully"
exit 0
