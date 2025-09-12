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
rm -rf ${OUTPUT_DIR}/reattachment-output
mkdir -p ${OUTPUT_DIR}/reattachment-output

echo "Starting reattachment test..."

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

# First attachment
echo "First attachment: Attaching profiler to PID $APP_PID for 5 seconds..."


# Run first rocprofv3 with --attach option
echo "About to launch first rocprofv3 process..."
${ROCPROFV3} --attach $APP_PID --attach-duration-msec 5000 --hsa-core-trace --hip-trace --kernel-trace --memory-copy-trace -f csv -d ${OUTPUT_DIR}/reattachment-output -o out &
FIRST_ROCPROF_PID=$!
ATTACH_PID=$FIRST_ROCPROF_PID
echo "First rocprofv3 PID: $FIRST_ROCPROF_PID"

# Wait for the first attach process to complete
wait $ATTACH_PID
ATTACH_EXIT_CODE=$?

if [ $ATTACH_EXIT_CODE -ne 0 ]; then
    echo "First rocprofv3_attach failed with exit code $ATTACH_EXIT_CODE"
    kill $APP_PID 2>/dev/null
    exit 1
fi

echo "First profiler detached successfully"

# Check temp files created by first run
echo "=== TEMP FILES AFTER FIRST RUN ==="
echo "Looking for temp files with target PID pattern ($PPID-$APP_PID):"
ls -la ${OUTPUT_DIR}/.rocprofv3/*$PPID-$APP_PID* 2>/dev/null || echo "No files with target PID pattern"
echo "Looking for temp files with first tool PID pattern ($PPID-$FIRST_ROCPROF_PID):"
ls -la ${OUTPUT_DIR}/.rocprofv3/*$PPID-$FIRST_ROCPROF_PID* 2>/dev/null || echo "No files with first tool PID pattern"
echo "All temp files:"
ls -la ${OUTPUT_DIR}/.rocprofv3/ 2>/dev/null || echo "No temp files directory"
echo "MD5 checksums of temp files:"
if [ -d "${OUTPUT_DIR}/.rocprofv3" ] && [ "$(ls -A ${OUTPUT_DIR}/.rocprofv3 2>/dev/null)" ]; then
    md5sum ${OUTPUT_DIR}/.rocprofv3/* 2>/dev/null || echo "No temp files to checksum"
else
    echo "No temp files to checksum"
fi

# Clear output files between attachments
echo "Clearing output files before second attachment..."
rm -rf ${OUTPUT_DIR}/reattachment-output/*

# Check if the application is still running
if ! kill -0 $APP_PID 2>/dev/null; then
    echo "Test application exited before second attachment"
    exit 1
fi

# Second attachment
echo "Second attachment: Attaching profiler to PID $APP_PID for 5 seconds..."


# Run second rocprofv3 with --attach option
echo "About to launch second rocprofv3 process..."
${ROCPROFV3} --attach $APP_PID --attach-duration-msec 5000 --hsa-core-trace --hip-trace --kernel-trace --memory-copy-trace -f csv -d ${OUTPUT_DIR}/reattachment-output -o out &
SECOND_ROCPROF_PID=$!
ATTACH_PID=$SECOND_ROCPROF_PID
echo "Second rocprofv3 PID: $SECOND_ROCPROF_PID"

# Wait for the second attach process to complete
wait $ATTACH_PID
ATTACH_EXIT_CODE=$?

if [ $ATTACH_EXIT_CODE -ne 0 ]; then
    echo "Second rocprofv3_attach failed with exit code $ATTACH_EXIT_CODE"
    kill $APP_PID 2>/dev/null
    exit 1
fi

echo "Second profiler detached successfully"

# Check temp files created by second run
echo "=== TEMP FILES AFTER SECOND RUN ==="
echo "Looking for temp files with target PID pattern ($PPID-$APP_PID):"
ls -la ${OUTPUT_DIR}/.rocprofv3/*$PPID-$APP_PID* 2>/dev/null || echo "No files with target PID pattern"
echo "Looking for temp files with second tool PID pattern ($PPID-$SECOND_ROCPROF_PID):"
ls -la ${OUTPUT_DIR}/.rocprofv3/*$PPID-$SECOND_ROCPROF_PID* 2>/dev/null || echo "No files with second tool PID pattern"
echo "All temp files:"
ls -la ${OUTPUT_DIR}/.rocprofv3/ 2>/dev/null || echo "No temp files directory"
echo "MD5 checksums of temp files:"
if [ -d "${OUTPUT_DIR}/.rocprofv3" ] && [ "$(ls -A ${OUTPUT_DIR}/.rocprofv3 2>/dev/null)" ]; then
    md5sum ${OUTPUT_DIR}/.rocprofv3/* 2>/dev/null || echo "No temp files to checksum"
else
    echo "No temp files to checksum"
fi

echo "=== PID COMPARISON SUMMARY ==="
echo "Target process PID: $APP_PID (constant)"
echo "Script PID: $$ (constant)"
echo "Script PPID: $PPID (constant)"
echo "First rocprofv3 PID: $FIRST_ROCPROF_PID"
echo "Second rocprofv3 PID: $SECOND_ROCPROF_PID"
echo "Expected mismatch: detach looks for $PPID-$APP_PID-* but finds $PPID-$SECOND_ROCPROF_PID-*"

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
ls -la ${OUTPUT_DIR}/reattachment-output/

# Check if output files were created
if [ ! -f "${OUTPUT_DIR}/reattachment-output/out_kernel_trace.csv" ]; then
    echo "Error: Expected output file out_kernel_trace.csv not found"
    exit 1
fi

echo "Reattachment test completed successfully"
exit 0
