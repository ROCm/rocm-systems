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
OUTPUT_FILENAME=${5:-out}

# Set environment variables required for attachment
export ROCP_TOOL_ATTACH=1

# Set output directory based on format
OUTPUT_SUBDIR="attachment-output"
EXPECTED_FILES=("${OUTPUT_FILENAME}_results.json" "${OUTPUT_FILENAME}_results.db")
OUTPUT_FORMAT="csv json rocpd"

# Clean up any existing output
rm -rf ${OUTPUT_DIR}/${OUTPUT_SUBDIR}
mkdir -p ${OUTPUT_DIR}/${OUTPUT_SUBDIR}

echo "Starting attachment test (${OUTPUT_FORMAT} format)..."

# Start the test application in the background
echo "Launching test application: ${TEST_APP}"
LD_PRELOAD=${ROCPROF_PRELOAD} ${TEST_APP} &
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
echo "First attachment: Attaching profiler to PID $APP_PID for 5 seconds (${OUTPUT_FORMAT} format)..."


# Run first rocprofv3 with --attach option
echo "About to launch first rocprofv3 process..."
LD_PRELOAD=${ROCPROF_PRELOAD} ${ROCPROFV3} --attach $APP_PID --attach-duration-msec 5000 -s -f ${OUTPUT_FORMAT} -d ${OUTPUT_DIR}/${OUTPUT_SUBDIR} -o ${OUTPUT_FILENAME:-out} &
FIRST_ROCPROF_PID=$!
ATTACH_PID=$FIRST_ROCPROF_PID
echo "First rocprofv3 PID: $FIRST_ROCPROF_PID"

# Wait for the first attach process to complete
wait $ATTACH_PID
ATTACH_EXIT_CODE=$?

if [ $ATTACH_EXIT_CODE -ne 0 ]; then
    echo "First rocprofv3_attach ${OUTPUT_FORMAT} test failed with exit code $ATTACH_EXIT_CODE"
    kill $APP_PID 2>/dev/null
    exit 1
fi

echo "First ${OUTPUT_FORMAT} profiler detached successfully"

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
rm -rf ${OUTPUT_DIR}/${OUTPUT_SUBDIR}/*

# Check if the application is still running
if ! kill -0 $APP_PID 2>/dev/null; then
    echo "Test application exited before second attachment"
    exit 1
fi

# Second attachment
echo "Second attachment: Attaching profiler to PID $APP_PID for 5 seconds (${OUTPUT_FORMAT} format)..."


# Run second rocprofv3 with --attach option
echo "About to launch second rocprofv3 process..."
LD_PRELOAD=${ROCPROF_PRELOAD} ${ROCPROFV3} --attach $APP_PID --attach-duration-msec 5000 -s -f ${OUTPUT_FORMAT} -d ${OUTPUT_DIR}/${OUTPUT_SUBDIR} -o ${OUTPUT_FILENAME:-out} &
SECOND_ROCPROF_PID=$!
ATTACH_PID=$SECOND_ROCPROF_PID
echo "Second rocprofv3 PID: $SECOND_ROCPROF_PID"

# Wait for the second attach process to complete
wait $ATTACH_PID
ATTACH_EXIT_CODE=$?

if [ $ATTACH_EXIT_CODE -ne 0 ]; then
    echo "Second rocprofv3_attach ${OUTPUT_FORMAT} test failed with exit code $ATTACH_EXIT_CODE"
    kill $APP_PID 2>/dev/null
    exit 1
fi

echo "Second ${OUTPUT_FORMAT} profiler detached successfully"

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
echo "Checking for generated ${OUTPUT_FORMAT} output files..."
ls -la ${OUTPUT_DIR}/${OUTPUT_SUBDIR}/

# Check if expected output files were created
# For CSV format, check if at least one CSV file was generated
CSV_COUNT=$(find ${OUTPUT_DIR}/${OUTPUT_SUBDIR}/ -name "*.csv" | wc -l)
if [ $CSV_COUNT -eq 0 ]; then
    echo "Error: No CSV files were generated"
    exit 1
else
    echo "Found $CSV_COUNT CSV file(s)"
fi

# For other formats, check specific expected files
for expected_file in "${EXPECTED_FILES[@]}"; do
    if [ ! -f "${OUTPUT_DIR}/${OUTPUT_SUBDIR}/${expected_file}" ]; then
        echo "Error: Expected output file ${OUTPUT_DIR}/${OUTPUT_SUBDIR}/${expected_file} not found"
        exit 1
    fi
done

echo "Reattachment ${OUTPUT_FORMAT} test completed successfully"
exit 0
