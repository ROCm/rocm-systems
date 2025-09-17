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

OUTPUT_SUBDIR="attachment-output"
# For CSV, we don't require specific files since different traces may or may not be generated
# We'll just check if at least one CSV file was created
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

echo "Attaching profiler to PID $APP_PID for 5 seconds (${OUTPUT_FORMAT} format)..."

# Output the command and environment for debugging
echo "===== COMMAND TO EXECUTE ====="
echo "${ROCPROFV3} --attach $APP_PID --attach-duration-msec 5000 -s -f ${OUTPUT_FORMAT} --stats --summary --group-by-queue -d ${OUTPUT_DIR}/${OUTPUT_SUBDIR} -o ${OUTPUT_FILENAME:-out}"
echo ""
echo "===== ENVIRONMENT VARIABLES ====="
env | sort
echo "===== END ENVIRONMENT ====="
echo ""

# Run rocprofv3 with --attach option
LD_PRELOAD=${ROCPROF_PRELOAD} ${ROCPROFV3} --attach $APP_PID --attach-duration-msec 5000 -s -f ${OUTPUT_FORMAT} --stats --summary --group-by-queue -d ${OUTPUT_DIR}/${OUTPUT_SUBDIR} -o ${OUTPUT_FILENAME:-out}

echo "${OUTPUT_FORMAT} profiler detached successfully"

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

echo "Attachment ${OUTPUT_FORMAT} test completed successfully"
exit 0
