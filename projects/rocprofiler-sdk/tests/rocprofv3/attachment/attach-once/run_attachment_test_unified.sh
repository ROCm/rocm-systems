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

wait_for_attach_ready() {
    local pid=$1
    local max_wait=30
    local elapsed=0
    echo "Waiting for rocp-bg-attach thread in PID ${pid}..."
    while [ $elapsed -lt $max_wait ]; do
        if grep -ql "rocp-bg-attach" /proc/${pid}/task/*/comm 2>/dev/null; then
            echo "Attachment ready (${elapsed}s elapsed)"
            return 0
        fi
        sleep 1
        elapsed=$((elapsed + 1))
    done
    echo "Timed out after ${max_wait}s waiting for rocp-bg-attach thread"
    return 1
}

# Arguments
TEST_APP=$1
ROCPROFV3=$2
OUTPUT_DIR=${3:-${PWD}}
LOG_LEVEL=${4:-info}
OUTPUT_FILENAME=${5:-out}

# Set environment variables required for attachment
export ROCP_TOOL_ATTACH=1

OUTPUT_SUBDIR="attachment-output"
OUTPUT_FORMAT="json rocpd"

# Clean up any existing output
rm -rf ${OUTPUT_DIR}/${OUTPUT_SUBDIR}
mkdir -p ${OUTPUT_DIR}/${OUTPUT_SUBDIR}

# Check for permissions. We need to be able to ptrace any process in the system. (ptrace_scope == 0)
# First, if the ptrace_scope variable is not present, we assume there is no restriction and we can proceed normally.
# Next, if ptrace_scope would disallow this test, also confirm we are not root (which would allow it anyways.) (id -u != 0)
# Finally, confirm this process or python3 doesn't have CAP_SYS_PTRACE, which would allow the test also.
if [ -e /proc/sys/kernel/yama/ptrace_scope ]                             \
&& [ $(cat /proc/sys/kernel/yama/ptrace_scope) -ne 0 ]                   \
&& [ $(id -u) -ne 0 ]                                                    \
&& [[ $(getpcaps self) != *"cap_sys_ptrace"* ]]                          \
&& [[ $(getcap $(readlink -f $(which python3))) != *"cap_sys_ptrace"* ]]
    then
    echo "ptrace_scope is not 0, user is not root, and CAP_SYS_PTRACE is not present, so test cannot be completed. This test is skipped."
    touch ${OUTPUT_DIR}/${OUTPUT_SUBDIR}/skipped
    exit 0
fi

# Start the test application in the background
echo "Launching test application: ${TEST_APP}"
LD_PRELOAD=${ROCPROF_PRELOAD} ${TEST_APP} &
APP_PID=$!

# Wait for the application to be ready for attachment
wait_for_attach_ready $APP_PID

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

# Attachment
echo "Attaching profiler to PID $APP_PID for 500 milliseconds..."

# Run rocprofv3 with --attach option.
# No -o flag: the process uses the default %hostname%/%pid% naming.
LD_PRELOAD=${ROCPROF_PRELOAD} ${ROCPROFV3} --attach $APP_PID --attach-duration-msec 500 -s -f ${OUTPUT_FORMAT} --stats --summary --group-by-queue --attach-sync-output -d ${OUTPUT_DIR}/${OUTPUT_SUBDIR} --log-level ${LOG_LEVEL} &
ROCPROF_PID=$!
echo "rocprofv3 PID: $ROCPROF_PID"

# Wait for the attach process to complete
wait $ROCPROF_PID
ROCPROF_EXIT_CODE=$?

if [ $ROCPROF_EXIT_CODE -ne 0 ]; then
    echo "rocprofv3_attach test failed with exit code $ROCPROF_EXIT_CODE"
    kill $APP_PID 2>/dev/null
    exit 1
fi

echo "Profiler detached successfully"

# End the running application
echo "Sending SIGINT to application..."
kill -2 $APP_PID 2>/dev/null
wait $APP_PID
APP_EXIT_CODE=$?

if [ $APP_EXIT_CODE -ne 0 ]; then
    echo "Test application failed with exit code $APP_EXIT_CODE"
    exit 1
fi

echo "Test application completed successfully"

echo "Checking for generated output files..."
ls -laR ${OUTPUT_DIR}/${OUTPUT_SUBDIR}/

# Check if expected output files were created
JSON_COUNT=$(find ${OUTPUT_DIR}/${OUTPUT_SUBDIR}/ -name "*.json" | wc -l)
if [ $JSON_COUNT -eq 0 ]; then
    echo "Error: No JSON files were generated"
    exit 1
else
    echo "Found $JSON_COUNT JSON file(s)"
fi

# Locate the process's output files. With default naming the files are under a
# subdirectory named after the hostname and contain the PID in the filename,
# e.g. attachment-output/<hostname>/<pid>_results.json
APP_JSON=$(find ${OUTPUT_DIR}/${OUTPUT_SUBDIR}/ -name "${APP_PID}_results.json" | head -1)
if [ -z "$APP_JSON" ]; then
    echo "Error: Could not find app (PID ${APP_PID}) JSON output in ${OUTPUT_DIR}/${OUTPUT_SUBDIR}/"
    exit 1
fi
echo "Found app JSON output: $APP_JSON"

APP_OUTPUT_DIR=$(dirname "$APP_JSON")

# Rename output files to well-known names so CMakeLists.txt can reference them
# without knowing the hostname or PID at configure time.
for src in "${APP_OUTPUT_DIR}/${APP_PID}"_*.json "${APP_OUTPUT_DIR}/${APP_PID}"_*.db; do
    [ -f "$src" ] || continue
    dst_name=$(basename "$src" | sed "s/^${APP_PID}_/${OUTPUT_FILENAME}_/")
    cp "$src" "${OUTPUT_DIR}/${OUTPUT_SUBDIR}/${dst_name}"
    echo "Copied $(basename $src) -> ${dst_name}"
done

# Verify the well-known files exist
for expected_file in "${OUTPUT_FILENAME}_results.json" "${OUTPUT_FILENAME}_results.db"; do
    if [ ! -f "${OUTPUT_DIR}/${OUTPUT_SUBDIR}/${expected_file}" ]; then
        echo "Error: Expected output file ${OUTPUT_DIR}/${OUTPUT_SUBDIR}/${expected_file} not found"
        exit 1
    else
        echo "Found ${expected_file}"
    fi
done

echo "Attachment test completed successfully"
exit 0
