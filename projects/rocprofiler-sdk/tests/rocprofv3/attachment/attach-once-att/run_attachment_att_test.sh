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

OUTPUT_SUBDIR="attachment-att-output"

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

# Attachment with ATT enabled.
# Target CU defaults to 1 (rocprofv3 default); shader engine mask 0x1 selects SE0.
# Small buffer (16MB) to keep the test fast.
# --attach-sync-output ensures ATT output is fully written before rocprofv3 exits.
# Note: ATT mode does not produce standard kernel_dispatch/memory_copy/hsa_api
# buffer records; it produces raw instruction trace data (*.att files) decoded
# into ui_output_agent_* directories.
echo "Attaching profiler with ATT to PID $APP_PID for 500 milliseconds..."

LD_PRELOAD=${ROCPROF_PRELOAD} ${ROCPROFV3} \
    --attach $APP_PID \
    --attach-duration-msec 500 \
    --att \
    --att-shader-engine-mask 0x1 \
    --att-buffer-size 0x1000000 \
    -f json rocpd \
    --attach-sync-output \
    -d ${OUTPUT_DIR}/${OUTPUT_SUBDIR} \
    --log-level ${LOG_LEVEL} &
ROCPROF_PID=$!
echo "rocprofv3 PID: $ROCPROF_PID"

# Wait for the attach process to complete
wait $ROCPROF_PID
ROCPROF_EXIT_CODE=$?

if [ $ROCPROF_EXIT_CODE -ne 0 ]; then
    echo "rocprofv3 attach test failed with exit code $ROCPROF_EXIT_CODE"
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

# In ATT attach mode the output structure is:
#
#   attachment-att-output/
#     <hostname>/<pid>_results.json                   <- profiling results JSON
#     <hostname>/<pid>_results.db                     <- rocpd database
#     <hostname>/<pid>_*.att                          <- raw ATT data
#     <hostname>/<pid>_*.out                          <- code objects
#     ui_output_agent_<app_pid>_dispatch_*/           <- decoded ATT output (top-level)
#     stats_ui_output_agent_*.csv                     <- decoded ATT stats (top-level)
#
# The hostname subdir name is not known at script-write time, so we search
# recursively (same pattern as the standard attach-once script).

APP_JSON=$(find ${OUTPUT_DIR}/${OUTPUT_SUBDIR}/ -name "${APP_PID}_results.json" \
    ! -path "*/ui_output_agent_*" | head -1)
if [ -z "$APP_JSON" ]; then
    echo "Error: Could not find app (PID ${APP_PID}) JSON output in ${OUTPUT_DIR}/${OUTPUT_SUBDIR}/"
    exit 1
fi
echo "Found results JSON: $APP_JSON"

APP_DB=$(find ${OUTPUT_DIR}/${OUTPUT_SUBDIR}/ -name "${APP_PID}_results.db" \
    ! -path "*/ui_output_agent_*" 2>/dev/null | head -1)

# Copy results to well-known names at the top of the output subdir so the
# validate step can reference them without knowing the PID.
cp "$APP_JSON" "${OUTPUT_DIR}/${OUTPUT_SUBDIR}/${OUTPUT_FILENAME}_results.json"
echo "Copied $(basename $APP_JSON) -> ${OUTPUT_FILENAME}_results.json"

if [ -n "$APP_DB" ]; then
    cp "$APP_DB" "${OUTPUT_DIR}/${OUTPUT_SUBDIR}/${OUTPUT_FILENAME}_results.db"
    echo "Copied $(basename $APP_DB) -> ${OUTPUT_FILENAME}_results.db"
fi

# Verify the well-known results JSON exists
if [ ! -f "${OUTPUT_DIR}/${OUTPUT_SUBDIR}/${OUTPUT_FILENAME}_results.json" ]; then
    echo "Error: Expected output file ${OUTPUT_DIR}/${OUTPUT_SUBDIR}/${OUTPUT_FILENAME}_results.json not found"
    exit 1
fi

# Verify ATT output directories exist at the top level
UI_DIR_COUNT=$(find ${OUTPUT_DIR}/${OUTPUT_SUBDIR}/ -maxdepth 1 -name "ui_output_agent_*" -type d | wc -l)
if [ $UI_DIR_COUNT -eq 0 ]; then
    echo "Error: No ui_output_agent_* directories found in ${OUTPUT_DIR}/${OUTPUT_SUBDIR}/"
    exit 1
fi
echo "Found ${UI_DIR_COUNT} ATT ui_output_agent_* director(ies)"

echo "Attachment ATT test completed successfully"
exit 0
