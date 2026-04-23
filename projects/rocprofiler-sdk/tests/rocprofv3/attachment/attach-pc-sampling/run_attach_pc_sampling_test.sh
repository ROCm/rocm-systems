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
OUTPUT_DIR=${3:-${PWD}}
LOG_LEVEL=${4:-info}
OUTPUT_FILENAME=${5:-out}
PC_SAMPLING_METHOD=${6:-host_trap}
PC_SAMPLING_UNIT=${7:-time}
PC_SAMPLING_INTERVAL=${8:-1}
ATTACH_DURATION_MSEC=${9:-5000}
# Iteration count for exec-mask-manipulation: keeps the app alive long enough
# for the profiler to attach, sample, and detach within ATTACH_DURATION_MSEC.
TEST_APP_ITER_NUM=${10:-1048576}

export ROCP_TOOL_ATTACH=1

OUTPUT_SUBDIR="attachment-pc-sampling-output"
EXPECTED_CSV_FILES=("${OUTPUT_FILENAME}_pc_sampling_${PC_SAMPLING_METHOD}.csv" "${OUTPUT_FILENAME}_agent_info.csv")

rm -rf ${OUTPUT_DIR}/${OUTPUT_SUBDIR}
mkdir -p ${OUTPUT_DIR}/${OUTPUT_SUBDIR}

# ptrace permission check: skip if ptrace_scope disallows and we lack privileges
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

echo "Starting PC sampling attachment test (method=${PC_SAMPLING_METHOD})..."

echo "Launching test application: ${TEST_APP} ${TEST_APP_ITER_NUM}"
LD_PRELOAD=${ROCPROF_PRELOAD} ${TEST_APP} ${TEST_APP_ITER_NUM} &
APP_PID=$!

sleep 1

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

echo "Attaching profiler to PID $APP_PID for ${ATTACH_DURATION_MSEC}ms with PC sampling (${PC_SAMPLING_METHOD})..."

echo "===== COMMAND TO EXECUTE ====="
echo "${ROCPROFV3} --attach $APP_PID --attach-duration-msec ${ATTACH_DURATION_MSEC} --pc-sampling-beta-enabled --pc-sampling-unit ${PC_SAMPLING_UNIT} --pc-sampling-method ${PC_SAMPLING_METHOD} --pc-sampling-interval ${PC_SAMPLING_INTERVAL} -f csv -d ${OUTPUT_DIR}/${OUTPUT_SUBDIR} --log-level ${LOG_LEVEL} -o ${OUTPUT_FILENAME}"
echo ""

LD_PRELOAD=${ROCPROF_PRELOAD} ${ROCPROFV3} --attach $APP_PID --attach-duration-msec ${ATTACH_DURATION_MSEC} --pc-sampling-beta-enabled --pc-sampling-unit ${PC_SAMPLING_UNIT} --pc-sampling-method ${PC_SAMPLING_METHOD} --pc-sampling-interval ${PC_SAMPLING_INTERVAL} -f csv -d ${OUTPUT_DIR}/${OUTPUT_SUBDIR} --log-level ${LOG_LEVEL} -o ${OUTPUT_FILENAME}

echo "Profiler detached successfully"

echo "Waiting for application to complete..."
wait $APP_PID
APP_EXIT_CODE=$?

if [ $APP_EXIT_CODE -ne 0 ]; then
    echo "Test application failed with exit code $APP_EXIT_CODE"
    exit 1
fi

echo "Test application completed successfully"

echo "Checking for generated output files..."
ls -la ${OUTPUT_DIR}/${OUTPUT_SUBDIR}/

for expected_file in "${EXPECTED_CSV_FILES[@]}"; do
    if [ ! -f "${OUTPUT_DIR}/${OUTPUT_SUBDIR}/${expected_file}" ]; then
        echo "Error: Expected output file ${OUTPUT_DIR}/${OUTPUT_SUBDIR}/${expected_file} not found"
        exit 1
    fi
done

echo "PC sampling attachment test (method=${PC_SAMPLING_METHOD}) completed successfully"
exit 0
