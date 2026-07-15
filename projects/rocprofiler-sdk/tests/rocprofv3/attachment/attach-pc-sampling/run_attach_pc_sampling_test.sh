#!/bin/bash

# MIT License
#
# Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

set -e

TEST_APP=$1
ROCPROFV3=$2
OUTPUT_DIR=${3:-${PWD}}
LOG_LEVEL=${4:-warning}
OUTPUT_FILENAME=${5:-out}
PC_SAMPLING_METHOD=${6:-host_trap}
PC_SAMPLING_UNIT=${7:-time}
PC_SAMPLING_INTERVAL=${8:-1000}
ATTACH_DURATION_MSEC=${9:-5000}
# Iteration count for exec-mask-manipulation: keeps the app alive long enough
# for the profiler to attach, sample, and detach within ATTACH_DURATION_MSEC.
TEST_APP_ITER_NUM=${10:-1048576}

# The target process must load rocprofiler-register to expose the attach thread.
export ROCP_TOOL_ATTACH=1

OUTPUT_SUBDIR="attachment-pc-sampling-output"
EXPECTED_FILES=("${OUTPUT_FILENAME}_pc_sampling_${PC_SAMPLING_METHOD}.csv" "${OUTPUT_FILENAME}_agent_info.csv")

rm -rf ${OUTPUT_DIR}/${OUTPUT_SUBDIR}
mkdir -p ${OUTPUT_DIR}/${OUTPUT_SUBDIR}

# Skip when ptrace is not permitted and we lack the privileges to override it.
if [ -e /proc/sys/kernel/yama/ptrace_scope ]                             \
&& [ $(cat /proc/sys/kernel/yama/ptrace_scope) -ne 0 ]                   \
&& [ $(id -u) -ne 0 ]                                                    \
&& [[ $(getpcaps self 2>/dev/null) != *"cap_sys_ptrace"* ]]
    then
    echo "ptrace not permitted (ptrace_scope != 0, not root, no CAP_SYS_PTRACE). This test is skipped."
    touch ${OUTPUT_DIR}/${OUTPUT_SUBDIR}/skipped
    exit 0
fi

wait_for_attach_ready() {
    local pid=$1 max_wait=30 elapsed=0
    while [ $elapsed -lt $max_wait ]; do
        if grep -ql "rocp-bg-attach" /proc/${pid}/task/*/comm 2>/dev/null; then
            echo "attach thread ready after ${elapsed}s"
            return 0
        fi
        kill -0 $pid 2>/dev/null || { echo "target exited before becoming attach-ready"; return 1; }
        sleep 1
        elapsed=$((elapsed + 1))
    done
    echo "timed out waiting for rocp-bg-attach thread"
    return 1
}

echo "Launching ${TEST_APP} ${TEST_APP_ITER_NUM} in the background..."
LD_PRELOAD=${ROCPROF_PRELOAD} ${TEST_APP} ${TEST_APP_ITER_NUM} &
APP_PID=$!

if ! wait_for_attach_ready $APP_PID; then
    kill $APP_PID 2>/dev/null || true
    exit 1
fi

echo "Attaching ${PC_SAMPLING_METHOD} PC sampling to PID ${APP_PID} for ${ATTACH_DURATION_MSEC}ms..."
LD_PRELOAD=${ROCPROF_PRELOAD} ${ROCPROFV3} --attach ${APP_PID} \
    --attach-duration-msec ${ATTACH_DURATION_MSEC} --attach-sync-output \
    --pc-sampling-beta-enabled --pc-sampling-unit ${PC_SAMPLING_UNIT} \
    --pc-sampling-method ${PC_SAMPLING_METHOD} --pc-sampling-interval ${PC_SAMPLING_INTERVAL} \
    -f csv json -d ${OUTPUT_DIR}/${OUTPUT_SUBDIR} -o ${OUTPUT_FILENAME} --log-level ${LOG_LEVEL}

# --attach-sync-output makes detach block until the output file is fully written,
# so the app can be stopped safely here. Removing that flag risks a truncated file.
echo "Profiler detached. Stopping the application..."
kill -2 ${APP_PID} 2>/dev/null || true
wait ${APP_PID} 2>/dev/null || true

for f in "${EXPECTED_FILES[@]}"; do
    if [ ! -f "${OUTPUT_DIR}/${OUTPUT_SUBDIR}/${f}" ]; then
        echo "Error: expected output file ${f} not found"
        exit 1
    fi
done

echo "PC sampling attachment test (method=${PC_SAMPLING_METHOD}) completed successfully"
exit 0
