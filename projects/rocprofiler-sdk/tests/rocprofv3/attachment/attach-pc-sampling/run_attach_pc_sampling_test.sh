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
OUTPUT_DIR=$3
PC_SAMPLING_METHOD=$4
PC_SAMPLING_UNIT=$5
PC_SAMPLING_INTERVAL=$6
PC_SAMPLING_SKIP_REGEX=${7:-}
ATTACH_DURATION_MSEC=5000

export ROCP_TOOL_ATTACH=1

rm -rf "${OUTPUT_DIR}"
mkdir -p "${OUTPUT_DIR}"

if [ -e /proc/sys/kernel/yama/ptrace_scope ]                             \
&& [ "$(cat /proc/sys/kernel/yama/ptrace_scope)" -ne 0 ]                 \
&& [ "$(id -u)" -ne 0 ]                                                  \
&& [[ $(getpcaps self 2>/dev/null) != *"cap_sys_ptrace"* ]]              \
&& [[ $(getcap "$(readlink -f "$(which python3)")") != *"cap_sys_ptrace"* ]]
then
    echo "This test is skipped. ptrace is not permitted."
    touch "${OUTPUT_DIR}/skipped"
    exit 0
fi

APP_PID=""
APP_LOG="${OUTPUT_DIR}/target.log"
cleanup() {
    if [ -n "${APP_PID}" ] && kill -0 "${APP_PID}" 2>/dev/null; then
        kill -2 "${APP_PID}" 2>/dev/null || true
        wait "${APP_PID}" 2>/dev/null || true
    fi
    if [ -f "${APP_LOG}" ]; then
        cat "${APP_LOG}"
    fi
}
trap cleanup EXIT

wait_for_attach_ready() {
    local pid=$1
    local elapsed=0
    local max_wait=30

    while [ "${elapsed}" -lt "${max_wait}" ]; do
        if grep -ql "rocp-bg-attach" /proc/"${pid}"/task/*/comm 2>/dev/null; then
            return 0
        fi
        if ! kill -0 "${pid}" 2>/dev/null; then
            echo "Target exited before its attach thread became ready"
            return 1
        fi
        sleep 1
        elapsed=$((elapsed + 1))
    done

    echo "Timed out waiting for the target attach thread"
    return 1
}

ROCPROFILER_ATTACH_TEST_DELAY_MSEC=1 \
    LD_PRELOAD="${ROCPROF_PRELOAD:-}" "${TEST_APP}" >"${APP_LOG}" 2>&1 &
APP_PID=$!
wait_for_attach_ready "${APP_PID}"

ROCPROFV3_LOG="${OUTPUT_DIR}/rocprofv3.log"
set +e
LD_PRELOAD=${ROCPROF_PRELOAD} "${ROCPROFV3}" --attach "${APP_PID}" \
    --attach-duration-msec "${ATTACH_DURATION_MSEC}" --attach-sync-output --kernel-trace \
    --pc-sampling-beta-enabled --pc-sampling-method "${PC_SAMPLING_METHOD}" \
    --pc-sampling-unit "${PC_SAMPLING_UNIT}" --pc-sampling-interval "${PC_SAMPLING_INTERVAL}" \
    --output-format json -d "${OUTPUT_DIR}" -o out >"${ROCPROFV3_LOG}" 2>&1
ROCPROFV3_STATUS=$?
set -e
cat "${ROCPROFV3_LOG}"

if [ -n "${PC_SAMPLING_SKIP_REGEX}" ] \
&& grep -Eq -- "${PC_SAMPLING_SKIP_REGEX}" "${ROCPROFV3_LOG}" "${APP_LOG}"
then
    echo "This test is skipped. PC sampling is unavailable."
    touch "${OUTPUT_DIR}/skipped"
    exit 0
fi

if [ "${ROCPROFV3_STATUS}" -ne 0 ]; then
    exit "${ROCPROFV3_STATUS}"
fi

if ! kill -0 "${APP_PID}" 2>/dev/null; then
    echo "Target did not survive PC-sampling attachment"
    exit 1
fi

kill -2 "${APP_PID}"
wait "${APP_PID}"
APP_PID=""
cat "${APP_LOG}"
trap - EXIT

if [ ! -s "${OUTPUT_DIR}/out_results.json" ]; then
    echo "PC-sampling attachment did not produce a nonempty JSON result"
    exit 1
fi
