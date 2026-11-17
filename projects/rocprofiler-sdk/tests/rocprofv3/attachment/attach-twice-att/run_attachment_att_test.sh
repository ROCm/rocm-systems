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
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
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
    while [ "${elapsed}" -lt "${max_wait}" ]; do
        if grep -ql "rocp-bg-attach" /proc/"${pid}"/task/*/comm 2>/dev/null; then
            echo "Attachment ready (${elapsed}s elapsed)"
            return 0
        fi
        sleep 1
        elapsed=$((elapsed + 1))
    done

    echo "Error: timed out after ${max_wait}s waiting for rocp-bg-attach thread"
    return 1
}

require_cycle_output() {
    local cycle_dir=$1
    local cycle_name=$2
    local raw_att
    local results_json
    local ui_dir

    raw_att=$(find "${cycle_dir}" -type f -name "*.att" -size +0c -print -quit)
    if [ -z "${raw_att}" ]; then
        echo "Error: ${cycle_name} produced no nonempty raw ATT file"
        return 1
    fi

    results_json=$(find "${cycle_dir}" -type f -size +0c \
        \( -name "${APP_PID}_results.json" -o -name "${APP_PID}_*_results.json" \) \
        -print -quit)
    if [ -z "${results_json}" ]; then
        echo "Error: ${cycle_name} produced no results JSON for PID ${APP_PID}"
        return 1
    fi

    ui_dir=$(find "${cycle_dir}" -maxdepth 1 -type d -name "ui_output_agent_*" -print -quit)
    if [ -z "${ui_dir}" ] || [ ! -s "${ui_dir}/filenames.json" ]; then
        echo "Error: ${cycle_name} produced no decoded ATT manifest"
        return 1
    fi

    echo "${cycle_name} output:"
    echo "  raw ATT: ${raw_att}"
    echo "  results: ${results_json}"
    echo "  decoded: ${ui_dir}"
}

run_attach_cycle() {
    local cycle_name=$1
    local cycle_dir="${OUTPUT_ROOT}/${cycle_name}"

    mkdir -p "${cycle_dir}"
    echo "Attaching profiler for ${cycle_name}..."

    LD_PRELOAD="${ROCPROF_PRELOAD:-}" "${ROCPROFV3}" \
        --attach "${APP_PID}" \
        --attach-duration-msec 2000 \
        --att \
        "${ATT_LIB_ARGS[@]}" \
        --kernel-include-regex simple_kernel \
        --kernel-iteration-range "1-${EXPECTED_DISPATCHES}" \
        --att-target-cu 0 \
        --att-shader-engine-mask 0xFFFFFFFF \
        --att-buffer-size 0x1000000 \
        --output-format json \
        --attach-sync-output \
        --output-directory "${cycle_dir}" \
        --log-level "${LOG_LEVEL}"

    if ! kill -0 "${APP_PID}" 2>/dev/null; then
        echo "Error: test application exited during ${cycle_name}"
        return 1
    fi

    require_cycle_output "${cycle_dir}" "${cycle_name}"
}

TEST_APP=$1
ROCPROFV3=$2
OUTPUT_DIR=${3:-${PWD}}
LOG_LEVEL=${4:-info}
EXPECTED_DISPATCHES=${5:-3}
ATT_LIB_DIR=${6:-}

export ROCP_TOOL_ATTACH=1

OUTPUT_ROOT="${OUTPUT_DIR}/attachment-att-twice-output"
APP_PID=""
ATT_LIB_ARGS=()

if [ -n "${ATT_LIB_DIR}" ]; then
    ATT_LIB_ARGS=(--att-library-path "${ATT_LIB_DIR}")
fi

cleanup() {
    if [ -n "${APP_PID}" ] && kill -0 "${APP_PID}" 2>/dev/null; then
        kill -2 "${APP_PID}" 2>/dev/null || true
        wait "${APP_PID}" 2>/dev/null || true
    fi
}
trap cleanup EXIT

rm -rf "${OUTPUT_ROOT}"
mkdir -p "${OUTPUT_ROOT}"

if [ -e /proc/sys/kernel/yama/ptrace_scope ] \
    && [ "$(cat /proc/sys/kernel/yama/ptrace_scope)" -ne 0 ] \
    && [ "$(id -u)" -ne 0 ] \
    && [[ $(getpcaps self) != *"cap_sys_ptrace"* ]] \
    && [[ $(getcap "$(readlink -f "$(which python3)")") != *"cap_sys_ptrace"* ]]; then
    echo "This test is skipped. ptrace permission is unavailable."
    touch "${OUTPUT_ROOT}/skipped"
    exit 0
fi

# One worker on one WGP keeps ATT scheduling deterministic while continuously launching
# simple_kernel. Each session traces iterations 1-3, which must restart at the reattachment
# boundary. Keep cycle1 on disk because retained code-object metadata can reference it while
# cycle2 is decoded.
echo "Launching test application: ${TEST_APP}"
HSA_CU_MASK=0:0-1 LD_PRELOAD="${ROCPROF_PRELOAD:-}" "${TEST_APP}" 1 1 &
APP_PID=$!

wait_for_attach_ready "${APP_PID}"

if ! kill -0 "${APP_PID}" 2>/dev/null; then
    echo "Error: test application failed to start"
    exit 1
fi

run_attach_cycle cycle1
run_attach_cycle cycle2

echo "Stopping test application..."
kill -2 "${APP_PID}"
if ! wait "${APP_PID}"; then
    echo "Error: test application exited with a failure"
    exit 1
fi
APP_PID=""

echo "Repeated ATT attachment completed successfully"
