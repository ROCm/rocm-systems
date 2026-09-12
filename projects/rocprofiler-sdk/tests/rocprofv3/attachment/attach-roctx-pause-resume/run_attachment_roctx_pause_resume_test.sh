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

wait_for_profiler_attached() {
    local pid=$1
    local log_file=$2
    local max_wait=30
    local elapsed=0
    echo "Waiting for rocprofv3 to finish attaching to PID ${pid}..."
    while [ $elapsed -lt $max_wait ]; do
        if grep -q "Attaching to PID ${pid} .* :: success" "${log_file}" 2>/dev/null; then
            echo "rocprofv3 attach completed (${elapsed}s elapsed)"
            return 0
        fi
        if ! kill -0 "${ROCPROF_PID}" 2>/dev/null; then
            echo "rocprofv3 exited before reporting attach success"
            return 1
        fi
        sleep 1
        elapsed=$((elapsed + 1))
    done
    echo "Timed out after ${max_wait}s waiting for rocprofv3 attach success"
    return 1
}

TEST_APP=$1
ROCPROFV3=$2
OUTPUT_DIR=${3:-${PWD}}
LOG_LEVEL=${4:-info}
OUTPUT_FILENAME=${5:-out}
MODE=${6:-normal}

export ROCP_TOOL_ATTACH=1

OUTPUT_SUBDIR="attachment-roctx-${MODE}-output"
TRIGGER_FILE="${OUTPUT_DIR}/${OUTPUT_SUBDIR}/start-workload"
ROCPROF_LOG="${OUTPUT_DIR}/${OUTPUT_SUBDIR}/rocprofv3.log"
APP_PID=""
APP_OUTPUT_PID=""
ROCPROF_PID=""

cleanup() {
    if [ -n "${ROCPROF_PID}" ]; then
        kill "${ROCPROF_PID}" 2>/dev/null || true
        wait "${ROCPROF_PID}" 2>/dev/null || true
    fi

    if [ -n "${APP_PID}" ]; then
        kill -2 "${APP_PID}" 2>/dev/null || true
        wait "${APP_PID}" 2>/dev/null || true
    fi
}

trap cleanup EXIT

if [ "${MODE}" = "normal" ]; then
    ROCPROFV3_FLAGS=(--marker-trace --kernel-trace)
elif [ "${MODE}" = "selected" ]; then
    ROCPROFV3_FLAGS=(--selected-regions --kernel-trace)
else
    echo "Unknown test mode: ${MODE}"
    exit 1
fi

rm -rf "${OUTPUT_DIR}/${OUTPUT_SUBDIR}"
mkdir -p "${OUTPUT_DIR}/${OUTPUT_SUBDIR}"

if [ -e /proc/sys/kernel/yama/ptrace_scope ]                             \
&& [ $(cat /proc/sys/kernel/yama/ptrace_scope) -ne 0 ]                   \
&& [ $(id -u) -ne 0 ]                                                    \
&& [[ $(getpcaps self) != *"cap_sys_ptrace"* ]]                          \
&& [[ $(getcap $(readlink -f $(which python3))) != *"cap_sys_ptrace"* ]]
    then
    echo "ptrace_scope is not 0, user is not root, and CAP_SYS_PTRACE is not present, so test cannot be completed. This test is skipped."
    touch "${OUTPUT_DIR}/${OUTPUT_SUBDIR}/skipped"
    exit 0
fi

echo "Launching ROCTx attach pause/resume target in ${MODE} mode"
LD_PRELOAD="${ROCPROF_PRELOAD}" "${TEST_APP}" "${MODE}" "${TRIGGER_FILE}" &
APP_PID=$!
APP_OUTPUT_PID=$APP_PID

wait_for_attach_ready "${APP_PID}"

if ! kill -0 "${APP_PID}" 2>/dev/null; then
    echo "Test application failed to start or exited early"
    exit 1
fi

if [ ! -f "${ROCPROFV3}" ]; then
    echo "Error: rocprofv3 not found at ${ROCPROFV3}"
    exit 1
fi

echo "Attaching profiler to PID $APP_PID in ${MODE} mode..."
PYTHONUNBUFFERED=1 LD_PRELOAD="${ROCPROF_PRELOAD}" "${ROCPROFV3}" --attach "${APP_PID}" \
    --attach-duration-msec 8000 \
    "${ROCPROFV3_FLAGS[@]}" \
    -f json --attach-sync-output \
    -d "${OUTPUT_DIR}/${OUTPUT_SUBDIR}" \
    --log-level "${LOG_LEVEL}" >"${ROCPROF_LOG}" 2>&1 &
ROCPROF_PID=$!

if ! wait_for_profiler_attached "${APP_PID}" "${ROCPROF_LOG}"; then
    echo "rocprofv3 output:"
    cat "${ROCPROF_LOG}" 2>/dev/null || true
    wait "${ROCPROF_PID}" 2>/dev/null || true
    ROCPROF_PID=""
    exit 1
fi

touch "${TRIGGER_FILE}"

if wait "${ROCPROF_PID}"; then
    ROCPROF_PID=""
else
    ROCPROF_EXIT_CODE=$?
    ROCPROF_PID=""
    echo "rocprofv3 attach test failed with exit code $ROCPROF_EXIT_CODE"
    exit 1
fi

echo "Profiler detached successfully"

kill -2 "${APP_PID}" 2>/dev/null
if wait "${APP_PID}"; then
    APP_PID=""
else
    APP_EXIT_CODE=$?
    APP_PID=""
    echo "Test application failed with exit code $APP_EXIT_CODE"
    exit 1
fi

echo "Checking for generated output files..."
ls -laR "${OUTPUT_DIR}/${OUTPUT_SUBDIR}/"

JSON_COUNT=$(find "${OUTPUT_DIR}/${OUTPUT_SUBDIR}/" -name "*.json" | wc -l)
if [ "${JSON_COUNT}" -eq 0 ]; then
    echo "Error: No JSON files were generated"
    exit 1
fi

APP_JSON=$(find "${OUTPUT_DIR}/${OUTPUT_SUBDIR}/" -name "${APP_OUTPUT_PID}_results.json" | head -1)
if [ -z "$APP_JSON" ]; then
    echo "Error: Could not find app (PID ${APP_OUTPUT_PID}) JSON output in ${OUTPUT_DIR}/${OUTPUT_SUBDIR}/"
    exit 1
fi
echo "Found app JSON output: $APP_JSON"

APP_OUTPUT_DIR=$(dirname "$APP_JSON")

for src in "${APP_OUTPUT_DIR}/${APP_OUTPUT_PID}"_*.json; do
    [ -f "$src" ] || continue
    dst_name=$(basename "$src" | sed "s/^${APP_OUTPUT_PID}_/${OUTPUT_FILENAME}_/")
    cp "$src" "${OUTPUT_DIR}/${OUTPUT_SUBDIR}/${dst_name}"
    echo "Copied $(basename "$src") -> ${dst_name}"
done

if [ ! -f "${OUTPUT_DIR}/${OUTPUT_SUBDIR}/${OUTPUT_FILENAME}_results.json" ]; then
    echo "Error: Expected output file ${OUTPUT_DIR}/${OUTPUT_SUBDIR}/${OUTPUT_FILENAME}_results.json not found"
    exit 1
fi

echo "ROCTx attach pause/resume ${MODE} test completed successfully"
exit 0
