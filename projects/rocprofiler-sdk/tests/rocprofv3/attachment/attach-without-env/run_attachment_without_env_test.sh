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

set -euo pipefail

TEST_APP=$1
ROCPROFV3=$2
OUTPUT_DIR=$3
TARGET_LOG=${OUTPUT_DIR}/target.log
ATTACH_LOG=${OUTPUT_DIR}/attach.log

unset ROCP_TOOL_ATTACH
mkdir -p "${OUTPUT_DIR}"

cleanup()
{
    if [[ -n "${APP_PID:-}" ]] && kill -0 "${APP_PID}" 2>/dev/null; then
        kill "${APP_PID}" 2>/dev/null || true
        wait "${APP_PID}" 2>/dev/null || true
    fi
}
trap cleanup EXIT

LD_PRELOAD="${ROCPROF_PRELOAD:-}" "${TEST_APP}" 1 1 1 >"${TARGET_LOG}" 2>&1 &
APP_PID=$!

for _ in $(seq 1 50); do
    if grep -q "After first call" "${TARGET_LOG}"; then
        break
    fi
    if ! kill -0 "${APP_PID}" 2>/dev/null; then
        echo "Test application exited before attachment"
        cat "${TARGET_LOG}"
        exit 1
    fi
    sleep 0.1
done

if ! grep -q "After first call" "${TARGET_LOG}"; then
    echo "Test application did not become ready"
    cat "${TARGET_LOG}"
    exit 1
fi

set +e
LD_PRELOAD="${ROCPROF_PRELOAD:-}" "${ROCPROFV3}" --attach "${APP_PID}" \
    --attach-duration-msec 100 -s -d "${OUTPUT_DIR}" -o attach-without-env \
    >"${ATTACH_LOG}" 2>&1
ATTACH_STATUS=$?
set -e

if [[ ${ATTACH_STATUS} -eq 0 ]]; then
    echo "rocprofv3 attach succeeded without ROCP_TOOL_ATTACH=1"
    cat "${ATTACH_LOG}"
    exit 1
fi

if ! grep -q "does not have the rocprofiler attachment library loaded" "${ATTACH_LOG}"; then
    echo "rocprofv3 attach failed without reporting why attachment is unavailable"
    cat "${ATTACH_LOG}"
    exit 1
fi

if ! kill -0 "${APP_PID}" 2>/dev/null; then
    echo "Test application did not survive the failed attachment"
    cat "${TARGET_LOG}"
    cat "${ATTACH_LOG}"
    exit 1
fi

echo "rocprofv3 attach failed as expected and the target remained alive"
