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

# The prompt is printed only once attachment has completed and the SIGINT handler is
# installed, so a detach triggered earlier would be buffered or fatal.
wait_for_detach_prompt() {
    local pid=$1
    local log=$2
    local max_wait=30
    local elapsed=0
    echo "Waiting for rocprofv3 to prompt for detach..."
    while [ $elapsed -lt $max_wait ]; do
        if grep -q "Press Enter to detach" ${log} 2>/dev/null; then
            echo "Detach prompt shown (${elapsed}s elapsed)"
            return 0
        fi
        if ! kill -0 ${pid} 2>/dev/null; then
            echo "rocprofv3 exited before prompting for detach"
            cat ${log}
            return 1
        fi
        sleep 1
        elapsed=$((elapsed + 1))
    done
    echo "Timed out after ${max_wait}s waiting for the detach prompt"
    cat ${log}
    return 1
}

TEST_APP=$1
ROCPROFV3=$2
OUTPUT_DIR=${3:-${PWD}}
LOG_LEVEL=${4:-info}
OUTPUT_FILENAME=${5:-out}
DETACH_MODE=${6:-enter}

case ${DETACH_MODE} in
    enter | sigint) ;;
    *)
        echo "Error: unknown detach mode '${DETACH_MODE}'"
        exit 1
        ;;
esac

# Seconds to profile before triggering the detach.
SETTLE=3

export ROCP_TOOL_ATTACH=1

OUTPUT_SUBDIR="attachment-detach-${DETACH_MODE}-output"

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

echo "Launching test application: ${TEST_APP}"
LD_PRELOAD=${ROCPROF_PRELOAD} ${TEST_APP} &
APP_PID=$!

# The workload and the profiler outlive an early exit under `set -e`, and while they
# live they hold this script's output open, so ctest would block until its timeout.
cleanup_app() {
    for pid in ${ROCPROF_PID} ${APP_PID}; do
        if kill -0 ${pid} 2>/dev/null; then
            kill -9 ${pid} 2>/dev/null
        fi
    done
}
trap cleanup_app EXIT

wait_for_attach_ready $APP_PID

if ! kill -0 $APP_PID 2>/dev/null; then
    echo "Test application failed to start or exited early"
    exit 1
fi

echo "Test application started with PID: $APP_PID"

if [ ! -f "${ROCPROFV3}" ]; then
    echo "Error: rocprofv3 not found at ${ROCPROFV3}"
    exit 1
fi

# Omitting --attach-duration-msec selects the documented interactive detach: rocprofv3
# profiles until Enter arrives on stdin, or until it is sent SIGINT.
ROCPROFV3_ARGS=(
    --attach $APP_PID
    -s
    -f json
    --attach-sync-output
    -d ${OUTPUT_DIR}/${OUTPUT_SUBDIR}
    --log-level ${LOG_LEVEL}
)

# Hold stdin open so the prompt blocks instead of reaching EOF; the detach is then
# triggered only by the newline or SIGINT sent below.
FIFO=${OUTPUT_DIR}/${OUTPUT_SUBDIR}/stdin.fifo
ROCPROF_LOG=${OUTPUT_DIR}/${OUTPUT_SUBDIR}/rocprofv3.log
mkfifo ${FIFO}
exec 3<>${FIFO}

echo "Attaching to PID $APP_PID..."
LD_PRELOAD=${ROCPROF_PRELOAD} ${ROCPROFV3} "${ROCPROFV3_ARGS[@]}" <${FIFO} >${ROCPROF_LOG} &
ROCPROF_PID=$!

wait_for_detach_prompt ${ROCPROF_PID} ${ROCPROF_LOG}

echo "Profiling for ${SETTLE}s before detaching via '${DETACH_MODE}'..."
sleep ${SETTLE}

case ${DETACH_MODE} in
    enter)
        printf '\n' >&3
        ;;
    sigint)
        echo "Sending SIGINT to rocprofv3 (PID ${ROCPROF_PID})..."
        kill -2 ${ROCPROF_PID}
        ;;
esac

set +e
wait ${ROCPROF_PID}
ROCPROF_EXIT_CODE=$?
set -e
exec 3>&-
rm -f ${FIFO}
cat ${ROCPROF_LOG}

if [ ${ROCPROF_EXIT_CODE} -ne 0 ]; then
    echo "rocprofv3 attach test failed with exit code ${ROCPROF_EXIT_CODE}"
    exit 1
fi

echo "Profiler detached successfully"

# rocprofv3 exits 0 even when the detach call fails, so require the success report.
if ! grep -Eq "Detaching from PID ${APP_PID} using library .* :: success" ${ROCPROF_LOG}; then
    echo "Error: rocprofv3 did not report a successful detach from PID ${APP_PID}"
    exit 1
fi

# Only the SIGINT handler reports the signal, so this shows which trigger ended the attach.
case ${DETACH_MODE} in
    enter)
        if grep -q "Caught signal SIGINT" ${ROCPROF_LOG}; then
            echo "Error: detach was triggered by SIGINT instead of Enter"
            exit 1
        fi
        ;;
    sigint)
        if ! grep -q "Caught signal SIGINT" ${ROCPROF_LOG}; then
            echo "Error: rocprofv3 did not report catching SIGINT"
            exit 1
        fi
        ;;
esac

# The target must survive the detach; only the profiler was asked to stop.
if ! kill -0 $APP_PID 2>/dev/null; then
    echo "Error: test application exited when the profiler detached"
    exit 1
fi
echo "Test application still running after detach"

echo "Sending SIGINT to application..."
kill -2 $APP_PID 2>/dev/null
set +e
wait $APP_PID
APP_EXIT_CODE=$?
set -e

if [ $APP_EXIT_CODE -ne 0 ]; then
    echo "Test application failed with exit code $APP_EXIT_CODE"
    exit 1
fi

echo "Test application completed successfully"
ls -laR ${OUTPUT_DIR}/${OUTPUT_SUBDIR}/

APP_JSON=$(find ${OUTPUT_DIR}/${OUTPUT_SUBDIR}/ -name "${APP_PID}_results.json" | head -1)
if [ -z "$APP_JSON" ]; then
    echo "Error: Could not find app (PID ${APP_PID}) JSON output in ${OUTPUT_DIR}/${OUTPUT_SUBDIR}/"
    exit 1
fi
echo "Found results JSON: $APP_JSON"

cp "$APP_JSON" "${OUTPUT_DIR}/${OUTPUT_SUBDIR}/${OUTPUT_FILENAME}_results.json"
echo "Copied $(basename $APP_JSON) -> ${OUTPUT_FILENAME}_results.json"

echo "Attachment detach (${DETACH_MODE}) test completed successfully"
exit 0
