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

#
# ROCM-33989: hipStreamCreate deadlocked under rocprofv3 when a profiled application
# destroyed a stream while a HIP completion callback for it was still pending and then
# created a new stream. On the legacy (non-inline) queue path rocprofiler-sdk replaced the
# plain queue HIP requests through hsa_amd_queue_create with an InterceptQueue by
# destroying that plain queue inside the call; hsa_queue_destroy waits for ROCr's
# async-events thread, which was blocked on the HIP device lock held across
# hipStreamCreate. The execute step runs stream create/destroy churn with completion
# handlers in flight on the legacy queue path and must run to completion; TIMEOUT is the
# failure signal. The validate step confirms the churned streams' dispatches were traced,
# so the queues really were created as working InterceptQueues.
#
cmake_minimum_required(VERSION 3.21.0 FATAL_ERROR)

project(
    rocprofiler-sdk-tests-hip-stream-cycle-deadlock
    LANGUAGES CXX
    VERSION 0.0.0)

find_package(rocprofiler-sdk REQUIRED)

# Workload geometry, shared by both steps so they cannot drift. The poll phase exercises
# more stream creations than HIP's hardware-queue pool below (GPU_MAX_HW_QUEUES) so that
# every pool slot is created through hsa_amd_queue_create while handlers are in flight;
# the burst phase adds the many-pending-batches variant. A healthy run finishes in a few
# seconds.
set(STREAM_CYCLE_POLL_CYCLES 64)
set(STREAM_CYCLE_KERNEL_MS 20)
set(STREAM_CYCLE_BURST_CYCLES 8)
set(STREAM_CYCLE_BURST_LAUNCHES 4096)
math(
    EXPR
    STREAM_CYCLE_MIN_DISPATCHES
    "${STREAM_CYCLE_POLL_CYCLES} + ${STREAM_CYCLE_BURST_CYCLES} * ${STREAM_CYCLE_BURST_LAUNCHES}"
    )

rocprofiler_add_integration_execute_test(
    hip-stream-cycle-deadlock
    COMMAND
        $<TARGET_FILE:rocprofiler-sdk::rocprofv3> --kernel-trace --hip-graph-trace -d
        ${CMAKE_CURRENT_BINARY_DIR}/out -o out --output-format csv --
        $<TARGET_FILE:hip-stream-cycle> ${STREAM_CYCLE_POLL_CYCLES}
        ${STREAM_CYCLE_KERNEL_MS} ${STREAM_CYCLE_BURST_CYCLES}
        ${STREAM_CYCLE_BURST_LAUNCHES}
    DEPENDS hip-stream-cycle
    TIMEOUT 120
    WSL_TIMEOUT_SCALE
    LABELS "integration-tests"
    ENVIRONMENT
        # Every stream must get its own hardware queue so that each hipStreamCreate goes
        # through hsa_amd_queue_create; HIP's default pool of 4 queues is reused after the
        # first streams.
        "GPU_MAX_HW_QUEUES=64"
        # Pin the legacy (InterceptQueue) queue path, the one this regression is about.
        # --hip-graph-trace already selects it today; set it explicitly so a future
        # default-logic change cannot silently move the test onto the inline path.
        "ROCPROFILER_QUEUE_INTERPOSITION=0"
    FIXTURES_SETUP hip-stream-cycle-deadlock)

rocprofiler_add_integration_validate_test(
    hip-stream-cycle-deadlock
    TEST_PATHS validate.py
    COPY conftest.py
    CONFIG pytest.ini
    LABELS "integration-tests"
    TIMEOUT 45
    FIXTURES_REQUIRED hip-stream-cycle-deadlock
    ARGS --kernel-trace ${CMAKE_CURRENT_BINARY_DIR}/out/out_kernel_trace.csv
         --min-dispatches ${STREAM_CYCLE_MIN_DISPATCHES})
