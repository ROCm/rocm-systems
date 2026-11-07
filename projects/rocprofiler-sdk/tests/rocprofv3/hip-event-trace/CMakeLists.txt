# MIT License
#
# Copyright (c) 2025-2026 Advanced Micro Devices, Inc. All rights reserved.
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

cmake_minimum_required(VERSION 3.21.0 FATAL_ERROR)

project(
    rocprofiler-sdk-tests-rocprofv3-hip-event-trace
    LANGUAGES CXX
    VERSION 0.0.0)

find_package(rocprofiler-sdk REQUIRED)

string(REPLACE "LD_PRELOAD=" "ROCPROF_PRELOAD=" PRELOAD_ENV
               "${ROCPROFILER_MEMCHECK_PRELOAD_ENV}")

# The SDK logs an error and drops the HIP events a barrier times when the barrier has no
# valid GPU timestamps; treat that as a failure so a regression in GPU timing is not
# silent.
set(hip-event-fail-regex
    "${ROCPROFILER_DEFAULT_FAIL_REGEX}|no valid GPU timestamps for a HIP event")

# A deferred wait holds its correlation ID until a barrier claims it. One that CLR never
# turned into a GPU dependency (for example, a same-stream wait) is never claimed, and its
# ID is only retired as dangling at finalization.
string(APPEND hip-event-fail-regex "|retiring dangling correlation ID")

rocprofiler_add_integration_execute_test(
    rocprofv3-test-hip-event-trace
    COMMAND
        $<TARGET_FILE:rocprofiler-sdk::rocprofv3> --hip-event-trace --hip-runtime-trace
        --kernel-trace -d ${CMAKE_CURRENT_BINARY_DIR}/%tag%-trace -o out --output-format
        json rocpd --log-level env -- $<TARGET_FILE:hip-events>
    DEPENDS hip-events
    TIMEOUT 45
    LABELS "integration-tests;rocpd"
    PRELOAD "${PRELOAD_ENV}"
    FAIL_REGULAR_EXPRESSION "${hip-event-fail-regex}"
    FIXTURES_SETUP rocprofv3-test-hip-event-trace)

# Direct otf2 output from rocprofv3 goes through the deprecated C++ generator; the
# supported path is rocpd convert, which is what otf2.py implements.
set(rocprofv3-hip-event-rocpd-env
    "PYTHONPATH=${rocprofiler-sdk_LIB_DIR}/python${Python3_VERSION_MAJOR}.${Python3_VERSION_MINOR}/site-packages"
    )

find_package(Python3 REQUIRED)

rocprofiler_add_integration_execute_test(
    rocprofv3-test-hip-event-trace-otf2-generation
    COMMAND
        ${Python3_EXECUTABLE} -m rocpd convert -f otf2 -d
        ${CMAKE_CURRENT_BINARY_DIR}/hip-events-trace -o out -i
        ${CMAKE_CURRENT_BINARY_DIR}/hip-events-trace/out_results.db
    DEPENDS rocprofiler-sdk::rocprofv3
    TIMEOUT 120
    LABELS "integration-tests;rocpd"
    ENVIRONMENT "${rocprofv3-hip-event-rocpd-env}"
    FIXTURES_SETUP rocprofv3-test-hip-event-trace-otf2
    FIXTURES_REQUIRED rocprofv3-test-hip-event-trace)

# Convert to Perfetto with both kernel groupings: HIP events must follow kernels onto the
# stream track by default and onto the agent-queue track with --group-by-queue.
foreach(_grouping stream queue)
    if(_grouping STREQUAL "queue")
        set(_grouping_args --group-by-queue)
    else()
        set(_grouping_args)
    endif()

    rocprofiler_add_integration_execute_test(
        rocprofv3-test-hip-event-trace-pftrace-${_grouping}-generation
        COMMAND
            ${Python3_EXECUTABLE} -m rocpd convert -f pftrace ${_grouping_args} -d
            ${CMAKE_CURRENT_BINARY_DIR}/hip-events-trace -o out_${_grouping} -i
            ${CMAKE_CURRENT_BINARY_DIR}/hip-events-trace/out_results.db
        DEPENDS rocprofiler-sdk::rocprofv3
        TIMEOUT 120
        LABELS "integration-tests;rocpd"
        ENVIRONMENT "${rocprofv3-hip-event-rocpd-env}"
        FIXTURES_SETUP rocprofv3-test-hip-event-trace-pftrace
        FIXTURES_REQUIRED rocprofv3-test-hip-event-trace)
endforeach()

rocprofiler_add_integration_validate_test(
    rocprofv3-test-hip-event-trace
    TEST_PATHS validate.py
    COPY conftest.py
    CONFIG pytest.ini
    ARGS --json-input
         ${CMAKE_CURRENT_BINARY_DIR}/hip-events-trace/out_results.json
         --rocpd-input
         ${CMAKE_CURRENT_BINARY_DIR}/hip-events-trace/out_results.db
         --otf2-input
         ${CMAKE_CURRENT_BINARY_DIR}/hip-events-trace/out_results.otf2
         --pftrace-stream-input
         ${CMAKE_CURRENT_BINARY_DIR}/hip-events-trace/out_stream_results.pftrace
         --pftrace-queue-input
         ${CMAKE_CURRENT_BINARY_DIR}/hip-events-trace/out_queue_results.pftrace
    TIMEOUT 45
    LABELS "integration-tests;rocpd"
    FIXTURES_REQUIRED rocprofv3-test-hip-event-trace rocprofv3-test-hip-event-trace-otf2
                      rocprofv3-test-hip-event-trace-pftrace)

# Test hip event tracing without HIP API tracing enabled, exercising the correlation ID
# self-construction path.
rocprofiler_add_integration_execute_test(
    rocprofv3-test-hip-event-trace-no-hip-api
    COMMAND
        $<TARGET_FILE:rocprofiler-sdk::rocprofv3> --hip-event-trace --kernel-trace -d
        ${CMAKE_CURRENT_BINARY_DIR}/hip-events-no-hip-api-trace -o out --output-format
        json --log-level env -- $<TARGET_FILE:hip-events>
    DEPENDS hip-events
    TIMEOUT 45
    LABELS "integration-tests"
    PRELOAD "${PRELOAD_ENV}"
    FAIL_REGULAR_EXPRESSION "${hip-event-fail-regex}"
    FIXTURES_SETUP rocprofv3-test-hip-event-trace-no-hip-api)

rocprofiler_add_integration_validate_test(
    rocprofv3-test-hip-event-trace-no-hip-api
    TEST_PATHS validate_no_hip_api.py
    COPY conftest.py
    CONFIG pytest.ini
    ARGS --json-input
         ${CMAKE_CURRENT_BINARY_DIR}/hip-events-no-hip-api-trace/out_results.json
    TIMEOUT 45
    LABELS "integration-tests"
    FIXTURES_REQUIRED rocprofv3-test-hip-event-trace-no-hip-api)

# Test hip event tracing under kernel replay, which re-runs each dispatch once per counter
# group. Replay first waits for all work in flight on the agent, so a record that a later
# wait depends on can already be complete when the wait is made; CLR then creates no
# dependency and no WAIT is reported. validate_no_hip_api.py checks the records and their
# links rather than exact counts, so it holds here as well.
rocprofiler_add_integration_execute_test(
    rocprofv3-test-hip-event-trace-kernel-replay
    COMMAND
        $<TARGET_FILE:rocprofiler-sdk::rocprofv3> --hip-event-trace --kernel-trace --pmc
        SQ_WAVES SQ_INSTS_VALU GRBM_COUNT --pmc SQ_WAVES SQ_INSTS_VALU GRBM_GUI_ACTIVE
        --replay-mode kernel --kernel-replay-beta-enabled -d
        ${CMAKE_CURRENT_BINARY_DIR}/hip-events-kernel-replay-trace -o out --output-format
        json --log-level env -- $<TARGET_FILE:hip-events>
    DEPENDS hip-events
    TIMEOUT 120
    LABELS "integration-tests;kernel-replay"
    PRELOAD "${PRELOAD_ENV}"
    FAIL_REGULAR_EXPRESSION "${hip-event-fail-regex}"
    FIXTURES_SETUP rocprofv3-test-hip-event-trace-kernel-replay)

rocprofiler_add_integration_validate_test(
    rocprofv3-test-hip-event-trace-kernel-replay
    TEST_PATHS validate_no_hip_api.py
    COPY conftest.py
    CONFIG pytest.ini
    ARGS --json-input
         ${CMAKE_CURRENT_BINARY_DIR}/hip-events-kernel-replay-trace/out_results.json
    TIMEOUT 45
    LABELS "integration-tests;kernel-replay"
    FIXTURES_REQUIRED rocprofv3-test-hip-event-trace-kernel-replay)
