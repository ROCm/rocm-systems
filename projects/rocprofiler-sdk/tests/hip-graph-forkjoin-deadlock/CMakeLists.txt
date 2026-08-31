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
# Queue-interposition completion path under concurrency: a HIP-graph fork/join workload
# produces many mutually dependent completion signals in flight at once. The execute step
# asserts the workload runs to completion, with TIMEOUT as the failure signal; it does not
# complete if each outstanding completion needs a thread of its own to wait on it. The
# validate step confirms every dispatch record was emitted.
#
cmake_minimum_required(VERSION 3.21.0 FATAL_ERROR)

project(
    rocprofiler-sdk-tests-hip-graph-forkjoin-deadlock
    LANGUAGES CXX
    VERSION 0.0.0)

find_package(rocprofiler-sdk REQUIRED)

# Workload geometry. Both steps read it from here: the execute step runs with it and the
# validate step derives the minimum dispatch count it expects from it, so the two cannot
# drift. The values are empirical, chosen to keep enough completions outstanding at once
# for the concurrency this test is about. A healthy run finishes in a few seconds.
set(FORKJOIN_WIDTH 8)
set(FORKJOIN_DEPTH 8)
set(FORKJOIN_REPLAYS 30)

rocprofiler_add_integration_execute_test(
    hip-graph-forkjoin-deadlock
    TARGET hip-graph-forkjoin
    ARGS ${FORKJOIN_WIDTH} ${FORKJOIN_DEPTH} ${FORKJOIN_REPLAYS}
    TIMEOUT 60
    LABELS "integration-tests"
    PRELOAD "$<TARGET_FILE:rocprofiler-sdk-json-tool>"
    ENVIRONMENT
        "ROCPROFILER_TOOL_OUTPUT_FILE=hip-graph-forkjoin-deadlock-test.json"
        "ROCPROFILER_TOOL_CONTEXTS=KERNEL_DISPATCH_CALLBACK,KERNEL_DISPATCH_BUFFERED,CODE_OBJECT"
        # Pin interposition on: this regression only exercises the monitor path when the
        # inline path is active. It would default on for these contexts, but set it
        # explicitly so a future default-logic change cannot silently no-op the test.
        "ROCPROFILER_QUEUE_INTERPOSITION=1"
    FIXTURES_SETUP hip-graph-forkjoin-deadlock)

rocprofiler_add_integration_validate_test(
    hip-graph-forkjoin-deadlock
    TEST_PATHS validate.py
    COPY conftest.py
    CONFIG pytest.ini
    LABELS "integration-tests"
    TIMEOUT 45
    FIXTURES_REQUIRED hip-graph-forkjoin-deadlock
    ARGS --input ${CMAKE_CURRENT_BINARY_DIR}/hip-graph-forkjoin-deadlock-test.json
         --width ${FORKJOIN_WIDTH} --depth ${FORKJOIN_DEPTH} --replays
         ${FORKJOIN_REPLAYS})
