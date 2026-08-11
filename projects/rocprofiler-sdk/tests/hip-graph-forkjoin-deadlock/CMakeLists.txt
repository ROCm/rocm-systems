# MIT License
#
# Copyright (c) 2023-2026 Advanced Micro Devices, Inc. All rights reserved.
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
# Regression coverage for the queue-interposition completion-handler deadlock: a
# multi-stream HIP-graph fork/join workload produces many concurrent, mutually dependent
# completion signals. Under the previous per-signal blocking handler pool this deadlocked
# (the execute step would hang until TIMEOUT); the single monitor thread that waits on all
# signals at once completes normally. The validate step additionally confirms the expected
# dispatch records were emitted.
#
cmake_minimum_required(VERSION 3.21.0 FATAL_ERROR)

project(
    rocprofiler-sdk-tests-hip-graph-forkjoin-deadlock
    LANGUAGES CXX
    VERSION 0.0.0)

find_package(rocprofiler-sdk REQUIRED)

# width=8 depth=8 replays=30: enough concurrent barrier-coupled branches to saturate the
# completion path. The TIMEOUT is the failure signal for the deadlock -- the workload
# completes in a few seconds when healthy.
rocprofiler_add_integration_execute_test(
    hip-graph-forkjoin-deadlock
    TARGET hip-graph-forkjoin
    ARGS 8 8 30
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
    ARGS --input ${CMAKE_CURRENT_BINARY_DIR}/hip-graph-forkjoin-deadlock-test.json)
