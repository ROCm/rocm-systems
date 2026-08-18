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
    FIXTURES_SETUP rocprofv3-test-hip-event-trace)

rocprofiler_add_integration_validate_test(
    rocprofv3-test-hip-event-trace
    TEST_PATHS validate.py
    COPY conftest.py
    CONFIG pytest.ini
    ARGS --json-input ${CMAKE_CURRENT_BINARY_DIR}/hip-events-trace/out_results.json
         --rocpd-input ${CMAKE_CURRENT_BINARY_DIR}/hip-events-trace/out_results.db
    TIMEOUT 45
    LABELS "integration-tests;rocpd"
    FIXTURES_REQUIRED rocprofv3-test-hip-event-trace)
