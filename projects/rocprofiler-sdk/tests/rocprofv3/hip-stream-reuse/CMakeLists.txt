# MIT License
#
# Copyright (c) 2023-2025 Advanced Micro Devices, Inc. All rights reserved.
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
cmake_minimum_required(VERSION 3.21.0 FATAL_ERROR)

project(
    rocprofiler-sdk-tests-rocprofv3-hip-stream-reuse
    LANGUAGES CXX
    VERSION 0.0.0)

find_package(rocprofiler-sdk REQUIRED)

string(REPLACE "LD_PRELOAD=" "ROCPROF_PRELOAD=" PRELOAD_ENV
               "${ROCPROFILER_MEMCHECK_PRELOAD_ENV}")

set(hip-stream-reuse-env "${PRELOAD_ENV}")

rocprofiler_add_integration_execute_test(
    rocprofv3-test-hip-stream-reuse
    COMMAND
        $<TARGET_FILE:rocprofiler-sdk::rocprofv3> -s -d
        ${CMAKE_CURRENT_BINARY_DIR}/%tag%-trace -o out --output-format json --log-level
        env -- $<TARGET_FILE:hip-stream-reuse>
        ${CMAKE_CURRENT_BINARY_DIR}/hip-stream-reuse-trace/stream_pointers.txt
    DEPENDS hip-stream-reuse
    TIMEOUT 60
    LABELS "integration-tests"
    PRELOAD "${PRELOAD_ENV}"
    FIXTURES_SETUP rocprofv3-test-hip-stream-reuse
    DISABLED $<NOT:$<TARGET_EXISTS:hip-stream-reuse>>)

rocprofiler_add_integration_validate_test(
    rocprofv3-test-hip-stream-reuse
    TEST_PATHS validate.py
    COPY conftest.py
    CONFIG pytest.ini
    ARGS --json-input ${CMAKE_CURRENT_BINARY_DIR}/hip-stream-reuse-trace/out_results.json
         --pointer-file
         ${CMAKE_CURRENT_BINARY_DIR}/hip-stream-reuse-trace/stream_pointers.txt
    TIMEOUT 60
    LABELS "integration-tests"
    FIXTURES_REQUIRED rocprofv3-test-hip-stream-reuse
    DISABLED $<NOT:$<TARGET_EXISTS:hip-stream-reuse>>)
