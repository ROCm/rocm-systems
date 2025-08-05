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
    rocprofiler-sdk-tests-rocprofv3-hip-stream-display
    LANGUAGES CXX
    VERSION 0.0.0)

find_package(rocprofiler-sdk REQUIRED)

rocprofiler_configure_pytest_files(CONFIG pytest.ini COPY validate.py conftest.py)

string(REPLACE "LD_PRELOAD=" "ROCPROF_PRELOAD=" PRELOAD_ENV
               "${ROCPROFILER_MEMCHECK_PRELOAD_ENV}")

set(hip-stream-display-env "${PRELOAD_ENV}")

add_test(
    NAME rocprofv3-test-hip-stream-display-execute
    COMMAND
        $<TARGET_FILE:rocprofiler-sdk::rocprofv3> --kernel-rename -s -d
        ${CMAKE_CURRENT_BINARY_DIR}/%tag%-trace -o out --output-format json pftrace
        --log-level env -- $<TARGET_FILE:hip-streams>)

set_tests_properties(
    rocprofv3-test-hip-stream-display-execute
    PROPERTIES TIMEOUT
               60
               LABELS
               "integration-tests"
               ENVIRONMENT
               "${hip-stream-display-env}"
               FAIL_REGULAR_EXPRESSION
               "${ROCPROFILER_DEFAULT_FAIL_REGEX}"
               DISABLED
               $<NOT:$<TARGET_EXISTS:hip-streams>>)

add_test(
    NAME rocprofv3-test-hip-stream-display-validate
    COMMAND
        ${Python3_EXECUTABLE} ${CMAKE_CURRENT_BINARY_DIR}/validate.py --json-input
        ${CMAKE_CURRENT_BINARY_DIR}/hip-streams-trace/out_results.json --pftrace-input
        ${CMAKE_CURRENT_BINARY_DIR}/hip-streams-trace/out_results.pftrace)

set_tests_properties(
    rocprofv3-test-hip-stream-display-validate
    PROPERTIES TIMEOUT
               60
               LABELS
               "integration-tests"
               DEPENDS
               rocprofv3-test-hip-stream-display-execute
               FAIL_REGULAR_EXPRESSION
               "AssertionError"
               DISABLED
               $<NOT:$<TARGET_EXISTS:hip-streams>>)
