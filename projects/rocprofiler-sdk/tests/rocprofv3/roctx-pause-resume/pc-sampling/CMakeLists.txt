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
    rocprofiler-sdk-tests-rocprofv3-roctx-pause-resume
    LANGUAGES CXX
    VERSION 0.0.0)

# Checking whether host-trap PC sampling is disabled. If so, skip the tests.
rocprofiler_sdk_pc_sampling_disabled(IS_PC_SAMPLING_DISABLED)

find_package(rocprofiler-sdk REQUIRED)

rocprofiler_configure_pytest_files(CONFIG pytest.ini COPY validate.py conftest.py)

string(REPLACE "LD_PRELOAD=" "ROCPROF_PRELOAD=" PRELOAD_ENV
               "${ROCPROFILER_MEMCHECK_PRELOAD_ENV}")

set(roctx-pause-resume-env "${PRELOAD_ENV}")

add_test(
    NAME rocprofv3-test-roctx-pause-resume-pc-sampling-execute
    COMMAND
        $<TARGET_FILE:rocprofiler-sdk::rocprofv3> --marker-trace --pc-sampling-unit time
        --pc-sampling-method host_trap --pc-sampling-interval 1 -d
        ${CMAKE_CURRENT_BINARY_DIR}/%tag%-pc-sampling -o out --output-format json csv
        --log-level env -- $<TARGET_FILE:roctx-pause-resume>)

set_tests_properties(
    rocprofv3-test-roctx-pause-resume-pc-sampling-execute
    PROPERTIES TIMEOUT
               45
               LABELS
               "integration-tests;pc-sampling;pause-resume"
               ENVIRONMENT
               "${roctx-pause-resume-env}"
               FAIL_REGULAR_EXPRESSION
               "${ROCPROFILER_DEFAULT_FAIL_REGEX}"
               SKIP_REGULAR_EXPRESSION
               "${ROCPROFV3_TESTS_PC_SAMPLING_SKIP_REGEX}"
               DISABLED
               "${IS_PC_SAMPLING_DISABLED}")

add_test(
    NAME rocprofv3-test-roctx-pause-resume-pc-sampling-validate
    COMMAND ${Python3_EXECUTABLE} ${CMAKE_CURRENT_BINARY_DIR}/validate.py --json-input
            ${CMAKE_CURRENT_BINARY_DIR}/roctx-pause-resume-pc-sampling/out_results.json)

set_tests_properties(
    rocprofv3-test-roctx-pause-resume-pc-sampling-validate
    PROPERTIES TIMEOUT
               45
               LABELS
               "integration-tests;pc-sampling;pause-resume"
               DEPENDS
               rocprofv3-test-roctx-pause-resume-pc-sampling-execute
               FAIL_REGULAR_EXPRESSION
               "AssertionError"
               SKIP_REGULAR_EXPRESSION
               "${ROCPROFV3_TESTS_PC_SAMPLING_SKIP_REGEX}"
               DISABLED
               "${IS_PC_SAMPLING_DISABLED}")
