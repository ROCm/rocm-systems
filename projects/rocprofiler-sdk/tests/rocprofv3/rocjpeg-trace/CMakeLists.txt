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
    rocprofiler-sdk-tests-rocprofv3-rocjpeg-tracing
    LANGUAGES CXX
    VERSION 0.0.0)

find_package(rocprofiler-sdk REQUIRED)
find_package(rocJPEG)

rocprofiler_configure_pytest_files(CONFIG pytest.ini COPY validate.py conftest.py)

string(REPLACE "LD_PRELOAD=" "ROCPROF_PRELOAD=" PRELOAD_ENV
               "${ROCPROFILER_MEMCHECK_PRELOAD_ENV}")

set(rocjpeg-tracing-env "${PRELOAD_ENV}")

set(rocJPEG_IMAGE_DIR "${ROCM_PATH}/share/rocjpeg/images")
if(TARGET rocjpeg-demo AND NOT EXISTS "${rocJPEG_IMAGE_DIR}")
    message(
        WARNING "Unable to find image directory for rocjpeg tests: ${rocJPEG_IMAGE_DIR}")
    set(IS_DISABLED ON)
endif()

if(NOT TARGET rocjpeg-demo)
    set(IS_DISABLED ON)
endif()

# CI Sanitizer run gives the following error: No target "rocjpeg-demo" Adding if-statement
# to avoid tests for now to stop error
add_test(
    NAME rocprofv3-test-rocjpeg-tracing-execute
    COMMAND
        $<TARGET_FILE:rocprofiler-sdk::rocprofv3> --rocjpeg-trace -d
        ${CMAKE_CURRENT_BINARY_DIR}/%tag%-trace -o out --output-format json csv
        --log-level env --
        $<IF:$<TARGET_EXISTS:rocjpeg-demo>,$<$<TARGET_EXISTS:rocjpeg-demo>:$<TARGET_FILE:rocjpeg-demo>>,rocjpeg-demo>
        -i ${rocJPEG_IMAGE_DIR})

set_tests_properties(
    rocprofv3-test-rocjpeg-tracing-execute
    PROPERTIES TIMEOUT
               45
               LABELS
               "integration-tests"
               ENVIRONMENT
               "${rocjpeg-tracing-env}"
               FAIL_REGULAR_EXPRESSION
               "${ROCPROFILER_DEFAULT_FAIL_REGEX}"
               DISABLED
               "${IS_DISABLED}")

add_test(
    NAME rocprofv3-test-rocjpeg-tracing-validate
    COMMAND
        ${Python3_EXECUTABLE} ${CMAKE_CURRENT_BINARY_DIR}/validate.py --json-input
        ${CMAKE_CURRENT_BINARY_DIR}/rocjpeg-trace/out_results.json --otf2-input
        ${CMAKE_CURRENT_BINARY_DIR}/rocjpeg-trace/out_results.otf2 --pftrace-input
        ${CMAKE_CURRENT_BINARY_DIR}/rocjpeg-trace/out_results.pftrace --csv-input
        ${CMAKE_CURRENT_BINARY_DIR}/rocjpeg-trace/out_rocjpeg_api_trace.csv)

set_tests_properties(
    rocprofv3-test-rocjpeg-tracing-validate
    PROPERTIES TIMEOUT
               45
               LABELS
               "integration-tests"
               DEPENDS
               rocprofv3-test-rocjpeg-tracing-execute
               FAIL_REGULAR_EXPRESSION
               "AssertionError"
               DISABLED
               "${IS_DISABLED}")
