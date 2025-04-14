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
    rocprofiler-sdk-tests-conversion-script
    LANGUAGES CXX
    VERSION 0.0.0)

find_package(rocprofiler-sdk REQUIRED)
# copy to binary directory
rocprofiler_configure_pytest_files(COPY validate.py input.txt conftest.py
                                   CONFIG pytest.ini)

add_test(
    NAME test-conversion-script-execute
    COMMAND
        $<TARGET_FILE:rocprofiler-sdk::rocprofv3> -i
        ${CMAKE_CURRENT_BINARY_DIR}/input.txt -T -d
        ${CMAKE_CURRENT_BINARY_DIR}/out_conversion_script -o pmc1 --output-format csv --
        $<TARGET_FILE:vector-ops>)

string(REPLACE "LD_PRELOAD=" "ROCPROF_PRELOAD=" PRELOAD_ENV
               "${ROCPROFILER_MEMCHECK_PRELOAD_ENV}")

set(cc-env-pmc "${PRELOAD_ENV}")

set_tests_properties(
    test-conversion-script-execute
    PROPERTIES TIMEOUT 45 LABELS "integration-tests" ENVIRONMENT "${cc-env-pmc}"
               FAIL_REGULAR_EXPRESSION "${ROCPROFILER_DEFAULT_FAIL_REGEX}")

add_test(
    NAME test-conversion-script-convert
    COMMAND
        ${Python3_EXECUTABLE}
        $<TARGET_FILE:rocprofiler-sdk::convert-counters-collection-format> --input
        ${CMAKE_CURRENT_BINARY_DIR}/out_conversion_script --output
        ${CMAKE_CURRENT_BINARY_DIR}/out_conversion_script/converted.csv)

set_tests_properties(
    test-conversion-script-convert
    PROPERTIES TIMEOUT 45 LABELS "integration-tests" DEPENDS
               test-conversion-script-execute FAIL_REGULAR_EXPRESSION
               "${ROCPROFILER_DEFAULT_FAIL_REGEX}")

add_test(NAME test-conversion-script-validate
         COMMAND ${Python3_EXECUTABLE} ${CMAKE_CURRENT_BINARY_DIR}/validate.py --input
                 ${CMAKE_CURRENT_BINARY_DIR}/out_conversion_script/converted.csv)

set_tests_properties(
    test-conversion-script-validate
    PROPERTIES TIMEOUT 45 LABELS "integration-tests" DEPENDS
               test-conversion-script-convert FAIL_REGULAR_EXPRESSION
               "${ROCPROFILER_DEFAULT_FAIL_REGEX}")
