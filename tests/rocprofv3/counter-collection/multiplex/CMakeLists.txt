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
# rocprofv3 tool test
#
cmake_minimum_required(VERSION 3.21.0 FATAL_ERROR)

project(
    rocprofiler-sdk-tests-counter-collection-multiplex
    LANGUAGES CXX
    VERSION 0.0.0)

find_package(rocprofiler-sdk REQUIRED)

rocprofiler_configure_pytest_files(CONFIG pytest.ini COPY validate.py conftest.py
                                                          input.json input.yml)

# pmc1
add_test(
    NAME rocprofv3-test-counter-collection-multiplex-execute
    COMMAND
        $<TARGET_FILE:rocprofiler-sdk::rocprofv3> -i
        ${CMAKE_CURRENT_BINARY_DIR}/input.json -d ${CMAKE_CURRENT_BINARY_DIR}/%argt%-cc
        -o out_json -f csv -- $<TARGET_FILE:simple-transpose>)

add_test(
    NAME rocprofv3-test-counter-collection-multiple-yaml-execute
    COMMAND
        $<TARGET_FILE:rocprofiler-sdk::rocprofv3> -i
        ${CMAKE_CURRENT_BINARY_DIR}/input.yml -d ${CMAKE_CURRENT_BINARY_DIR}/%argt%-cc -o
        out_yaml -f csv -- $<TARGET_FILE:simple-transpose>)

string(REPLACE "LD_PRELOAD=" "ROCPROF_PRELOAD=" PRELOAD_ENV
               "${ROCPROFILER_MEMCHECK_PRELOAD_ENV}")

set(cc-env-pmc1 "${PRELOAD_ENV}")

set_tests_properties(
    rocprofv3-test-counter-collection-multiplex-execute
    rocprofv3-test-counter-collection-multiple-yaml-execute
    PROPERTIES TIMEOUT 45 LABELS "integration-tests" ENVIRONMENT "${cc-env-pmc1}"
               FAIL_REGULAR_EXPRESSION "${ROCPROFILER_DEFAULT_FAIL_REGEX}")

add_test(
    NAME rocprofv3-test-counter-collection-multiplex-validate
    COMMAND
        ${Python3_EXECUTABLE} ${CMAKE_CURRENT_BINARY_DIR}/validate.py --agent-input
        ${CMAKE_CURRENT_BINARY_DIR}/simple-transpose-cc/out_json_agent_info.csv
        --counter-input
        ${CMAKE_CURRENT_BINARY_DIR}/simple-transpose-cc/out_json_counter_collection.csv)

set_tests_properties(
    rocprofv3-test-counter-collection-multiplex-validate
    PROPERTIES TIMEOUT 45 LABELS "integration-tests" DEPENDS
               rocprofv3-test-counter-collection-multiplex-execute
               FAIL_REGULAR_EXPRESSION "${ROCPROFILER_DEFAULT_FAIL_REGEX}")

add_test(
    NAME rocprofv3-test-counter-collection-multiple-yaml-validate
    COMMAND
        ${Python3_EXECUTABLE} ${CMAKE_CURRENT_BINARY_DIR}/validate.py --agent-input
        ${CMAKE_CURRENT_BINARY_DIR}/simple-transpose-cc/out_yaml_agent_info.csv
        --counter-input
        ${CMAKE_CURRENT_BINARY_DIR}/simple-transpose-cc/out_yaml_counter_collection.csv)

set_tests_properties(
    rocprofv3-test-counter-collection-multiple-yaml-validate
    PROPERTIES TIMEOUT 45 LABELS "integration-tests" DEPENDS
               rocprofv3-test-counter-collection-multiple-yaml-execute
               FAIL_REGULAR_EXPRESSION "${ROCPROFILER_DEFAULT_FAIL_REGEX}")
