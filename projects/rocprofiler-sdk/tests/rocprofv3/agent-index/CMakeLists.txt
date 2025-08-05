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
    rocprofiler-sdk-tests-rocprofv3-agent-index
    LANGUAGES CXX
    VERSION 0.0.0)

find_package(rocprofiler-sdk REQUIRED)

string(REPLACE "LD_PRELOAD=" "ROCPROF_PRELOAD=" PRELOAD_ENV
               "${ROCPROFILER_MEMCHECK_PRELOAD_ENV}")

set(tracing-env "${PRELOAD_ENV}")

rocprofiler_configure_pytest_files(
    CONFIG pytest.ini
    COPY validate.py conftest.py input_absolute_agent_index.yaml
         input_type_relative_agent_index.yaml input_relative_agent_index.yaml)
if(ROCPROFILER_MEMCHECK STREQUAL "LeakSanitizer")
    set(LOG_LEVEL "warning") # info produces memory leak
else()
    set(LOG_LEVEL "info")
endif()

set(execute_tests
    rocprofv3-test-agent-index-json-absolute-execute
    rocprofv3-test-agent-index-json-type-relative-execute
    rocprofv3-test-agent-index-json-relative-execute)
set(input_files
    ${CMAKE_CURRENT_BINARY_DIR}/input_absolute_agent_index.yaml
    ${CMAKE_CURRENT_BINARY_DIR}/input_type_relative_agent_index.yaml
    ${CMAKE_CURRENT_BINARY_DIR}/input_relative_agent_index.yaml)
set(output_dir
    ${CMAKE_CURRENT_BINARY_DIR}/%argt%-agent-index-absolute
    ${CMAKE_CURRENT_BINARY_DIR}/%argt%-agent-index-type-relative
    ${CMAKE_CURRENT_BINARY_DIR}/%argt%-agent-index-relative)

list(LENGTH execute_tests execute_test_length)
foreach(i RANGE 0 ${execute_test_length})
    if(${i} EQUAL ${execute_test_length})
        break()
    endif()
    list(GET execute_tests ${i} test_name)
    list(GET input_files ${i} input_file)
    list(GET output_dir ${i} output)
    add_test(NAME ${test_name}
             COMMAND $<TARGET_FILE:rocprofiler-sdk::rocprofv3> -i ${input_file} -d
                     ${output} --log-level ${LOG_LEVEL} -- $<TARGET_FILE:transpose>)
    set_tests_properties(
        ${test_name}
        PROPERTIES TIMEOUT 45 LABELS "integration-tests" ENVIRONMENT "${tracing-env}"
                   FAIL_REGULAR_EXPRESSION "${ROCPROFILER_DEFAULT_FAIL_REGEX}")
endforeach()

set(agent_index_types absolute relative type-relative)
set(validate_tests
    rocprofv3-test-agent-index-json-absolute-validate
    rocprofv3-test-agent-index-json-relative-validate
    rocprofv3-test-agent-index-json-type-relative-validate)
set(output_dir_name transpose-agent-index-absolute transpose-agent-index-relative
                    transpose-agent-index-type-relative)

list(LENGTH validate_tests validate_test_length)
foreach(i RANGE 0 ${validate_test_length})
    if(${i} EQUAL ${validate_test_length})
        break()
    endif()
    list(GET validate_tests ${i} test_name)
    list(GET agent_index_types ${i} agent_index)
    list(GET output_dir_name ${i} output)
    add_test(
        NAME ${test_name}
        COMMAND
            ${Python3_EXECUTABLE} ${CMAKE_CURRENT_BINARY_DIR}/validate.py --agent-index
            ${agent_index} --csv-kernel-input
            ${CMAKE_CURRENT_BINARY_DIR}/${output}/out_kernel_trace.csv
            --csv-memory-allocation-input
            ${CMAKE_CURRENT_BINARY_DIR}/${output}/out_memory_allocation_trace.csv
            --csv-memory-copy-input
            ${CMAKE_CURRENT_BINARY_DIR}/${output}/out_memory_copy_trace.csv --json-input
            ${CMAKE_CURRENT_BINARY_DIR}/transpose-agent-index-absolute/out_results.json)
endforeach()

set(dependency_tests
    rocprofv3-test-agent-index-json-absolute-execute
    rocprofv3-test-agent-index-json-relative-execute;rocprofv3-test-agent-index-json-absolute-execute
    rocprofv3-test-agent-index-json-type-relative-execute;rocprofv3-test-agent-index-json-absolute-execute
    )

foreach(i RANGE 0 ${validate_test_length})
    if(${i} EQUAL ${validate_test_length})
        break()
    endif()
    list(GET validate_tests ${i} test_name)
    list(GET dependency_tests ${i} dep)
    set_tests_properties(
        ${test_name} PROPERTIES TIMEOUT 45 LABELS "integration-tests" DEPENDS ${dep}
                                FAIL_REGULAR_EXPRESSION "AssertionError")
endforeach()
