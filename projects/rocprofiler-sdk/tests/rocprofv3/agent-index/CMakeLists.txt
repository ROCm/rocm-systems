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

if(ROCPROFILER_MEMCHECK STREQUAL "LeakSanitizer")
    set(LOG_LEVEL "warning") # info produces memory leak
else()
    set(LOG_LEVEL "info")
endif()

rocprofiler_add_integration_execute_test(
    rocprofv3-test-agent-index-json-absolute
    COMMAND
        $<TARGET_FILE:rocprofiler-sdk::rocprofv3> -i
        ${CMAKE_CURRENT_BINARY_DIR}/input_absolute_agent_index.yaml -d
        ${CMAKE_CURRENT_BINARY_DIR}/%argt%-agent-index-absolute --log-level ${LOG_LEVEL}
        -- $<TARGET_FILE:transpose>
    DEPENDS transpose
    TIMEOUT 45
    LABELS "integration-tests"
    PRELOAD "${PRELOAD_ENV}"
    FIXTURES_SETUP rocprofv3-test-agent-index-json-absolute)

rocprofiler_add_integration_execute_test(
    rocprofv3-test-agent-index-json-type-relative
    COMMAND
        $<TARGET_FILE:rocprofiler-sdk::rocprofv3> -i
        ${CMAKE_CURRENT_BINARY_DIR}/input_type_relative_agent_index.yaml -d
        ${CMAKE_CURRENT_BINARY_DIR}/%argt%-agent-index-type-relative --log-level
        ${LOG_LEVEL} -- $<TARGET_FILE:transpose>
    DEPENDS transpose
    TIMEOUT 45
    LABELS "integration-tests"
    PRELOAD "${PRELOAD_ENV}"
    FIXTURES_SETUP rocprofv3-test-agent-index-json-type-relative)

rocprofiler_add_integration_execute_test(
    rocprofv3-test-agent-index-json-relative
    COMMAND
        $<TARGET_FILE:rocprofiler-sdk::rocprofv3> -i
        ${CMAKE_CURRENT_BINARY_DIR}/input_relative_agent_index.yaml -d
        ${CMAKE_CURRENT_BINARY_DIR}/%argt%-agent-index-relative --log-level ${LOG_LEVEL}
        -- $<TARGET_FILE:transpose>
    DEPENDS transpose
    TIMEOUT 45
    LABELS "integration-tests"
    PRELOAD "${PRELOAD_ENV}"
    FIXTURES_SETUP rocprofv3-test-agent-index-json-relative)

rocprofiler_add_integration_validate_test(
    rocprofv3-test-agent-index-json-absolute
    TEST_PATHS validate.py
    COPY conftest.py input_absolute_agent_index.yaml input_type_relative_agent_index.yaml
         input_relative_agent_index.yaml
    CONFIG pytest.ini
    TIMEOUT 45
    LABELS "integration-tests"
    FIXTURES_REQUIRED rocprofv3-test-agent-index-json-absolute
    ARGS --agent-index
         absolute
         --csv-kernel-input
         ${CMAKE_CURRENT_BINARY_DIR}/transpose-agent-index-absolute/out_kernel_trace.csv
         --csv-memory-allocation-input
         ${CMAKE_CURRENT_BINARY_DIR}/transpose-agent-index-absolute/out_memory_allocation_trace.csv
         --csv-memory-copy-input
         ${CMAKE_CURRENT_BINARY_DIR}/transpose-agent-index-absolute/out_memory_copy_trace.csv
         --json-input
         ${CMAKE_CURRENT_BINARY_DIR}/transpose-agent-index-absolute/out_results.json
    FAIL_REGULAR_EXPRESSION "AssertionError")

rocprofiler_add_integration_validate_test(
    rocprofv3-test-agent-index-json-relative
    TEST_PATHS validate.py
    COPY conftest.py input_absolute_agent_index.yaml input_type_relative_agent_index.yaml
         input_relative_agent_index.yaml
    CONFIG pytest.ini
    TIMEOUT 45
    LABELS "integration-tests"
    FIXTURES_REQUIRED rocprofv3-test-agent-index-json-relative
                      rocprofv3-test-agent-index-json-absolute
    ARGS --agent-index
         relative
         --csv-kernel-input
         ${CMAKE_CURRENT_BINARY_DIR}/transpose-agent-index-relative/out_kernel_trace.csv
         --csv-memory-allocation-input
         ${CMAKE_CURRENT_BINARY_DIR}/transpose-agent-index-relative/out_memory_allocation_trace.csv
         --csv-memory-copy-input
         ${CMAKE_CURRENT_BINARY_DIR}/transpose-agent-index-relative/out_memory_copy_trace.csv
         --json-input
         ${CMAKE_CURRENT_BINARY_DIR}/transpose-agent-index-absolute/out_results.json
    FAIL_REGULAR_EXPRESSION "AssertionError")

rocprofiler_add_integration_validate_test(
    rocprofv3-test-agent-index-json-type-relative
    TEST_PATHS validate.py
    COPY conftest.py input_absolute_agent_index.yaml input_type_relative_agent_index.yaml
         input_relative_agent_index.yaml
    CONFIG pytest.ini
    TIMEOUT 45
    LABELS "integration-tests"
    FIXTURES_REQUIRED rocprofv3-test-agent-index-json-type-relative
                      rocprofv3-test-agent-index-json-absolute
    ARGS --agent-index
         type-relative
         --csv-kernel-input
         ${CMAKE_CURRENT_BINARY_DIR}/transpose-agent-index-type-relative/out_kernel_trace.csv
         --csv-memory-allocation-input
         ${CMAKE_CURRENT_BINARY_DIR}/transpose-agent-index-type-relative/out_memory_allocation_trace.csv
         --csv-memory-copy-input
         ${CMAKE_CURRENT_BINARY_DIR}/transpose-agent-index-type-relative/out_memory_copy_trace.csv
         --json-input
         ${CMAKE_CURRENT_BINARY_DIR}/transpose-agent-index-absolute/out_results.json
    FAIL_REGULAR_EXPRESSION "AssertionError")
