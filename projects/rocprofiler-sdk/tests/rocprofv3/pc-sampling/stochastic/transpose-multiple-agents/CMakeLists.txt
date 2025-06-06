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
    rocprofiler-tests-pc-sampling
    LANGUAGES CXX
    VERSION 0.0.0)

find_package(rocprofiler-sdk REQUIRED)

# Checking whether stochastic PC sampling is disabled. If so, skip the tests.
rocprofiler_sdk_pc_sampling_stochastic_disabled(IS_PC_SAMPLING_STOCHASTIC_DISABLED)

rocprofiler_configure_pytest_files(CONFIG pytest.ini COPY validate.py conftest.py
                                                          input.json input.yml)

# To ensure we sample all agents, use 16 threads each assigned to one agent in a round
# robin manner. To keep the job per device reasonable, each thread offloads 100 instances
# of a transpose kernel.
set(TRANSPOSE_NUM_THREADS 16)
set(TRANSPOSE_NUM_ITERATIONS 100)

add_test(
    NAME rocprofv3-test-pc-sampling-stochastic-transpose-multiple-agents-input-cmd-execute
    COMMAND
        $<TARGET_FILE:rocprofiler-sdk::rocprofv3> --kernel-trace --pc-sampling-unit
        cycles --pc-sampling-method stochastic --pc-sampling-interval 1048576 -d
        ${CMAKE_CURRENT_BINARY_DIR}/pc_sampling_cmd_input -o out --output-format csv json
        -- $<TARGET_FILE:transpose> ${TRANSPOSE_NUM_THREADS} ${TRANSPOSE_NUM_ITERATIONS})

string(REPLACE "LD_PRELOAD=" "ROCPROF_PRELOAD=" PRELOAD_ENV
               "${ROCPROFILER_MEMCHECK_PRELOAD_ENV}")

set(pc-sampling-env-stochastic-transpose-multiple-agents "${PRELOAD_ENV}")

set_tests_properties(
    rocprofv3-test-pc-sampling-stochastic-transpose-multiple-agents-input-cmd-execute
    PROPERTIES TIMEOUT
               45
               LABELS
               "integration-tests;pc-sampling;stochastic"
               ENVIRONMENT
               "${pc-sampling-env-stochastic-transpose-multiple-agents}"
               FAIL_REGULAR_EXPRESSION
               "${ROCPROFILER_DEFAULT_FAIL_REGEX}"
               SKIP_REGULAR_EXPRESSION
               "PC sampling unavailable"
               DISABLED
               "${IS_PC_SAMPLING_STOCHASTIC_DISABLED}")

add_test(
    NAME rocprofv3-test-pc-sampling-stochastic-transpose-multiple-agents-input-json-execute
    COMMAND
        $<TARGET_FILE:rocprofiler-sdk::rocprofv3> -i
        ${CMAKE_CURRENT_BINARY_DIR}/input.json -d
        ${CMAKE_CURRENT_BINARY_DIR}/pc_sampling_json_input -- $<TARGET_FILE:transpose>
        ${TRANSPOSE_NUM_THREADS} ${TRANSPOSE_NUM_ITERATIONS})

set_tests_properties(
    rocprofv3-test-pc-sampling-stochastic-transpose-multiple-agents-input-json-execute
    PROPERTIES TIMEOUT
               45
               LABELS
               "integration-tests;pc-sampling;stochastic"
               ENVIRONMENT
               "${pc-sampling-env-stochastic-transpose-multiple-agents}"
               FAIL_REGULAR_EXPRESSION
               "${ROCPROFILER_DEFAULT_FAIL_REGEX}"
               SKIP_REGULAR_EXPRESSION
               "PC sampling unavailable"
               DISABLED
               "${IS_PC_SAMPLING_STOCHASTIC_DISABLED}")

add_test(
    NAME rocprofv3-test-pc-sampling-stochastic-transpose-multiple-agents-input-yaml-execute
    COMMAND
        $<TARGET_FILE:rocprofiler-sdk::rocprofv3> -i
        ${CMAKE_CURRENT_BINARY_DIR}/input.yml -d
        ${CMAKE_CURRENT_BINARY_DIR}/pc_sampling_yaml_input -- $<TARGET_FILE:transpose>
        ${TRANSPOSE_NUM_THREADS} ${TRANSPOSE_NUM_ITERATIONS})

set_tests_properties(
    rocprofv3-test-pc-sampling-stochastic-transpose-multiple-agents-input-yaml-execute
    PROPERTIES TIMEOUT
               45
               LABELS
               "integration-tests;pc-sampling;stochastic"
               ENVIRONMENT
               "${pc-sampling-env-stochastic-transpose-multiple-agents}"
               FAIL_REGULAR_EXPRESSION
               "${ROCPROFILER_DEFAULT_FAIL_REGEX}"
               SKIP_REGULAR_EXPRESSION
               "PC sampling unavailable"
               DISABLED
               "${IS_PC_SAMPLING_STOCHASTIC_DISABLED}")

# ========================= Validation tests

add_test(
    NAME rocprofv3-test-pc-sampling-stochastic-transpose-multiple-agents-input-cmd-validate
    COMMAND
        ${Python3_EXECUTABLE} ${CMAKE_CURRENT_BINARY_DIR}/validate.py -k test_
        --input-samples-csv
        ${CMAKE_CURRENT_BINARY_DIR}/pc_sampling_cmd_input/out_pc_sampling_stochastic.csv
        --input-samples-json
        ${CMAKE_CURRENT_BINARY_DIR}/pc_sampling_cmd_input/out_results.json
        --input-kernel-trace-csv
        ${CMAKE_CURRENT_BINARY_DIR}/pc_sampling_cmd_input/out_kernel_trace.csv
        --input-agent-info-csv
        ${CMAKE_CURRENT_BINARY_DIR}/pc_sampling_cmd_input/out_agent_info.csv)

set_tests_properties(
    rocprofv3-test-pc-sampling-stochastic-transpose-multiple-agents-input-cmd-validate
    PROPERTIES
        TIMEOUT
        60
        LABELS
        "integration-tests;pc-sampling;stochastic"
        DEPENDS
        "rocprofv3-test-pc-sampling-stochastic-transpose-multiple-agents-input-cmd-execute"
        FAIL_REGULAR_EXPRESSION
        "${ROCPROFILER_DEFAULT_FAIL_REGEX}"
        SKIP_REGULAR_EXPRESSION
        "PC sampling unavailable"
        DISABLED
        "${IS_PC_SAMPLING_STOCHASTIC_DISABLED}")

add_test(
    NAME rocprofv3-test-pc-sampling-stochastic-transpose-multiple-agents-input-json-validate
    COMMAND
        ${Python3_EXECUTABLE} ${CMAKE_CURRENT_BINARY_DIR}/validate.py -k test_
        --input-samples-csv
        ${CMAKE_CURRENT_BINARY_DIR}/pc_sampling_json_input/out_pc_sampling_stochastic.csv
        --input-samples-json
        ${CMAKE_CURRENT_BINARY_DIR}/pc_sampling_json_input/out_results.json
        --input-kernel-trace-csv
        ${CMAKE_CURRENT_BINARY_DIR}/pc_sampling_json_input/out_kernel_trace.csv
        --input-agent-info-csv
        ${CMAKE_CURRENT_BINARY_DIR}/pc_sampling_json_input/out_agent_info.csv)

set_tests_properties(
    rocprofv3-test-pc-sampling-stochastic-transpose-multiple-agents-input-json-validate
    PROPERTIES
        TIMEOUT
        60
        LABELS
        "integration-tests;pc-sampling;stochastic"
        DEPENDS
        "rocprofv3-test-pc-sampling-stochastic-transpose-multiple-agents-input-json-execute"
        FAIL_REGULAR_EXPRESSION
        "${ROCPROFILER_DEFAULT_FAIL_REGEX}"
        SKIP_REGULAR_EXPRESSION
        "PC sampling unavailable"
        DISABLED
        "${IS_PC_SAMPLING_STOCHASTIC_DISABLED}")

add_test(
    NAME rocprofv3-test-pc-sampling-stochastic-transpose-multiple-agents-input-yaml-validate
    COMMAND
        ${Python3_EXECUTABLE} ${CMAKE_CURRENT_BINARY_DIR}/validate.py -k test_
        --input-samples-csv
        ${CMAKE_CURRENT_BINARY_DIR}/pc_sampling_yaml_input/out_pc_sampling_stochastic.csv
        --input-samples-json
        ${CMAKE_CURRENT_BINARY_DIR}/pc_sampling_yaml_input/out_results.json
        --input-kernel-trace-csv
        ${CMAKE_CURRENT_BINARY_DIR}/pc_sampling_yaml_input/out_kernel_trace.csv
        --input-agent-info-csv
        ${CMAKE_CURRENT_BINARY_DIR}/pc_sampling_yaml_input/out_agent_info.csv)

set_tests_properties(
    rocprofv3-test-pc-sampling-stochastic-transpose-multiple-agents-input-yaml-validate
    PROPERTIES
        TIMEOUT
        60
        LABELS
        "integration-tests;pc-sampling;stochastic"
        DEPENDS
        "rocprofv3-test-pc-sampling-stochastic-transpose-multiple-agents-input-yaml-execute"
        FAIL_REGULAR_EXPRESSION
        "${ROCPROFILER_DEFAULT_FAIL_REGEX}"
        SKIP_REGULAR_EXPRESSION
        "PC sampling unavailable"
        DISABLED
        "${IS_PC_SAMPLING_STOCHASTIC_DISABLED}")
