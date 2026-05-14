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

string(REPLACE "LD_PRELOAD=" "ROCPROF_PRELOAD=" PRELOAD_ENV
               "${ROCPROFILER_MEMCHECK_PRELOAD_ENV}")

# To ensure we sample all agents, use 16 threads each assigned to one agent in a round
# robin manner. To keep the job per device reasonable, each thread offloads 100 instances
# of a transpose kernel.
set(TRANSPOSE_NUM_THREADS 16)
set(TRANSPOSE_NUM_ITERATIONS 100)

rocprofiler_add_integration_execute_test(
    rocprofv3-test-pc-sampling-stochastic-transpose-multiple-agents-input-cmd
    COMMAND
        $<TARGET_FILE:rocprofiler-sdk::rocprofv3> --kernel-trace --pc-sampling-unit cycles
        --pc-sampling-method stochastic --pc-sampling-interval 1048576 -d
        ${CMAKE_CURRENT_BINARY_DIR}/pc_sampling_cmd_input -o out --output-format csv json
        -- $<TARGET_FILE:transpose> ${TRANSPOSE_NUM_THREADS} ${TRANSPOSE_NUM_ITERATIONS}
    DEPENDS transpose
    TIMEOUT 90
    LABELS "integration-tests;pc-sampling;stochastic"
    ENVIRONMENT "${PRELOAD_ENV}"
    SKIP_REGULAR_EXPRESSION "${ROCPROFV3_TESTS_PC_SAMPLING_SKIP_REGEX}"
    FIXTURES_SETUP
        rocprofv3-test-pc-sampling-stochastic-transpose-multiple-agents-input-cmd
    DISABLED "${IS_PC_SAMPLING_STOCHASTIC_DISABLED}")

rocprofiler_add_integration_validate_test(
    rocprofv3-test-pc-sampling-stochastic-transpose-multiple-agents-input-cmd
    TEST_PATHS validate.py
    COPY conftest.py input.json input.yml
    CONFIG pytest.ini
    ARGS -k
         test_
         --input-samples-csv
         ${CMAKE_CURRENT_BINARY_DIR}/pc_sampling_cmd_input/out_pc_sampling_stochastic.csv
         --input-samples-json
         ${CMAKE_CURRENT_BINARY_DIR}/pc_sampling_cmd_input/out_results.json
         --input-kernel-trace-csv
         ${CMAKE_CURRENT_BINARY_DIR}/pc_sampling_cmd_input/out_kernel_trace.csv
         --input-agent-info-csv
         ${CMAKE_CURRENT_BINARY_DIR}/pc_sampling_cmd_input/out_agent_info.csv
    TIMEOUT 60
    LABELS "integration-tests;pc-sampling;stochastic"
    FIXTURES_REQUIRED
        rocprofv3-test-pc-sampling-stochastic-transpose-multiple-agents-input-cmd
    SKIP_REGULAR_EXPRESSION "${ROCPROFV3_TESTS_PC_SAMPLING_SKIP_REGEX}"
    DISABLED "${IS_PC_SAMPLING_STOCHASTIC_DISABLED}")

rocprofiler_add_integration_execute_test(
    rocprofv3-test-pc-sampling-stochastic-transpose-multiple-agents-input-json
    COMMAND
        $<TARGET_FILE:rocprofiler-sdk::rocprofv3> -i
        ${CMAKE_CURRENT_BINARY_DIR}/input.json -d
        ${CMAKE_CURRENT_BINARY_DIR}/pc_sampling_json_input -- $<TARGET_FILE:transpose>
        ${TRANSPOSE_NUM_THREADS} ${TRANSPOSE_NUM_ITERATIONS}
    DEPENDS transpose
    TIMEOUT 90
    LABELS "integration-tests;pc-sampling;stochastic"
    ENVIRONMENT "${PRELOAD_ENV}"
    SKIP_REGULAR_EXPRESSION "${ROCPROFV3_TESTS_PC_SAMPLING_SKIP_REGEX}"
    FIXTURES_SETUP
        rocprofv3-test-pc-sampling-stochastic-transpose-multiple-agents-input-json
    DISABLED "${IS_PC_SAMPLING_STOCHASTIC_DISABLED}")

rocprofiler_add_integration_validate_test(
    rocprofv3-test-pc-sampling-stochastic-transpose-multiple-agents-input-json
    TEST_PATHS validate.py
    COPY conftest.py input.json input.yml
    CONFIG pytest.ini
    ARGS -k
         test_
         --input-samples-csv
         ${CMAKE_CURRENT_BINARY_DIR}/pc_sampling_json_input/out_pc_sampling_stochastic.csv
         --input-samples-json
         ${CMAKE_CURRENT_BINARY_DIR}/pc_sampling_json_input/out_results.json
         --input-kernel-trace-csv
         ${CMAKE_CURRENT_BINARY_DIR}/pc_sampling_json_input/out_kernel_trace.csv
         --input-agent-info-csv
         ${CMAKE_CURRENT_BINARY_DIR}/pc_sampling_json_input/out_agent_info.csv
    TIMEOUT 60
    LABELS "integration-tests;pc-sampling;stochastic"
    FIXTURES_REQUIRED
        rocprofv3-test-pc-sampling-stochastic-transpose-multiple-agents-input-json
    SKIP_REGULAR_EXPRESSION "${ROCPROFV3_TESTS_PC_SAMPLING_SKIP_REGEX}"
    DISABLED "${IS_PC_SAMPLING_STOCHASTIC_DISABLED}")

rocprofiler_add_integration_execute_test(
    rocprofv3-test-pc-sampling-stochastic-transpose-multiple-agents-input-yaml
    COMMAND
        $<TARGET_FILE:rocprofiler-sdk::rocprofv3> -i ${CMAKE_CURRENT_BINARY_DIR}/input.yml
        -d ${CMAKE_CURRENT_BINARY_DIR}/pc_sampling_yaml_input -- $<TARGET_FILE:transpose>
        ${TRANSPOSE_NUM_THREADS} ${TRANSPOSE_NUM_ITERATIONS}
    DEPENDS transpose
    TIMEOUT 90
    LABELS "integration-tests;pc-sampling;stochastic"
    ENVIRONMENT "${PRELOAD_ENV}"
    SKIP_REGULAR_EXPRESSION "${ROCPROFV3_TESTS_PC_SAMPLING_SKIP_REGEX}"
    FIXTURES_SETUP
        rocprofv3-test-pc-sampling-stochastic-transpose-multiple-agents-input-yaml
    DISABLED "${IS_PC_SAMPLING_STOCHASTIC_DISABLED}")

rocprofiler_add_integration_validate_test(
    rocprofv3-test-pc-sampling-stochastic-transpose-multiple-agents-input-yaml
    TEST_PATHS validate.py
    COPY conftest.py input.json input.yml
    CONFIG pytest.ini
    ARGS -k
         test_
         --input-samples-csv
         ${CMAKE_CURRENT_BINARY_DIR}/pc_sampling_yaml_input/out_pc_sampling_stochastic.csv
         --input-samples-json
         ${CMAKE_CURRENT_BINARY_DIR}/pc_sampling_yaml_input/out_results.json
         --input-kernel-trace-csv
         ${CMAKE_CURRENT_BINARY_DIR}/pc_sampling_yaml_input/out_kernel_trace.csv
         --input-agent-info-csv
         ${CMAKE_CURRENT_BINARY_DIR}/pc_sampling_yaml_input/out_agent_info.csv
    TIMEOUT 60
    LABELS "integration-tests;pc-sampling;stochastic"
    FIXTURES_REQUIRED
        rocprofv3-test-pc-sampling-stochastic-transpose-multiple-agents-input-yaml
    SKIP_REGULAR_EXPRESSION "${ROCPROFV3_TESTS_PC_SAMPLING_SKIP_REGEX}"
    DISABLED "${IS_PC_SAMPLING_STOCHASTIC_DISABLED}")
