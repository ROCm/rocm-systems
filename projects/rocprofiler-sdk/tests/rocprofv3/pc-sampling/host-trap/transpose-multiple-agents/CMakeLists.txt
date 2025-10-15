#
# rocprofv3 tool test
#
cmake_minimum_required(VERSION 3.21.0 FATAL_ERROR)

project(
    rocprofiler-sdk-tests-pc-sampling-host-trap-transpose-multiple-agents
    LANGUAGES CXX
    VERSION 0.0.0)

find_package(rocprofiler-sdk REQUIRED)

# Check if PC sampling is disabled and whether we should skip the tests.
rocprofiler_sdk_pc_sampling_disabled(IS_PC_SAMPLING_DISABLED)

string(REPLACE "LD_PRELOAD=" "ROCPROF_PRELOAD=" PRELOAD_ENV
               "${ROCPROFILER_MEMCHECK_PRELOAD_ENV}")

# To ensure we sample all agents, use 16 threads each assigned to one agent in a round
# robin manner. To keep the job per device reasonable, each thread offloads 100 instances
# of a transpose kernel.
set(TRANSPOSE_NUM_THREADS 16)
set(TRANSPOSE_NUM_ITERATIONS 100)

rocprofiler_add_integration_execute_test(
    rocprofv3-test-pc-sampling-host-trap-transpose-multiple-agents-input-cmd
    COMMAND
        $<TARGET_FILE:rocprofiler-sdk::rocprofv3> --kernel-trace --pc-sampling-unit time
        --pc-sampling-method host_trap --pc-sampling-interval 1 --agent-index absolute -d
        ${CMAKE_CURRENT_BINARY_DIR}/pc_sampling_cmd_input -o out --output-format csv --
        $<TARGET_FILE:transpose> ${TRANSPOSE_NUM_THREADS} ${TRANSPOSE_NUM_ITERATIONS}
    DEPENDS transpose
    TIMEOUT 90
    LABELS "integration-tests;pc-sampling"
    PRELOAD "${PRELOAD_ENV}"
    SKIP_REGULAR_EXPRESSION "${ROCPROFV3_TESTS_PC_SAMPLING_SKIP_REGEX}"
    FIXTURES_SETUP
        rocprofv3-test-pc-sampling-host-trap-transpose-multiple-agents-input-cmd
    DISABLED "${IS_PC_SAMPLING_DISABLED}")

rocprofiler_add_integration_validate_test(
    rocprofv3-test-pc-sampling-host-trap-transpose-multiple-agents-input-cmd
    TEST_PATHS validate.py
    COPY conftest.py input.json input.yml
    CONFIG pytest.ini
    ARGS -k
         test_multi_agent_support
         --input-samples-csv
         ${CMAKE_CURRENT_BINARY_DIR}/pc_sampling_cmd_input/out_pc_sampling_host_trap.csv
         --input-kernel-trace-csv
         ${CMAKE_CURRENT_BINARY_DIR}/pc_sampling_cmd_input/out_kernel_trace.csv
         --input-agent-info-csv
         ${CMAKE_CURRENT_BINARY_DIR}/pc_sampling_cmd_input/out_agent_info.csv
    TIMEOUT 60
    LABELS "integration-tests;pc-sampling"
    FIXTURES_REQUIRED
        rocprofv3-test-pc-sampling-host-trap-transpose-multiple-agents-input-cmd
    SKIP_REGULAR_EXPRESSION "${ROCPROFV3_TESTS_PC_SAMPLING_SKIP_REGEX}"
    DISABLED "${IS_PC_SAMPLING_DISABLED}")

rocprofiler_add_integration_execute_test(
    rocprofv3-test-pc-sampling-host-trap-transpose-multiple-agents-input-json
    COMMAND
        $<TARGET_FILE:rocprofiler-sdk::rocprofv3> -i
        ${CMAKE_CURRENT_BINARY_DIR}/input.json -d
        ${CMAKE_CURRENT_BINARY_DIR}/pc_sampling_json_input -- $<TARGET_FILE:transpose>
        ${TRANSPOSE_NUM_THREADS} ${TRANSPOSE_NUM_ITERATIONS}
    DEPENDS transpose
    TIMEOUT 90
    LABELS "integration-tests;pc-sampling"
    PRELOAD "${PRELOAD_ENV}"
    SKIP_REGULAR_EXPRESSION "${ROCPROFV3_TESTS_PC_SAMPLING_SKIP_REGEX}"
    FIXTURES_SETUP
        rocprofv3-test-pc-sampling-host-trap-transpose-multiple-agents-input-json
    DISABLED "${IS_PC_SAMPLING_DISABLED}")

rocprofiler_add_integration_validate_test(
    rocprofv3-test-pc-sampling-host-trap-transpose-multiple-agents-input-json
    TEST_PATHS validate.py
    COPY conftest.py input.json input.yml
    CONFIG pytest.ini
    ARGS -k
         test_multi_agent_support
         --input-samples-csv
         ${CMAKE_CURRENT_BINARY_DIR}/pc_sampling_json_input/out_pc_sampling_host_trap.csv
         --input-kernel-trace-csv
         ${CMAKE_CURRENT_BINARY_DIR}/pc_sampling_json_input/out_kernel_trace.csv
         --input-agent-info-csv
         ${CMAKE_CURRENT_BINARY_DIR}/pc_sampling_json_input/out_agent_info.csv
    TIMEOUT 60
    LABELS "integration-tests;pc-sampling"
    FIXTURES_REQUIRED
        rocprofv3-test-pc-sampling-host-trap-transpose-multiple-agents-input-json
    SKIP_REGULAR_EXPRESSION "${ROCPROFV3_TESTS_PC_SAMPLING_SKIP_REGEX}"
    DISABLED "${IS_PC_SAMPLING_DISABLED}")

rocprofiler_add_integration_execute_test(
    rocprofv3-test-pc-sampling-host-trap-transpose-multiple-agents-input-yaml
    COMMAND
        $<TARGET_FILE:rocprofiler-sdk::rocprofv3> -i ${CMAKE_CURRENT_BINARY_DIR}/input.yml
        -d ${CMAKE_CURRENT_BINARY_DIR}/pc_sampling_yaml_input -o out --output-format csv
        -- $<TARGET_FILE:transpose> ${TRANSPOSE_NUM_THREADS} ${TRANSPOSE_NUM_ITERATIONS}
    DEPENDS transpose
    TIMEOUT 90
    LABELS "integration-tests;pc-sampling"
    PRELOAD "${PRELOAD_ENV}"
    SKIP_REGULAR_EXPRESSION "${ROCPROFV3_TESTS_PC_SAMPLING_SKIP_REGEX}"
    FIXTURES_SETUP
        rocprofv3-test-pc-sampling-host-trap-transpose-multiple-agents-input-yaml
    DISABLED "${IS_PC_SAMPLING_DISABLED}")

rocprofiler_add_integration_validate_test(
    rocprofv3-test-pc-sampling-host-trap-transpose-multiple-agents-input-yaml
    TEST_PATHS validate.py
    COPY conftest.py input.json input.yml
    CONFIG pytest.ini
    ARGS -k
         test_multi_agent_support
         --input-samples-csv
         ${CMAKE_CURRENT_BINARY_DIR}/pc_sampling_yaml_input/out_pc_sampling_host_trap.csv
         --input-kernel-trace-csv
         ${CMAKE_CURRENT_BINARY_DIR}/pc_sampling_yaml_input/out_kernel_trace.csv
         --input-agent-info-csv
         ${CMAKE_CURRENT_BINARY_DIR}/pc_sampling_yaml_input/out_agent_info.csv
    TIMEOUT 60
    LABELS "integration-tests;pc-sampling"
    FIXTURES_REQUIRED
        rocprofv3-test-pc-sampling-host-trap-transpose-multiple-agents-input-yaml
    SKIP_REGULAR_EXPRESSION "${ROCPROFV3_TESTS_PC_SAMPLING_SKIP_REGEX}"
    DISABLED "${IS_PC_SAMPLING_DISABLED}")
