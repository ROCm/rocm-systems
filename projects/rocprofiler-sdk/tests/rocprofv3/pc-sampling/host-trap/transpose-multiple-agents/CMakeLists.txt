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

rocprofiler_configure_pytest_files(CONFIG pytest.ini COPY validate.py conftest.py
                                                          input.json input.yml)

# To ensure we sample all agents, use 16 threads each assigned to one agent in a round
# robin manner. To keep the job per device reasonable, each thread offloads 100 instances
# of a transpose kernel.
set(TRANSPOSE_NUM_THREADS 16)
set(TRANSPOSE_NUM_ITERATIONS 100)

list(GET rocprofiler-sdk-tests-gfx-info 0 gpu-0-gfx-info)
if(gpu-0-gfx-info MATCHES "gfx120[0-9]")
    # reducing job for Navi4x GPUs
    set(TRANSPOSE_NUM_THREADS 4)
    set(TRANSPOSE_NUM_ITERATIONS 50)
endif()

add_test(
    NAME rocprofv3-test-pc-sampling-host-trap-transpose-multiple-agents-input-cmd-execute
    COMMAND
        $<TARGET_FILE:rocprofiler-sdk::rocprofv3> --kernel-trace --pc-sampling-unit time
        --pc-sampling-method host_trap --pc-sampling-interval 1 --agent-index absolute -d
        ${CMAKE_CURRENT_BINARY_DIR}/pc_sampling_cmd_input -o out --output-format csv --
        $<TARGET_FILE:transpose> ${TRANSPOSE_NUM_THREADS} ${TRANSPOSE_NUM_ITERATIONS})

string(REPLACE "LD_PRELOAD=" "ROCPROF_PRELOAD=" PRELOAD_ENV
               "${ROCPROFILER_MEMCHECK_PRELOAD_ENV}")

set(pc-sampling-env-host-trap-transpose-multiple-agents "${PRELOAD_ENV}")

set_tests_properties(
    rocprofv3-test-pc-sampling-host-trap-transpose-multiple-agents-input-cmd-execute
    PROPERTIES TIMEOUT
               90
               LABELS
               "integration-tests;pc-sampling"
               ENVIRONMENT
               "${pc-sampling-env-host-trap-transpose-multiple-agents}"
               FAIL_REGULAR_EXPRESSION
               "${ROCPROFILER_DEFAULT_FAIL_REGEX}"
               SKIP_REGULAR_EXPRESSION
               "${ROCPROFV3_TESTS_PC_SAMPLING_SKIP_REGEX}"
               DISABLED
               "${IS_PC_SAMPLING_DISABLED}")

add_test(
    NAME rocprofv3-test-pc-sampling-host-trap-transpose-multiple-agents-input-json-execute
    COMMAND
        $<TARGET_FILE:rocprofiler-sdk::rocprofv3> -i
        ${CMAKE_CURRENT_BINARY_DIR}/input.json -d
        ${CMAKE_CURRENT_BINARY_DIR}/pc_sampling_json_input -- $<TARGET_FILE:transpose>
        ${TRANSPOSE_NUM_THREADS} ${TRANSPOSE_NUM_ITERATIONS})

set_tests_properties(
    rocprofv3-test-pc-sampling-host-trap-transpose-multiple-agents-input-json-execute
    PROPERTIES TIMEOUT
               90
               LABELS
               "integration-tests;pc-sampling"
               ENVIRONMENT
               "${pc-sampling-env-host-trap-transpose-multiple-agents}"
               FAIL_REGULAR_EXPRESSION
               "${ROCPROFILER_DEFAULT_FAIL_REGEX}"
               SKIP_REGULAR_EXPRESSION
               "${ROCPROFV3_TESTS_PC_SAMPLING_SKIP_REGEX}"
               DISABLED
               "${IS_PC_SAMPLING_DISABLED}")

add_test(
    NAME rocprofv3-test-pc-sampling-host-trap-transpose-multiple-agents-input-yaml-execute
    COMMAND
        $<TARGET_FILE:rocprofiler-sdk::rocprofv3> -i
        ${CMAKE_CURRENT_BINARY_DIR}/input.yml -d
        ${CMAKE_CURRENT_BINARY_DIR}/pc_sampling_yaml_input -o out --output-format csv --
        $<TARGET_FILE:transpose> ${TRANSPOSE_NUM_THREADS} ${TRANSPOSE_NUM_ITERATIONS})

set_tests_properties(
    rocprofv3-test-pc-sampling-host-trap-transpose-multiple-agents-input-yaml-execute
    PROPERTIES TIMEOUT
               90
               LABELS
               "integration-tests;pc-sampling"
               ENVIRONMENT
               "${pc-sampling-env-host-trap-transpose-multiple-agents}"
               FAIL_REGULAR_EXPRESSION
               "${ROCPROFILER_DEFAULT_FAIL_REGEX}"
               SKIP_REGULAR_EXPRESSION
               "${ROCPROFV3_TESTS_PC_SAMPLING_SKIP_REGEX}"
               DISABLED
               "${IS_PC_SAMPLING_DISABLED}")

# ========================= Validation tests

add_test(
    NAME rocprofv3-test-pc-sampling-host-trap-transpose-multiple-agents-input-cmd-validate
    COMMAND
        ${Python3_EXECUTABLE} ${CMAKE_CURRENT_BINARY_DIR}/validate.py -k
        test_multi_agent_support --input-samples-csv
        ${CMAKE_CURRENT_BINARY_DIR}/pc_sampling_cmd_input/out_pc_sampling_host_trap.csv
        --input-kernel-trace-csv
        ${CMAKE_CURRENT_BINARY_DIR}/pc_sampling_cmd_input/out_kernel_trace.csv
        --input-agent-info-csv
        ${CMAKE_CURRENT_BINARY_DIR}/pc_sampling_cmd_input/out_agent_info.csv)

set_tests_properties(
    rocprofv3-test-pc-sampling-host-trap-transpose-multiple-agents-input-cmd-validate
    PROPERTIES
        TIMEOUT
        60
        LABELS
        "integration-tests;pc-sampling"
        DEPENDS
        "rocprofv3-test-pc-sampling-host-trap-transpose-multiple-agents-input-cmd-execute"
        FAIL_REGULAR_EXPRESSION
        "${ROCPROFILER_DEFAULT_FAIL_REGEX}"
        SKIP_REGULAR_EXPRESSION
        "${ROCPROFV3_TESTS_PC_SAMPLING_SKIP_REGEX}"
        DISABLED
        "${IS_PC_SAMPLING_DISABLED}")

add_test(
    NAME rocprofv3-test-pc-sampling-host-trap-transpose-multiple-agents-input-json-validate
    COMMAND
        ${Python3_EXECUTABLE} ${CMAKE_CURRENT_BINARY_DIR}/validate.py -k
        test_multi_agent_support --input-samples-csv
        ${CMAKE_CURRENT_BINARY_DIR}/pc_sampling_json_input/out_pc_sampling_host_trap.csv
        --input-kernel-trace-csv
        ${CMAKE_CURRENT_BINARY_DIR}/pc_sampling_json_input/out_kernel_trace.csv
        --input-agent-info-csv
        ${CMAKE_CURRENT_BINARY_DIR}/pc_sampling_json_input/out_agent_info.csv)

set_tests_properties(
    rocprofv3-test-pc-sampling-host-trap-transpose-multiple-agents-input-json-validate
    PROPERTIES
        TIMEOUT
        60
        LABELS
        "integration-tests;pc-sampling"
        DEPENDS
        "rocprofv3-test-pc-sampling-host-trap-transpose-multiple-agents-input-json-execute"
        FAIL_REGULAR_EXPRESSION
        "${ROCPROFILER_DEFAULT_FAIL_REGEX}"
        SKIP_REGULAR_EXPRESSION
        "${ROCPROFV3_TESTS_PC_SAMPLING_SKIP_REGEX}"
        DISABLED
        "${IS_PC_SAMPLING_DISABLED}")

add_test(
    NAME rocprofv3-test-pc-sampling-host-trap-transpose-multiple-agents-input-yaml-validate
    COMMAND
        ${Python3_EXECUTABLE} ${CMAKE_CURRENT_BINARY_DIR}/validate.py -k
        test_multi_agent_support --input-samples-csv
        ${CMAKE_CURRENT_BINARY_DIR}/pc_sampling_yaml_input/out_pc_sampling_host_trap.csv
        --input-kernel-trace-csv
        ${CMAKE_CURRENT_BINARY_DIR}/pc_sampling_yaml_input/out_kernel_trace.csv
        --input-agent-info-csv
        ${CMAKE_CURRENT_BINARY_DIR}/pc_sampling_yaml_input/out_agent_info.csv)

set_tests_properties(
    rocprofv3-test-pc-sampling-host-trap-transpose-multiple-agents-input-yaml-validate
    PROPERTIES
        TIMEOUT
        60
        LABELS
        "integration-tests;pc-sampling"
        DEPENDS
        "rocprofv3-test-pc-sampling-host-trap-transpose-multiple-agents-input-yaml-execute"
        FAIL_REGULAR_EXPRESSION
        "${ROCPROFILER_DEFAULT_FAIL_REGEX}"
        SKIP_REGULAR_EXPRESSION
        "${ROCPROFV3_TESTS_PC_SAMPLING_SKIP_REGEX}"
        DISABLED
        "${IS_PC_SAMPLING_DISABLED}")
