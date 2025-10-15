#
# rocprofv3 tool test
#
cmake_minimum_required(VERSION 3.21.0 FATAL_ERROR)

project(
    rocprofiler-sdk-tests-pc-sampling-host-trap-exec-mask-manipulation
    LANGUAGES CXX
    VERSION 0.0.0)

# Checking whether host-trap PC sampling is disabled. If so, skip the tests.
rocprofiler_sdk_pc_sampling_disabled(IS_PC_SAMPLING_DISABLED)

find_package(rocprofiler-sdk REQUIRED)

string(REPLACE "LD_PRELOAD=" "ROCPROF_PRELOAD=" PRELOAD_ENV
               "${ROCPROFILER_MEMCHECK_PRELOAD_ENV}")

rocprofiler_add_integration_execute_test(
    rocprofv3-test-pc-sampling-host-trap-exec-mask-manipulation-input-cmd
    COMMAND
        $<TARGET_FILE:rocprofiler-sdk::rocprofv3> --pc-sampling-unit time
        --pc-sampling-method host_trap --pc-sampling-interval 1 -d
        ${CMAKE_CURRENT_BINARY_DIR}/pc_sampling_cmd_input -o out --output-format csv json
        -- $<TARGET_FILE:exec-mask-manipulation>
    DEPENDS exec-mask-manipulation
    TIMEOUT 90
    LABELS "integration-tests;pc-sampling"
    PRELOAD "${PRELOAD_ENV}"
    SKIP_REGULAR_EXPRESSION "${ROCPROFV3_TESTS_PC_SAMPLING_SKIP_REGEX}"
    FIXTURES_SETUP rocprofv3-test-pc-sampling-host-trap-exec-mask-manipulation-input-cmd
    DISABLED "${IS_PC_SAMPLING_DISABLED}")

rocprofiler_add_integration_validate_test(
    rocprofv3-test-pc-sampling-host-trap-exec-mask-manipulation-input-cmd
    TEST_PATHS validate.py
    COPY conftest.py input.json input.yml
    CONFIG pytest.ini
    DISCOVERY_ARGS -k test_validate_pc_sampling_exec_mask_manipulation_
    ARGS --input-csv
         ${CMAKE_CURRENT_BINARY_DIR}/pc_sampling_cmd_input/out_pc_sampling_host_trap.csv
         --input-json
         ${CMAKE_CURRENT_BINARY_DIR}/pc_sampling_cmd_input/out_results.json
         --all-sampled
         False
    TIMEOUT 60
    LABELS "integration-tests;pc-sampling"
    FIXTURES_REQUIRED
        rocprofv3-test-pc-sampling-host-trap-exec-mask-manipulation-input-cmd
    SKIP_REGULAR_EXPRESSION "${ROCPROFV3_TESTS_PC_SAMPLING_SKIP_REGEX}"
    DISABLED "${IS_PC_SAMPLING_DISABLED}")

rocprofiler_add_integration_execute_test(
    rocprofv3-test-pc-sampling-host-trap-exec-mask-manipulation-input-json
    COMMAND
        $<TARGET_FILE:rocprofiler-sdk::rocprofv3> -i
        ${CMAKE_CURRENT_BINARY_DIR}/input.json -d
        ${CMAKE_CURRENT_BINARY_DIR}/pc_sampling_json_input --
        $<TARGET_FILE:exec-mask-manipulation>
    DEPENDS exec-mask-manipulation
    TIMEOUT 90
    LABELS "integration-tests;pc-sampling"
    PRELOAD "${PRELOAD_ENV}"
    SKIP_REGULAR_EXPRESSION "${ROCPROFV3_TESTS_PC_SAMPLING_SKIP_REGEX}"
    FIXTURES_SETUP rocprofv3-test-pc-sampling-host-trap-exec-mask-manipulation-input-json
    DISABLED "${IS_PC_SAMPLING_DISABLED}")

rocprofiler_add_integration_validate_test(
    rocprofv3-test-pc-sampling-host-trap-exec-mask-manipulation-input-json
    TEST_PATHS validate.py
    COPY conftest.py input.json input.yml
    CONFIG pytest.ini
    DISCOVERY_ARGS -k test_validate_pc_sampling_exec_mask_manipulation_
    ARGS --input-csv
         ${CMAKE_CURRENT_BINARY_DIR}/pc_sampling_json_input/out_pc_sampling_host_trap.csv
         --input-json
         ${CMAKE_CURRENT_BINARY_DIR}/pc_sampling_json_input/out_results.json
         --all-sampled
         False
    TIMEOUT 60
    LABELS "integration-tests;pc-sampling"
    FIXTURES_REQUIRED
        rocprofv3-test-pc-sampling-host-trap-exec-mask-manipulation-input-json
    SKIP_REGULAR_EXPRESSION "${ROCPROFV3_TESTS_PC_SAMPLING_SKIP_REGEX}"
    DISABLED "${IS_PC_SAMPLING_DISABLED}")

rocprofiler_add_integration_execute_test(
    rocprofv3-test-pc-sampling-host-trap-exec-mask-manipulation-input-yaml
    COMMAND
        $<TARGET_FILE:rocprofiler-sdk::rocprofv3> -i ${CMAKE_CURRENT_BINARY_DIR}/input.yml
        -d ${CMAKE_CURRENT_BINARY_DIR}/pc_sampling_yaml_input -o out --output-format csv
        json -- $<TARGET_FILE:exec-mask-manipulation>
    DEPENDS exec-mask-manipulation
    TIMEOUT 90
    LABELS "integration-tests;pc-sampling"
    PRELOAD "${PRELOAD_ENV}"
    SKIP_REGULAR_EXPRESSION "${ROCPROFV3_TESTS_PC_SAMPLING_SKIP_REGEX}"
    FIXTURES_SETUP rocprofv3-test-pc-sampling-host-trap-exec-mask-manipulation-input-yaml
    DISABLED "${IS_PC_SAMPLING_DISABLED}")

rocprofiler_add_integration_validate_test(
    rocprofv3-test-pc-sampling-host-trap-exec-mask-manipulation-input-yaml
    TEST_PATHS validate.py
    COPY conftest.py input.json input.yml
    CONFIG pytest.ini
    DISCOVERY_ARGS -k test_validate_pc_sampling_exec_mask_manipulation_
    ARGS --input-csv
         ${CMAKE_CURRENT_BINARY_DIR}/pc_sampling_yaml_input/out_pc_sampling_host_trap.csv
         --input-json
         ${CMAKE_CURRENT_BINARY_DIR}/pc_sampling_yaml_input/out_results.json
         --all-sampled
         False
    TIMEOUT 60
    LABELS "integration-tests;pc-sampling"
    FIXTURES_REQUIRED
        rocprofv3-test-pc-sampling-host-trap-exec-mask-manipulation-input-yaml
    SKIP_REGULAR_EXPRESSION "${ROCPROFV3_TESTS_PC_SAMPLING_SKIP_REGEX}"
    DISABLED "${IS_PC_SAMPLING_DISABLED}")
