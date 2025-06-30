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

rocprofiler_configure_pytest_files(CONFIG pytest.ini COPY validate.py conftest.py
                                                          input.json input.yml)

add_test(
    NAME rocprofv3-test-pc-sampling-host-trap-exec-mask-manipulation-input-cmd-execute
    COMMAND
        $<TARGET_FILE:rocprofiler-sdk::rocprofv3> --pc-sampling-unit time
        --pc-sampling-method host_trap --pc-sampling-interval 1 -d
        ${CMAKE_CURRENT_BINARY_DIR}/pc_sampling_cmd_input -o out --output-format csv json
        -- $<TARGET_FILE:exec-mask-manipulation>)

string(REPLACE "LD_PRELOAD=" "ROCPROF_PRELOAD=" PRELOAD_ENV
               "${ROCPROFILER_MEMCHECK_PRELOAD_ENV}")

set(pc-sampling-env-host-trap-exec-mask-manipulation "${PRELOAD_ENV}")

set_tests_properties(
    rocprofv3-test-pc-sampling-host-trap-exec-mask-manipulation-input-cmd-execute
    PROPERTIES TIMEOUT
               45
               LABELS
               "integration-tests;pc-sampling"
               ENVIRONMENT
               "${pc-sampling-env-host-trap-exec-mask-manipulation}"
               FAIL_REGULAR_EXPRESSION
               "${ROCPROFILER_DEFAULT_FAIL_REGEX}"
               SKIP_REGULAR_EXPRESSION
               "${ROCPROFV3_TESTS_PC_SAMPLING_SKIP_REGEX}"
               DISABLED
               "${IS_PC_SAMPLING_DISABLED}")

add_test(
    NAME rocprofv3-test-pc-sampling-host-trap-exec-mask-manipulation-input-json-execute
    COMMAND
        $<TARGET_FILE:rocprofiler-sdk::rocprofv3> -i
        ${CMAKE_CURRENT_BINARY_DIR}/input.json -d
        ${CMAKE_CURRENT_BINARY_DIR}/pc_sampling_json_input --
        $<TARGET_FILE:exec-mask-manipulation>)

set_tests_properties(
    rocprofv3-test-pc-sampling-host-trap-exec-mask-manipulation-input-json-execute
    PROPERTIES TIMEOUT
               45
               LABELS
               "integration-tests;pc-sampling"
               ENVIRONMENT
               "${pc-sampling-env-host-trap-exec-mask-manipulation}"
               FAIL_REGULAR_EXPRESSION
               "${ROCPROFILER_DEFAULT_FAIL_REGEX}"
               SKIP_REGULAR_EXPRESSION
               "${ROCPROFV3_TESTS_PC_SAMPLING_SKIP_REGEX}"
               DISABLED
               "${IS_PC_SAMPLING_DISABLED}")

add_test(
    NAME rocprofv3-test-pc-sampling-host-trap-exec-mask-manipulation-input-yaml-execute
    COMMAND
        $<TARGET_FILE:rocprofiler-sdk::rocprofv3> -i
        ${CMAKE_CURRENT_BINARY_DIR}/input.yml -d
        ${CMAKE_CURRENT_BINARY_DIR}/pc_sampling_yaml_input -o out --output-format csv
        json -- $<TARGET_FILE:exec-mask-manipulation>)

set_tests_properties(
    rocprofv3-test-pc-sampling-host-trap-exec-mask-manipulation-input-yaml-execute
    PROPERTIES TIMEOUT
               45
               LABELS
               "integration-tests;pc-sampling"
               ENVIRONMENT
               "${pc-sampling-env-host-trap-exec-mask-manipulation}"
               FAIL_REGULAR_EXPRESSION
               "${ROCPROFILER_DEFAULT_FAIL_REGEX}"
               SKIP_REGULAR_EXPRESSION
               "${ROCPROFV3_TESTS_PC_SAMPLING_SKIP_REGEX}"
               DISABLED
               "${IS_PC_SAMPLING_DISABLED}")

# ========================= Validation tests

add_test(
    NAME rocprofv3-test-pc-sampling-host-trap-exec-mask-manipulation-input-cmd-validate
    COMMAND
        ${Python3_EXECUTABLE} ${CMAKE_CURRENT_BINARY_DIR}/validate.py -k
        test_validate_pc_sampling_exec_mask_manipulation_ --input-csv
        ${CMAKE_CURRENT_BINARY_DIR}/pc_sampling_cmd_input/out_pc_sampling_host_trap.csv
        --input-json ${CMAKE_CURRENT_BINARY_DIR}/pc_sampling_cmd_input/out_results.json
        --all-sampled False)

set_tests_properties(
    rocprofv3-test-pc-sampling-host-trap-exec-mask-manipulation-input-cmd-validate
    PROPERTIES
        TIMEOUT
        60
        LABELS
        "integration-tests;pc-sampling"
        DEPENDS
        "rocprofv3-test-pc-sampling-host-trap-exec-mask-manipulation-input-cmd-execute"
        FAIL_REGULAR_EXPRESSION
        "${ROCPROFILER_DEFAULT_FAIL_REGEX}"
        SKIP_REGULAR_EXPRESSION
        "${ROCPROFV3_TESTS_PC_SAMPLING_SKIP_REGEX}"
        DISABLED
        "${IS_PC_SAMPLING_DISABLED}")

add_test(
    NAME rocprofv3-test-pc-sampling-host-trap-exec-mask-manipulation-input-json-validate
    COMMAND
        ${Python3_EXECUTABLE} ${CMAKE_CURRENT_BINARY_DIR}/validate.py -k
        test_validate_pc_sampling_exec_mask_manipulation_ --input-csv
        ${CMAKE_CURRENT_BINARY_DIR}/pc_sampling_json_input/out_pc_sampling_host_trap.csv
        --input-json ${CMAKE_CURRENT_BINARY_DIR}/pc_sampling_json_input/out_results.json
        --all-sampled False)

set_tests_properties(
    rocprofv3-test-pc-sampling-host-trap-exec-mask-manipulation-input-json-validate
    PROPERTIES
        TIMEOUT
        60
        LABELS
        "integration-tests;pc-sampling"
        DEPENDS
        "rocprofv3-test-pc-sampling-host-trap-exec-mask-manipulation-input-json-execute"
        FAIL_REGULAR_EXPRESSION
        "${ROCPROFILER_DEFAULT_FAIL_REGEX}"
        SKIP_REGULAR_EXPRESSION
        "${ROCPROFV3_TESTS_PC_SAMPLING_SKIP_REGEX}"
        DISABLED
        "${IS_PC_SAMPLING_DISABLED}")

add_test(
    NAME rocprofv3-test-pc-sampling-host-trap-exec-mask-manipulation-input-yaml-validate
    COMMAND
        ${Python3_EXECUTABLE} ${CMAKE_CURRENT_BINARY_DIR}/validate.py -k
        test_validate_pc_sampling_exec_mask_manipulation_ --input-csv
        ${CMAKE_CURRENT_BINARY_DIR}/pc_sampling_yaml_input/out_pc_sampling_host_trap.csv
        --input-json ${CMAKE_CURRENT_BINARY_DIR}/pc_sampling_yaml_input/out_results.json
        --all-sampled False)

set_tests_properties(
    rocprofv3-test-pc-sampling-host-trap-exec-mask-manipulation-input-yaml-validate
    PROPERTIES
        TIMEOUT
        60
        LABELS
        "integration-tests;pc-sampling"
        DEPENDS
        "rocprofv3-test-pc-sampling-host-trap-exec-mask-manipulation-input-yaml-execute"
        FAIL_REGULAR_EXPRESSION
        "${ROCPROFILER_DEFAULT_FAIL_REGEX}"
        SKIP_REGULAR_EXPRESSION
        "${ROCPROFV3_TESTS_PC_SAMPLING_SKIP_REGEX}"
        DISABLED
        "${IS_PC_SAMPLING_DISABLED}")
