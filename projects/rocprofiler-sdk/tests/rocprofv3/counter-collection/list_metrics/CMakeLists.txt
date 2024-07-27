#
# rocprofv3 tool test
#
cmake_minimum_required(VERSION 3.21.0 FATAL_ERROR)

project(
    rocprofiler-tests-counter-collection
    LANGUAGES CXX
    VERSION 0.0.0)

find_package(rocprofiler-sdk REQUIRED)

rocprofiler_configure_pytest_files(CONFIG pytest.ini COPY validate.py conftest.py
                                                          input.json)

# basic-metrics
add_test(NAME rocprofv3-test-list-metrics-execute
         COMMAND $<TARGET_FILE:rocprofiler-sdk::rocprofv3> -d
                 ${CMAKE_CURRENT_BINARY_DIR}/out_cc_2 -o metrics --list-metrics)

# list-metrics-stdout
add_test(NAME rocprofv3-test-list-metrics-std-out-execute
         COMMAND $<TARGET_FILE:rocprofiler-sdk::rocprofv3> -i
                 ${CMAKE_CURRENT_BINARY_DIR}/input.json)

string(REPLACE "LD_PRELOAD=" "ROCPROF_PRELOAD=" PRELOAD_ENV
               "${ROCPROFILER_MEMCHECK_PRELOAD_ENV}")

set(cc-env-list-metrics "${PRELOAD_ENV}")

set_tests_properties(
    rocprofv3-test-list-metrics-execute
    PROPERTIES TIMEOUT 45 LABELS "integration-tests" ENVIRONMENT "${cc-env-list-metrics}"
               FAIL_REGULAR_EXPRESSION "${ROCPROFILER_DEFAULT_FAIL_REGEX}")

set_tests_properties(
    rocprofv3-test-list-metrics-std-out-execute
    PROPERTIES
        TIMEOUT
        45
        LABELS
        "integration-tests"
        ENVIRONMENT
        "${cc-env-list-metrics}"
        PASS_REGULAR_EXPRESSION
        "gpu-agent[0-9]*:[a-zA-Z_]*\\n;Description:(.*)\\n*;Expression:(.)*\\n*;Block:[a-zA-Z]*\\n*;Dimensions:([A-Z_]*\\[[0-9]*:[0-9]*\\])*\\n*;"
    )
set(VALIDATION_FILES ${CMAKE_CURRENT_BINARY_DIR}/out_cc_2/metrics_basic_metrics.csv
                     ${CMAKE_CURRENT_BINARY_DIR}/out_cc_2/metrics_derived_metrics.csv)

add_test(
    NAME rocprofv3-test-list-metrics-validate
    COMMAND
        ${Python3_EXECUTABLE} ${CMAKE_CURRENT_BINARY_DIR}/validate.py
        --derived-metrics-input
        ${CMAKE_CURRENT_BINARY_DIR}/out_cc_2/metrics_derived_metrics.csv
        --basic-metrics-input
        ${CMAKE_CURRENT_BINARY_DIR}/out_cc_2/metrics_basic_metrics.csv)

set_tests_properties(
    rocprofv3-test-list-metrics-validate
    PROPERTIES TIMEOUT
               45
               LABELS
               "integration-tests"
               DEPENDS
               rocprofv3-test-list-metrics-execute
               FAIL_REGULAR_EXPRESSION
               "${ROCPROFILER_DEFAULT_FAIL_REGEX}"
               ATTACHED_FILES_ON_FAIL
               "${VALIDATION_FILES}")
