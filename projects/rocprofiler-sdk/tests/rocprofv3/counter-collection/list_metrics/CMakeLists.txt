#
# rocprofv3 tool test
#
cmake_minimum_required(VERSION 3.21.0 FATAL_ERROR)

project(
    rocprofiler-sdk-tests-counter-collection-list-metrics
    LANGUAGES CXX
    VERSION 0.0.0)

find_package(rocprofiler-sdk REQUIRED)

rocprofiler_configure_pytest_files(CONFIG pytest.ini COPY validate.py conftest.py
                                                          input.json)

# basic-metrics
add_test(
    NAME rocprofv3-test-list-avail-execute
    COMMAND $<TARGET_FILE:rocprofiler-sdk::rocprofv3> -d
            ${CMAKE_CURRENT_BINARY_DIR}/out_counter_collection_2 -o metrics --list-avail)

# list-metrics-stdout
add_test(NAME rocprofv3-test-list-avail-std-out-execute
         COMMAND $<TARGET_FILE:rocprofiler-sdk::rocprofv3> -i
                 ${CMAKE_CURRENT_BINARY_DIR}/input.json)

# list-metrics with tracing
add_test(
    NAME rocprofv3-test-list-avail-trace-execute
    COMMAND
        $<TARGET_FILE:rocprofiler-sdk::rocprofv3> -d
        ${CMAKE_CURRENT_BINARY_DIR}/out_counter_collection_2 -o metrics --list-avail
        --sys-trace --log-level warning -- $<TARGET_FILE:transpose>)

string(REPLACE "LD_PRELOAD=" "ROCPROF_PRELOAD=" PRELOAD_ENV
               "${ROCPROFILER_MEMCHECK_PRELOAD_ENV}")

set(cc-env-list-metrics "${PRELOAD_ENV}" "ROCPD_LOG_LEVEL=trace")

set_tests_properties(
    rocprofv3-test-list-avail-execute
    PROPERTIES TIMEOUT 45 LABELS "integration-tests" ENVIRONMENT "${cc-env-list-metrics}"
               FAIL_REGULAR_EXPRESSION "${ROCPROFILER_DEFAULT_FAIL_REGEX}")

set_tests_properties(
    rocprofv3-test-list-avail-std-out-execute
    PROPERTIES
        TIMEOUT
        45
        LABELS
        "integration-tests"
        ENVIRONMENT
        "${cc-env-list-metrics}"
        PASS_REGULAR_EXPRESSION
        "gpu-agent:[0-9]*\\n*; Name:\\t[a-zA-Z_]*\\n;Description:\\t(.*)\\n*;Expression:\\t(.)*\\n*;Block:\\t[a-zA-Z]*\\n*;Dimensions:\\t([A-Z_]*\\[[0-9]*:[0-9]*\\])*\\n*;Method:\\t(.*);Unit:\\t(.*);Minimum_Interval:\\t[0-9]*;Maximum_Interval:\\t[0-9]*\\n*;"
    )

set_tests_properties(
    rocprofv3-test-list-avail-trace-execute
    PROPERTIES TIMEOUT 45 LABELS "integration-tests" ENVIRONMENT "${cc-env-list-metrics}"
               FAIL_REGULAR_EXPRESSION "${ROCPROFILER_DEFAULT_FAIL_REGEX}")

set(VALIDATION_FILES
    ${CMAKE_CURRENT_BINARY_DIR}/out_counter_collection_2/metrics_basic_metrics.csv
    ${CMAKE_CURRENT_BINARY_DIR}/out_counter_collection_2/metrics_derived_metrics.csv
    ${CMAKE_CURRENT_BINARY_DIR}/out_counter_collection_2/metrics_pc_sample_config.csv)

add_test(
    NAME rocprofv3-test-list-avail-validate
    COMMAND
        ${Python3_EXECUTABLE} ${CMAKE_CURRENT_BINARY_DIR}/validate.py
        --derived-metrics-input
        ${CMAKE_CURRENT_BINARY_DIR}/out_counter_collection_2/metrics_derived_metrics.csv
        --basic-metrics-input
        ${CMAKE_CURRENT_BINARY_DIR}/out_counter_collection_2/metrics_basic_metrics.csv
        --pc-sample-config-input
        ${CMAKE_CURRENT_BINARY_DIR}/out_counter_collection_2/metrics_pc_sample_config.csv)

set_tests_properties(
    rocprofv3-test-list-avail-validate
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
