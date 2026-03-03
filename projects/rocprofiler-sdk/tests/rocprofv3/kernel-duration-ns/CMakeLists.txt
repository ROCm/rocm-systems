#
# rocprofv3 tool tests for kernel duration
#
cmake_minimum_required(VERSION 3.21.0 FATAL_ERROR)

project(
    rocprofiler-sdk-tests-rocprofv3-kernel-trace-duration
    LANGUAGES CXX
    VERSION 0.0.0)

find_package(rocprofiler-sdk REQUIRED)
find_package(Python3 REQUIRED)

set(rocprofv3-env
    "${ROCPROFILER_MEMCHECK_PRELOAD_ENV}"
    "PYTHONPATH=${rocprofiler-sdk_LIB_DIR}/python${Python3_VERSION_MAJOR}.${Python3_VERSION_MINOR}/site-packages"
    "ROCPROF_OUTPUT_FORMAT=json,rocpd")

rocprofiler_configure_pytest_files(CONFIG pytest.ini COPY conftest.py validate.py)

# Generate BOTH JSON and rocpd in a SINGLE execution using environment variable
add_test(
    NAME rocprofv3-test-kernel-trace-duration-generate
    COMMAND
        $<TARGET_FILE:rocprofiler-sdk::rocprofv3> -d
        ${CMAKE_CURRENT_BINARY_DIR}/kernel-trace-duration -o out_combined --kernel-trace
        -- $<TARGET_FILE:simple-transpose>)

set_tests_properties(
    rocprofv3-test-kernel-trace-duration-generate
    PROPERTIES TIMEOUT
               120
               LABELS
               "integration-tests;kernel-trace-duration"
               ENVIRONMENT
               "${rocprofv3-env}"
               FAIL_REGULAR_EXPRESSION
               "${ROCPROFILER_DEFAULT_FAIL_REGEX}"
               FIXTURES_SETUP
               rocprofv3-test-kernel-trace-duration-data)

# Validate: Convert rocpd DB -> CSV and compare with JSON
add_test(
    NAME rocprofv3-test-kernel-trace-duration-validation
    COMMAND
        ${Python3_EXECUTABLE} ${CMAKE_CURRENT_BINARY_DIR}/validate.py --json-input
        ${CMAKE_CURRENT_BINARY_DIR}/kernel-trace-duration/out_combined_results.json
        --db-input
        ${CMAKE_CURRENT_BINARY_DIR}/kernel-trace-duration/out_combined_results.db)

set_tests_properties(
    rocprofv3-test-kernel-trace-duration-validation
    PROPERTIES TIMEOUT
               120
               LABELS
               "integration-tests;kernel-trace-duration"
               ENVIRONMENT
               "${rocprofv3-env}"
               FIXTURES_REQUIRED
               rocprofv3-test-kernel-trace-duration-data
               FAIL_REGULAR_EXPRESSION
               "${ROCPROFILER_DEFAULT_FAIL_REGEX}")
