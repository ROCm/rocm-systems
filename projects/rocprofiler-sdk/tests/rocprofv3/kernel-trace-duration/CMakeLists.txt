#
# rocprofv3 tool tests for kernel duration
#
cmake_minimum_required(VERSION 3.24 FATAL_ERROR)

project(
    rocprofiler-sdk-tests-rocprofv3-kernel-trace-duration
    LANGUAGES CXX
    VERSION 0.0.0)

find_package(rocprofiler-sdk REQUIRED)
find_package(Python3 REQUIRED)

set(rocprofv3-generate-env
    "PYTHONPATH=${rocprofiler-sdk_LIB_DIR}/python${Python3_VERSION_MAJOR}.${Python3_VERSION_MINOR}/site-packages"
    "ROCPROF_OUTPUT_FORMAT=json,rocpd")

# Generate BOTH JSON and rocpd in a SINGLE execution using environment variable
rocprofiler_add_integration_execute_test(
    rocprofv3-test-kernel-trace-duration-generate
    COMMAND
        $<TARGET_FILE:rocprofiler-sdk::rocprofv3> -d
        ${CMAKE_CURRENT_BINARY_DIR}/kernel-trace-duration -o out_combined --kernel-trace
        -- $<TARGET_FILE:simple-transpose>
    DEPENDS simple-transpose
    TIMEOUT 120
    LABELS "kernel-trace-duration"
    ENVIRONMENT "${rocprofv3-generate-env}"
    FIXTURES_SETUP rocprofv3-test-kernel-trace-duration-data)

# Validate: compare rocpd kernel query output with JSON
rocprofiler_add_integration_validate_test(
    rocprofv3-test-kernel-trace-duration-validation BUNDLE_TESTS
    TEST_PATHS validate.py
    COPY conftest.py
    ARGS --json-input
         ${CMAKE_CURRENT_BINARY_DIR}/kernel-trace-duration/out_combined_results.json
         --db-input
         ${CMAKE_CURRENT_BINARY_DIR}/kernel-trace-duration/out_combined_results.db
         -x
    TIMEOUT 120
    LABELS "kernel-trace-duration"
    PYTHON_PATH
        "${rocprofiler-sdk_LIB_DIR}/python${Python3_VERSION_MAJOR}.${Python3_VERSION_MINOR}/site-packages"
    FIXTURES_REQUIRED rocprofv3-test-kernel-trace-duration-data)
