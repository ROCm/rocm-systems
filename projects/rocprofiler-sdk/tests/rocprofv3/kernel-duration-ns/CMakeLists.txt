#
# rocprofv3 tool tests for kernel duration
#
cmake_minimum_required(VERSION 3.21.0 FATAL_ERROR)

project(rocprofiler-sdk-tests-rocprofv3-kernel-trace-duration LANGUAGES CXX VERSION 0.0.0)

find_package(rocprofiler-sdk REQUIRED)
find_package(Python3 REQUIRED)

set(rocprofv3-env
    "${ROCPROFILER_MEMCHECK_PRELOAD_ENV}"
    "PYTHONPATH=${rocprofiler-sdk_LIB_DIR}/python${Python3_VERSION_MAJOR}.${Python3_VERSION_MINOR}/site-packages"
)

rocprofiler_configure_pytest_files(CONFIG pytest.ini COPY conftest.py validate.py)

# 1) Generate JSON (ground truth)
add_test(
    NAME rocprofv3-test-kernel-trace-duration-json
    COMMAND
        $<TARGET_FILE:rocprofiler-sdk::rocprofv3>
        -d ${CMAKE_CURRENT_BINARY_DIR}/kernel-trace-duration
        -o out_json
        --output-format json
        --kernel-trace
        -- $<TARGET_FILE:simple-transpose>)

set_tests_properties(
    rocprofv3-test-kernel-trace-duration-json
    PROPERTIES TIMEOUT 120
               LABELS "integration-tests;kernel-trace-duration"
               ENVIRONMENT "${rocprofv3-env}"
               FAIL_REGULAR_EXPRESSION "${ROCPROFILER_DEFAULT_FAIL_REGEX}"
               FIXTURES_SETUP rocprofv3-test-kernel-trace-duration-run)

# 2) Generate rocpd DB
add_test(
    NAME rocprofv3-test-kernel-trace-duration-rocpd
    COMMAND
        $<TARGET_FILE:rocprofiler-sdk::rocprofv3>
        -d ${CMAKE_CURRENT_BINARY_DIR}/kernel-trace-duration
        -o out_rocpd
        --output-format rocpd
        --kernel-trace
        -- $<TARGET_FILE:simple-transpose>)

set_tests_properties(
    rocprofv3-test-kernel-trace-duration-rocpd
    PROPERTIES TIMEOUT 120
               LABELS "integration-tests;kernel-trace-duration"
               ENVIRONMENT "${rocprofv3-env}"
               FAIL_REGULAR_EXPRESSION "${ROCPROFILER_DEFAULT_FAIL_REGEX}"
               FIXTURES_REQUIRED rocprofv3-test-kernel-trace-duration-run)

# 3) Convert rocpd DB -> CSV and validate vs JSON
add_test(
    NAME rocprofv3-test-kernel-trace-duration-validation
    COMMAND ${Python3_EXECUTABLE} ${CMAKE_CURRENT_BINARY_DIR}/validate.py
            --json-input ${CMAKE_CURRENT_BINARY_DIR}/kernel-trace-duration/out_json_results.json
            --db-input ${CMAKE_CURRENT_BINARY_DIR}/kernel-trace-duration/out_rocpd_results.db)

set_tests_properties(
    rocprofv3-test-kernel-trace-duration-validation
    PROPERTIES TIMEOUT 120
               LABELS "integration-tests;kernel-trace-duration"
               ENVIRONMENT "${rocprofv3-env}"
               DEPENDS "rocprofv3-test-kernel-trace-duration-json;rocprofv3-test-kernel-trace-duration-rocpd"
               FAIL_REGULAR_EXPRESSION "${ROCPROFILER_DEFAULT_FAIL_REGEX}")
