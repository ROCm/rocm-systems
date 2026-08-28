#
# Multiple anytime tool configuration integration test
#
cmake_minimum_required(VERSION 3.21.0 FATAL_ERROR)

project(
    rocprofiler-sdk-tests-multi-tool-test
    LANGUAGES CXX
    VERSION 0.0.0)

find_package(rocprofiler-sdk REQUIRED)
find_package(Python3 REQUIRED)

set(multi-tool-test-env
    "LD_LIBRARY_PATH=$<TARGET_FILE_DIR:rocprofiler-sdk::rocprofiler-sdk-shared-library>:$<TARGET_FILE_DIR:kinetoesque-tool>:$ENV{LD_LIBRARY_PATH}"
    "PYTHONPATH=${CMAKE_BINARY_DIR}/lib/python3/site-packages"
    "ROCPROFILER_SDK_TEST_LIB_DIR=$<TARGET_FILE_DIR:kinetoesque-tool>"
    "TEST_OUTPUT_DIR=${CMAKE_CURRENT_BINARY_DIR}"
    "KINETOESQUE_OUTPUT_FILE=${CMAKE_CURRENT_BINARY_DIR}/kinetoesque-trace.json"
    "PROTONESQUE_OUTPUT_FILE=${CMAKE_CURRENT_BINARY_DIR}/protonesque-trace.json")

rocprofiler_configure_pytest_files(CONFIG pytest.ini COPY conftest.py)

# Execute the multi-tool test - generates JSON output from both profilers
rocprofiler_add_integration_execute_test(
    multi-tool-test
    COMMAND ${Python3_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/test_multiple_tools.py
    DEPENDS kinetoesque-tool protonesque-tool python-hip-kernels
    TIMEOUT 120
    LABELS "integration-tests;multi-tool"
    ENVIRONMENT "${multi-tool-test-env}"
    SKIP_REGULAR_EXPRESSION "test skipped"
    FAIL_REGULAR_EXPRESSION "${ROCPROFILER_DEFAULT_FAIL_REGEX}"
    FIXTURES_SETUP multi-tool-test)

# Validate JSON output using pytest
rocprofiler_add_integration_validate_test(
    multi-tool-test
    TEST_PATHS validate.py
    COPY conftest.py
    CONFIG pytest.ini
    TIMEOUT 45
    LABELS "integration-tests;multi-tool"
    FAIL_REGULAR_EXPRESSION "${ROCPROFILER_DEFAULT_FAIL_REGEX}"
    FIXTURES_REQUIRED multi-tool-test
    ARGS --kinetoesque-input ${CMAKE_CURRENT_BINARY_DIR}/kinetoesque-trace.json
         --protonesque-input ${CMAKE_CURRENT_BINARY_DIR}/protonesque-trace.json)
