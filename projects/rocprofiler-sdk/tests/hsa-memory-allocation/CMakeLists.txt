#
#
#
cmake_minimum_required(VERSION 3.21.0 FATAL_ERROR)

project(
    rocprofiler-sdk-tests-memory-allocation-tracing
    LANGUAGES CXX
    VERSION 0.0.0)

find_package(rocprofiler-sdk REQUIRED)

rocprofiler_add_integration_execute_test(
    memory-allocation-tracing
    TARGET hsa-memory-allocation
    TIMEOUT 45
    LABELS "integration-tests"
    PRELOAD "$<TARGET_FILE:rocprofiler-sdk-json-tool>"
    ENVIRONMENT "ROCPROFILER_TOOL_OUTPUT_FILE=memory-allocation-tracing-test.json"
    FIXTURES_SETUP memory-allocation-tracing)

rocprofiler_add_integration_validate_test(
    memory-allocation-tracing
    TEST_PATHS validate.py
    COPY conftest.py
    CONFIG pytest.ini
    LABELS "integration-tests"
    TIMEOUT 45
    FIXTURES_REQUIRED memory-allocation-tracing
    ARGS --input ${CMAKE_CURRENT_BINARY_DIR}/memory-allocation-tracing-test.json)
