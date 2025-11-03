#
#
#
cmake_minimum_required(VERSION 3.21.0 FATAL_ERROR)

project(
    rocprofiler-sdk-tests-async-copy-tracing
    LANGUAGES CXX
    VERSION 0.0.0)

find_package(rocprofiler-sdk REQUIRED)

if(ROCPROFILER_MEMCHECK_PRELOAD_ENV_VALUE)
    set(PRELOAD_ENV
        "${ROCPROFILER_MEMCHECK_PRELOAD_ENV_VALUE}:$<TARGET_FILE:rocprofiler-sdk-json-tool>"
        )
else()
    set(PRELOAD_ENV "$<TARGET_FILE:rocprofiler-sdk-json-tool>")
endif()

rocprofiler_add_integration_execute_test(
    async-copy-tracing
    TARGET transpose
    TIMEOUT 45
    LABELS "integration-tests"
    PRELOAD "${PRELOAD_ENV}"
    ENVIRONMENT "ROCPROFILER_TOOL_OUTPUT_FILE=async-copy-tracing-test.json"
    FIXTURES_SETUP async-copy-tracing)

rocprofiler_add_integration_validate_test(
    async-copy-tracing
    TEST_PATHS validate.py
    COPY conftest.py
    CONFIG pytest.ini
    LABELS "integration-tests"
    TIMEOUT 45
    FIXTURES_REQUIRED async-copy-tracing
    ARGS --input ${CMAKE_CURRENT_BINARY_DIR}/async-copy-tracing-test.json)
