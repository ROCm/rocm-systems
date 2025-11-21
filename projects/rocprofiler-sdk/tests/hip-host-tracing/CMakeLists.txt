#
#
#
cmake_minimum_required(VERSION 3.21.0 FATAL_ERROR)

project(
    rocprofiler-sdk-tests-hip-host-tracing
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
    hip-host-tracing
    TARGET hip-host
    TIMEOUT 100
    LABELS "integration-tests"
    PRELOAD "${PRELOAD_ENV}"
    ENVIRONMENT "ROCPROFILER_TOOL_OUTPUT_FILE=hip-host-tracing-test.json"
    FAIL_REGULAR_EXPRESSION "${ROCPROFILER_DEFAULT_FAIL_REGEX}"
    FIXTURES_SETUP hip-host-tracing)

rocprofiler_add_integration_validate_test(
    hip-host-tracing
    TEST_PATHS validate.py
    COPY conftest.py
    CONFIG pytest.ini
    TIMEOUT 45
    LABELS "integration-tests"
    FAIL_REGULAR_EXPRESSION "${ROCPROFILER_DEFAULT_FAIL_REGEX}"
    FIXTURES_REQUIRED hip-host-tracing
    ARGS --input ${CMAKE_CURRENT_BINARY_DIR}/hip-host-tracing-test.json)
