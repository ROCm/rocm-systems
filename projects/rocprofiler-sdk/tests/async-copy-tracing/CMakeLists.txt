#
#
#
cmake_minimum_required(VERSION 3.24 FATAL_ERROR)

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
    TARGET hip-memcpy
    TIMEOUT 45
    WSL_TIMEOUT_SCALE
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
    DISCOVERY_ARGS -k "not completion_signal_lifecycle"
    ARGS --input ${CMAKE_CURRENT_BINARY_DIR}/async-copy-tracing-test.json)

string(REPLACE "LD_PRELOAD=" "ROCPROF_PRELOAD=" ROCPROFV3_PRELOAD_ENV
               "${ROCPROFILER_MEMCHECK_PRELOAD_ENV}")

rocprofiler_add_integration_execute_test(
    async-copy-tracing-app-copy-time
    COMMAND
        $<TARGET_FILE:rocprofiler-sdk::rocprofv3> --memory-copy-trace --marker-trace -d
        ${CMAKE_CURRENT_BINARY_DIR}/app-copy-time-trace -o out --output-format csv --
        $<TARGET_FILE:hsa-async-copy-time>
        ${CMAKE_CURRENT_BINARY_DIR}/app-copy-time-application-times.csv
    DEPENDS hsa-async-copy-time
    TIMEOUT 45
    WSL_TIMEOUT_SCALE
    LABELS "integration-tests"
    ENVIRONMENT "${ROCPROFV3_PRELOAD_ENV}"
    FIXTURES_SETUP async-copy-tracing-app-copy-time)

rocprofiler_add_integration_validate_test(
    async-copy-tracing-app-copy-time
    TEST_PATHS validate.py
    COPY conftest.py
    CONFIG pytest.ini
    LABELS "integration-tests"
    TIMEOUT 45
    FIXTURES_REQUIRED async-copy-tracing-app-copy-time
    DISCOVERY_ARGS -k completion_signal_lifecycle
    ARGS --application-times
         ${CMAKE_CURRENT_BINARY_DIR}/app-copy-time-application-times.csv --copy-trace
         ${CMAKE_CURRENT_BINARY_DIR}/app-copy-time-trace/out_memory_copy_trace.csv)
