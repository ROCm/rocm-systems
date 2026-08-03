#
#
#
cmake_minimum_required(VERSION 3.21.0 FATAL_ERROR)

project(
    rocprofiler-sdk-tests-gpu-event-tracing
    LANGUAGES CXX
    VERSION 0.0.0)

find_package(rocprofiler-sdk REQUIRED)

set(gpu-event-tracing-env "ROCPROFILER_TOOL_DISABLE_PERFETTO=1"
                          "ROCPROFILER_TOOL_OUTPUT_FILE=gpu-event-tracing-test.json")

rocprofiler_add_integration_execute_test(
    gpu-event-tracing
    TARGET gpu-events
    TIMEOUT 100
    LABELS "integration-tests"
    ENVIRONMENT "${gpu-event-tracing-env}"
    PRELOAD "$<TARGET_FILE:rocprofiler-sdk-json-tool>"
    FIXTURES_SETUP gpu-event-tracing)

rocprofiler_add_integration_validate_test(
    gpu-event-tracing
    TEST_PATHS validate.py
    COPY conftest.py
    CONFIG pytest.ini
    LABELS "integration-tests"
    TIMEOUT 45
    FIXTURES_REQUIRED gpu-event-tracing
    ARGS --input ${CMAKE_CURRENT_BINARY_DIR}/gpu-event-tracing-test.json)
