#
#
#
cmake_minimum_required(VERSION 3.21.0 FATAL_ERROR)

project(
    rocprofiler-sdk-tests-hip-graph-tracing
    LANGUAGES CXX
    VERSION 0.0.0)

find_package(rocprofiler-sdk REQUIRED)

set(PYTEST_ARGS)
if(ROCPROFILER_MEMCHECK MATCHES "(Address|Thread)Sanitizer" OR ROCPROFILER_BUILD_CODECOV)
    set(PYTEST_ARGS -k "not test_total_runtime")
endif()

rocprofiler_add_integration_execute_test(
    hip-graph-tracing
    TARGET hip-graph
    TIMEOUT 100
    LABELS "integration-tests"
    PRELOAD "$<TARGET_FILE:rocprofiler-sdk-json-tool>"
    ENVIRONMENT
        "ROCPROFILER_TOOL_OUTPUT_FILE=hip-graph-tracing-test.json"
        "ROCPROFILER_TOOL_CONTEXTS=HIP_API_CALLBACK,HIP_API_BUFFERED,KERNEL_DISPATCH_CALLBACK,KERNEL_DISPATCH_BUFFERED,CODE_OBJECT"
    FIXTURES_SETUP hip-graph-tracing)

rocprofiler_add_integration_validate_test(
    graph-tracing
    TEST_PATHS validate.py
    COPY conftest.py
    CONFIG pytest.ini
    LABELS "integration-tests"
    TIMEOUT 45
    FIXTURES_REQUIRED hip-graph-tracing
    DISCOVERY_ARGS ${PYTEST_ARGS}
    ARGS --input ${CMAKE_CURRENT_BINARY_DIR}/hip-graph-tracing-test.json)
