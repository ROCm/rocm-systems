#
#
#
cmake_minimum_required(VERSION 3.24 FATAL_ERROR)

project(
    rocprofiler-sdk-tests-hip-graph-capture-update
    LANGUAGES CXX
    VERSION 0.0.0)

find_package(rocprofiler-sdk REQUIRED)

rocprofiler_add_integration_execute_test(
    hip-graph-capture-update
    TARGET hip-graph-capture-update
    TIMEOUT 60
    LABELS "integration-tests"
    PRELOAD "$<TARGET_FILE:rocprofiler-sdk-json-tool>"
    ENVIRONMENT
        "ROCPROFILER_TOOL_OUTPUT_FILE=hip-graph-capture-update-test.json"
        "ROCPROFILER_TOOL_CONTEXTS=HIP_API_CALLBACK,HIP_API_BUFFERED,KERNEL_DISPATCH_CALLBACK,KERNEL_DISPATCH_BUFFERED,CODE_OBJECT"
    FIXTURES_SETUP hip-graph-capture-update)

rocprofiler_add_integration_validate_test(
    graph-capture-update
    TEST_PATHS validate.py
    COPY conftest.py
    CONFIG pytest.ini
    LABELS "integration-tests"
    TIMEOUT 45
    FIXTURES_REQUIRED hip-graph-capture-update
    ARGS --input ${CMAKE_CURRENT_BINARY_DIR}/hip-graph-capture-update-test.json)
