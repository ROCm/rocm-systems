#
# Per-operation filtering of MEMORY_ALLOCATION tracing (callback and buffered)
#
cmake_minimum_required(VERSION 3.24 FATAL_ERROR)

project(
    rocprofiler-sdk-tests-memory-allocation-op-filter
    LANGUAGES CXX
    VERSION 0.0.0)

find_package(rocprofiler-sdk REQUIRED)

add_library(memory-allocation-op-filter-client SHARED)
target_sources(memory-allocation-op-filter-client PRIVATE client.cpp)
target_link_libraries(memory-allocation-op-filter-client
                      PRIVATE rocprofiler-sdk::rocprofiler-sdk)

rocprofiler_add_integration_execute_test(
    memory-allocation-op-filter
    TARGET hsa-memory-allocation
    DEPENDS memory-allocation-op-filter-client
    TIMEOUT 45
    LABELS "integration-tests"
    PRELOAD "$<TARGET_FILE:memory-allocation-op-filter-client>"
    ENVIRONMENT "ROCPROFILER_TOOL_OUTPUT_FILE=memory-allocation-op-filter.json"
    FIXTURES_SETUP memory-allocation-op-filter)

rocprofiler_add_integration_validate_test(
    memory-allocation-op-filter
    TEST_PATHS validate.py
    COPY conftest.py
    CONFIG pytest.ini
    LABELS "integration-tests"
    TIMEOUT 45
    FIXTURES_REQUIRED memory-allocation-op-filter
    ARGS --input ${CMAKE_CURRENT_BINARY_DIR}/memory-allocation-op-filter.json)
