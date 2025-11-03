#
#
#
cmake_minimum_required(VERSION 3.21.0 FATAL_ERROR)

project(
    rocprofiler-sdk-tests-scratch-memory-tracing
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
    test-scratch-memory-tracing
    COMMAND $<TARGET_FILE:scratch-memory>
    DEPENDS scratch-memory
    TIMEOUT 45
    LABELS "integration-tests"
    PRELOAD "${PRELOAD_ENV}"
    ENVIRONMENT
        "ROCPROFILER_TOOL_OUTPUT_FILE=scratch-memory-tracing-test.json"
        "LD_LIBRARY_PATH=$<TARGET_FILE_DIR:rocprofiler-sdk::rocprofiler-sdk-shared-library>:$ENV{LD_LIBRARY_PATH}"
    FIXTURES_SETUP test-scratch-memory-tracing
    UNSTABLE)

# copy to binary directory
rocprofiler_add_integration_validate_test(
    test-scratch-memory-tracing
    TEST_PATHS validate.py
    COPY conftest.py
    CONFIG pytest.ini
    ARGS --input ${CMAKE_CURRENT_BINARY_DIR}/scratch-memory-tracing-test.json
    TIMEOUT 45
    LABELS "integration-tests"
    FIXTURES_REQUIRED test-scratch-memory-tracing
    UNSTABLE)
