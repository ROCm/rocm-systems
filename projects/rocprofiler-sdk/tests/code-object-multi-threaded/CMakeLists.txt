#
# Multi-threaded code object loading test
#
cmake_minimum_required(VERSION 3.24 FATAL_ERROR)

project(rocprofiler-sdk-tests-code-object-multi-threaded LANGUAGES CXX)

find_package(rocprofiler-sdk REQUIRED)

# Build the client tool library
add_library(code-object-multithread-client SHARED)
target_sources(code-object-multithread-client PRIVATE client.cpp)
target_link_libraries(code-object-multithread-client
                      PRIVATE rocprofiler-sdk::rocprofiler-sdk)

# Register the test with CTest
rocprofiler_add_integration_execute_test(
    test-code-object-multithread
    COMMAND $<TARGET_FILE:code-object-multithread-testapp>
    DEPENDS code-object-multithread-client
    TIMEOUT 120
    LABELS "integration-tests"
    PRELOAD "$<TARGET_FILE:code-object-multithread-client>"
    ENVIRONMENT
        "LD_LIBRARY_PATH=$<TARGET_FILE_DIR:rocprofiler-sdk::rocprofiler-sdk-shared-library>:$ENV{LD_LIBRARY_PATH}"
    PASS_REGULAR_EXPRESSION
        "Test PASSED: Successfully traced multi-threaded code object loading!"
    FAIL_REGULAR_EXPRESSION
        "${ROCPROFILER_DEFAULT_FAIL_REGEX}|SEGFAULT|Failed to \.*|Context is not valid|ERROR: \.*"
    )
