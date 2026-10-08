#
# Host function id assignment test
#
cmake_minimum_required(VERSION 3.21.0 FATAL_ERROR)

project(rocprofiler-sdk-tests-code-object-host-symbols LANGUAGES CXX)

find_package(rocprofiler-sdk REQUIRED)

# Build the client tool library
add_library(code-object-host-symbols-client SHARED)
target_sources(code-object-host-symbols-client PRIVATE client.cpp)
target_link_libraries(code-object-host-symbols-client
                      PRIVATE rocprofiler-sdk::rocprofiler-sdk)

# Register the test with CTest. The helper supplies the sanitizer preload, the
# integration-tests label and LD_LIBRARY_PATH.
rocprofiler_add_integration_execute_test(
    test-code-object-host-symbols
    COMMAND $<TARGET_FILE:vector-ops>
    DEPENDS code-object-host-symbols-client
    TIMEOUT 120
    PRELOAD "$<TARGET_FILE:code-object-host-symbols-client>"
    PASS_REGULAR_EXPRESSION
        "Test PASSED: Successfully verified host function id assignment!"
    FAIL_REGULAR_EXPRESSION
        "${ROCPROFILER_DEFAULT_FAIL_REGEX}|SEGFAULT|Failed to \.*|Context is not valid|ERROR: \.*"
    )
