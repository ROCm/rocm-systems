#
# HIP buffered dangling-argument test
#
cmake_minimum_required(VERSION 3.21.0 FATAL_ERROR)

project(rocprofiler-sdk-tests-hip-buffered-dangling-arg LANGUAGES CXX)

find_package(rocprofiler-sdk REQUIRED)

# Build the client tool library
add_library(hip-buffered-dangling-arg-client SHARED)
target_sources(hip-buffered-dangling-arg-client PRIVATE client.cpp)
target_link_libraries(hip-buffered-dangling-arg-client
                      PRIVATE rocprofiler-sdk::rocprofiler-sdk)

# Register the test with CTest. The helper prepends ROCPROFILER_MEMCHECK_PRELOAD_ENV_VALUE
# to PRELOAD under sanitizer builds, so only the client library is listed here.
rocprofiler_add_integration_execute_test(
    test-hip-buffered-dangling-arg
    COMMAND $<TARGET_FILE:hip-buffered-dangling-arg-testapp>
    DEPENDS hip-buffered-dangling-arg-client
    TIMEOUT 120
    LABELS "integration-tests"
    PRELOAD "$<TARGET_FILE:hip-buffered-dangling-arg-client>"
    ENVIRONMENT
        "LD_LIBRARY_PATH=$<TARGET_FILE_DIR:rocprofiler-sdk::rocprofiler-sdk-shared-library>:$ENV{LD_LIBRARY_PATH}"
    PASS_REGULAR_EXPRESSION
        "Test PASSED: buffered HIP API string arguments stringified safely!"
    FAIL_REGULAR_EXPRESSION
        "${ROCPROFILER_DEFAULT_FAIL_REGEX}|SEGFAULT|Failed to \.*|Context is not valid|ERROR: \.*"
    )
