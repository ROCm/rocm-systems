#
#
#
cmake_minimum_required(VERSION 3.21.0 FATAL_ERROR)

project(
    rocprofiler-sdk-tests-c-tool
    LANGUAGES CXX
    VERSION 0.0.0)

find_package(rocprofiler-sdk REQUIRED)

if(ROCPROFILER_MEMCHECK_PRELOAD_ENV)
    set(PRELOAD_ENV
        "${ROCPROFILER_MEMCHECK_PRELOAD_ENV}:$<TARGET_FILE:rocprofiler-sdk-c-tool>")
else()
    set(PRELOAD_ENV "LD_PRELOAD=$<TARGET_FILE:rocprofiler-sdk-c-tool>")
endif()

add_test(NAME test-c-tool-execute COMMAND $<TARGET_FILE:transpose> 1)

set(c-tool-env
    "${PRELOAD_ENV}"
    "LD_LIBRARY_PATH=$<TARGET_FILE_DIR:rocprofiler-sdk::rocprofiler-sdk-shared-library>:$ENV{LD_LIBRARY_PATH}"
    )

set_tests_properties(
    test-c-tool-execute
    PROPERTIES
        TIMEOUT
        45
        LABELS
        "integration-tests"
        ENVIRONMENT
        "${c-tool-env}"
        PASS_REGULAR_EXPRESSION
        "Test C tool \\(priority=0\\) is using rocprofiler-sdk v([0-9]+\\.[0-9]+\\.[0-9]+)"
        FAIL_REGULAR_EXPRESSION
        "${ROCPROFILER_DEFAULT_FAIL_REGEX}|Internal thread for rocprofiler-sdk should not be created"
    )

# this test uses ROCP_TOOL_LIBRARIES instead of LD_PRELOAD
add_test(NAME test-c-tool-rocp-tool-lib-execute COMMAND $<TARGET_FILE:transpose> 1)

set(c-tool-rocp-tool-lib-env
    "${ROCPROFILER_MEMCHECK_PRELOAD_ENV}"
    "ROCP_TOOL_LIBRARIES=$<TARGET_FILE:rocprofiler-sdk-c-tool>"
    "LD_LIBRARY_PATH=$<TARGET_FILE_DIR:rocprofiler-sdk::rocprofiler-sdk-shared-library>:$ENV{LD_LIBRARY_PATH}"
    )

set_tests_properties(
    test-c-tool-rocp-tool-lib-execute
    PROPERTIES
        TIMEOUT
        45
        LABELS
        "integration-tests"
        ENVIRONMENT
        "${c-tool-rocp-tool-lib-env}"
        PASS_REGULAR_EXPRESSION
        "Test C tool \\(priority=0\\) is using rocprofiler-sdk v([0-9]+\\.[0-9]+\\.[0-9]+)"
        FAIL_REGULAR_EXPRESSION
        "${ROCPROFILER_DEFAULT_FAIL_REGEX}|Internal thread for rocprofiler-sdk should not be created"
    )
