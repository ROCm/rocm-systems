#
#
#
cmake_minimum_required(VERSION 3.21.0 FATAL_ERROR)

project(
    rocprofiler-sdk-tests-c-tool
    LANGUAGES CXX
    VERSION 0.0.0)

find_package(rocprofiler-sdk REQUIRED)

if(ROCPROFILER_MEMCHECK_PRELOAD_ENV_VALUE)
    set(PRELOAD_ENV
        "${ROCPROFILER_MEMCHECK_PRELOAD_ENV_VALUE}:$<TARGET_FILE:rocprofiler-sdk-c-tool>")
else()
    set(PRELOAD_ENV "$<TARGET_FILE:rocprofiler-sdk-c-tool>")
endif()

rocprofiler_add_integration_execute_test(
    test-c-tool
    COMMAND $<TARGET_FILE:transpose> 1
    DEPENDS transpose
    TIMEOUT 45
    LABELS "integration-tests"
    PRELOAD "${PRELOAD_ENV}"
    ENVIRONMENT
        "LD_LIBRARY_PATH=$<TARGET_FILE_DIR:rocprofiler-sdk::rocprofiler-sdk-shared-library>:$ENV{LD_LIBRARY_PATH}"
    PASS_REGULAR_EXPRESSION
        "Test C tool \\(priority=0\\) is using rocprofiler-sdk v([0-9]+\\.[0-9]+\\.[0-9]+)"
    FAIL_REGULAR_EXPRESSION
        "${ROCPROFILER_DEFAULT_FAIL_REGEX}|Internal thread for rocprofiler-sdk should not be created"
    )

# this test uses ROCP_TOOL_LIBRARIES instead of LD_PRELOAD
rocprofiler_add_integration_execute_test(
    test-c-tool-rocp-tool-lib
    COMMAND $<TARGET_FILE:transpose> 1
    DEPENDS transpose rocprofiler-sdk-c-tool
    TIMEOUT 45
    LABELS "integration-tests"
    PRELOAD "${ROCPROFILER_MEMCHECK_PRELOAD_ENV_VALUE}"
    ENVIRONMENT
        "ROCP_TOOL_LIBRARIES=$<TARGET_FILE:rocprofiler-sdk-c-tool>"
        "LD_LIBRARY_PATH=$<TARGET_FILE_DIR:rocprofiler-sdk::rocprofiler-sdk-shared-library>:$ENV{LD_LIBRARY_PATH}"
    PASS_REGULAR_EXPRESSION
        "Test C tool \\(priority=0\\) is using rocprofiler-sdk v([0-9]+\\.[0-9]+\\.[0-9]+)"
    FAIL_REGULAR_EXPRESSION
        "${ROCPROFILER_DEFAULT_FAIL_REGEX}|Internal thread for rocprofiler-sdk should not be created"
    )
