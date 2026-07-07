#
# rocprofv3 tool test
#
cmake_minimum_required(VERSION 3.21.0 FATAL_ERROR)

project(
    rocprofiler-sdk-tests-rocattach
    LANGUAGES CXX
    VERSION 0.0.0)

find_package(rocprofiler-sdk REQUIRED)
find_package(rocprofiler-sdk-rocattach REQUIRED)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# TODO: sanitizer tests do not cooperate with attachment code injection
set(ROCPROFILER_MEMCHECK_TYPES "AddressSanitizer" "UndefinedBehaviorSanitizer"
                               "ThreadSanitizer")

if(ROCPROFILER_MEMCHECK AND ROCPROFILER_MEMCHECK IN_LIST ROCPROFILER_MEMCHECK_TYPES)
    set(IS_DISABLED ON)
else()
    set(IS_DISABLED OFF)
endif()

if(ROCPROFILER_MEMCHECK STREQUAL "LeakSanitizer")
    set(LOG_LEVEL "warning") # info produces memory leak
else()
    set(LOG_LEVEL "trace")
endif()

set(rocattach-test-env
    "ROCP_TOOL_ATTACH=1"
    "ROCPROFILER_REGISTER_LOG_LEVEL=${LOG_LEVEL}"
    "ROCPROFILER_LOG_LEVEL=${LOG_LEVEL}"
    "ROCATTACH_LOG_LEVEL=${LOG_LEVEL}"
    "LD_LIBRARY_PATH=$<TARGET_FILE_DIR:rocprofiler-sdk::rocprofiler-sdk-shared-library>:$ENV{LD_LIBRARY_PATH}"
    )

add_executable(rocattach-parallel-attach-test)
target_sources(rocattach-parallel-attach-test PRIVATE main.cpp)
target_link_libraries(
    rocattach-parallel-attach-test
    PRIVATE rocprofiler-sdk::rocprofiler-sdk rocprofiler-sdk::tests-build-flags
            rocprofiler-sdk::tests-common-library
            rocprofiler-sdk::rocprofiler-sdk-rocattach-shared-library)

# Utilize rocattach to use a generic C tool in a basic app
rocprofiler_add_integration_execute_test(
    rocattach-parallel-attach-test-execute
    COMMAND $<TARGET_FILE:rocattach-parallel-attach-test> $<TARGET_FILE:attachment-test>
            $<TARGET_FILE:rocprofiler-sdk-c-tool>
    DEPENDS rocattach-parallel-attach-test
    TIMEOUT 45
    LABELS "attachment"
    ENVIRONMENT "${rocattach-test-env}"
    DISABLED ${IS_DISABLED}
    PRELOAD "${ROCPROFILER_MEMCHECK_PRELOAD_ENV_VALUE}"
    PASS_REGULAR_EXPRESSION "Test C tool \\(priority=0\\) is using rocprofiler-sdk v"
    FAIL_REGULAR_EXPRESSION "${ROCPROFILER_DEFAULT_FAIL_REGEX}")

# Utilize rocattach to verify signals are correctly forwarded to a target application
# during attachment
rocprofiler_add_integration_execute_test(
    rocattach-parallel-attach-test-signal-execute
    COMMAND $<TARGET_FILE:rocattach-parallel-attach-test> $<TARGET_FILE:attachment-test>
            $<TARGET_FILE:rocprofiler-sdk-c-tool> --send-signal
    DEPENDS rocattach-parallel-attach-test
    TIMEOUT 45
    LABELS "attachment"
    ENVIRONMENT "${rocattach-test-env}"
    DISABLED ${IS_DISABLED}
    PRELOAD "${ROCPROFILER_MEMCHECK_PRELOAD_ENV_VALUE}"
    PASS_REGULAR_EXPRESSION "Attachment test process [0-9]+ received signal 28"
    FAIL_REGULAR_EXPRESSION "${ROCPROFILER_DEFAULT_FAIL_REGEX}")

add_executable(rocattach-attach-tree-test)
target_sources(rocattach-attach-tree-test PRIVATE attach_tree.cpp)
target_link_libraries(
    rocattach-attach-tree-test
    PRIVATE rocprofiler-sdk::rocprofiler-sdk rocprofiler-sdk::tests-build-flags
            rocprofiler-sdk::tests-common-library
            rocprofiler-sdk::rocprofiler-sdk-rocattach-shared-library)

rocprofiler_add_integration_execute_test(
    rocattach-attach-tree-test-execute
    TARGET rocattach-attach-tree-test
    ARGS $<TARGET_FILE:attachment-test> $<TARGET_FILE:rocprofiler-sdk-c-tool>
    TIMEOUT 45
    LABELS "attachment"
    ENVIRONMENT "${rocattach-test-env}"
    DISABLED ${IS_DISABLED}
    PRELOAD "${ROCPROFILER_MEMCHECK_PRELOAD_ENV_VALUE}"
    PASS_REGULAR_EXPRESSION "verified: grandchild pid [0-9]+ loaded the tool library"
    FAIL_REGULAR_EXPRESSION "${ROCPROFILER_DEFAULT_FAIL_REGEX}")

# Test that attach fails gracefully when target process lacks ROCP_TOOL_ATTACH=1. Verifies
# that rocattach returns an error status without crashing the target.
add_executable(rocattach-without-env-test)
target_sources(rocattach-without-env-test PRIVATE attach_without_env.cpp)
target_link_libraries(
    rocattach-without-env-test
    PRIVATE rocprofiler-sdk::rocprofiler-sdk rocprofiler-sdk::tests-build-flags
            rocprofiler-sdk::tests-common-library
            rocprofiler-sdk::rocprofiler-sdk-rocattach-shared-library)

# Deliberately omit ROCP_TOOL_ATTACH=1 to test error handling
set(rocattach-without-env-test-env
    "ROCPROFILER_REGISTER_LOG_LEVEL=${LOG_LEVEL}"
    "ROCPROFILER_LOG_LEVEL=${LOG_LEVEL}"
    "ROCATTACH_LOG_LEVEL=${LOG_LEVEL}"
    "LD_LIBRARY_PATH=$<TARGET_FILE_DIR:rocprofiler-sdk::rocprofiler-sdk-shared-library>:$ENV{LD_LIBRARY_PATH}"
    )

rocprofiler_add_integration_execute_test(
    rocattach-without-env-test-execute
    TARGET rocattach-without-env-test
    ARGS $<TARGET_FILE:attachment-test> attach
    TIMEOUT 45
    DISABLED ${IS_DISABLED}
    LABELS "attachment"
    PRELOAD "${PRELOAD_ENV}"
    ENVIRONMENT "${rocattach-without-env-test-env}"
    PASS_REGULAR_EXPRESSION
        "Test PASSED: attach failed gracefully without crashing target"
    FAIL_REGULAR_EXPRESSION "Test FAILED|error: attach succeeded|unexpected attach status"
    )

rocprofiler_add_integration_execute_test(
    rocattach-without-env-tree-test-execute
    TARGET rocattach-without-env-test
    ARGS $<TARGET_FILE:attachment-test> attach-tree
    TIMEOUT 45
    DISABLED ${IS_DISABLED}
    LABELS "attachment"
    PRELOAD "${PRELOAD_ENV}"
    ENVIRONMENT "${rocattach-without-env-test-env}"
    PASS_REGULAR_EXPRESSION
        "Test PASSED: attach failed gracefully without crashing target"
    FAIL_REGULAR_EXPRESSION "Test FAILED|error: attach succeeded|unexpected attach status"
    )
