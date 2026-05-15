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

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

add_executable(rocattach-parallel-attach-test)
target_sources(rocattach-parallel-attach-test PRIVATE main.cpp)
target_link_libraries(
    rocattach-parallel-attach-test
    PRIVATE rocprofiler-sdk::rocprofiler-sdk rocprofiler-sdk::tests-build-flags
            rocprofiler-sdk::tests-common-library
            rocprofiler-sdk::rocprofiler-sdk-rocattach-shared-library)

add_test(NAME rocattach-parallel-attach-test-execute
         COMMAND $<TARGET_FILE:rocattach-parallel-attach-test>
                 $<TARGET_FILE:attachment-test> $<TARGET_FILE:rocprofiler-sdk-c-tool>)

set(rocattach-test-env
    "${ROCPROFILER_MEMCHECK_PRELOAD_ENV}"
    "ROCP_TOOL_ATTACH=1"
    "ROCPROFILER_REGISTER_LOG_LEVEL=${LOG_LEVEL}"
    "ROCPROFILER_LOG_LEVEL=${LOG_LEVEL}"
    "ROCATTACH_LOG_LEVEL=${LOG_LEVEL}"
    "LD_LIBRARY_PATH=$<TARGET_FILE_DIR:rocprofiler-sdk::rocprofiler-sdk-shared-library>:$ENV{LD_LIBRARY_PATH}"
    )

set_tests_properties(
    rocattach-parallel-attach-test-execute
    PROPERTIES TIMEOUT
               45
               LABELS
               "integration-tests"
               ENVIRONMENT
               "${rocattach-test-env}"
               PASS_REGULAR_EXPRESSION
               "Test C tool \\(priority=0\\) is using rocprofiler-sdk v"
               FAIL_REGULAR_EXPRESSION
               "${ROCPROFILER_DEFAULT_FAIL_REGEX}"
               DISABLED
               "${IS_DISABLED}")
