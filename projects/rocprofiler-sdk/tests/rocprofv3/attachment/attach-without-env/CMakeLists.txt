#
# rocprofv3 attachment test without ROCP_TOOL_ATTACH
#
cmake_minimum_required(VERSION 3.21.0 FATAL_ERROR)

project(
    rocprofiler-sdk-tests-rocprofv3-attachment-attach-without-env
    LANGUAGES CXX
    VERSION 0.0.0)

find_package(rocprofiler-sdk REQUIRED)

set(ROCPROFILER_MEMCHECK_TYPES "AddressSanitizer" "UndefinedBehaviorSanitizer"
                               "ThreadSanitizer")

if(ROCPROFILER_MEMCHECK AND ROCPROFILER_MEMCHECK IN_LIST ROCPROFILER_MEMCHECK_TYPES)
    set(IS_DISABLED ON)
else()
    set(IS_DISABLED OFF)
endif()

string(REPLACE "LD_PRELOAD=" "ROCPROF_PRELOAD=" PRELOAD_ENV
               "${ROCPROFILER_MEMCHECK_PRELOAD_ENV}")

set(attachment-env
    "${PRELOAD_ENV}"
    "LD_LIBRARY_PATH=$<TARGET_FILE_DIR:rocprofiler-sdk::rocprofiler-sdk-shared-library>:$ENV{LD_LIBRARY_PATH}"
    )

add_test(
    NAME rocprofv3-test-attachment-without-env-execute
    COMMAND
        bash ${CMAKE_CURRENT_SOURCE_DIR}/run_attachment_without_env_test.sh
        $<TARGET_FILE:attachment-test> $<TARGET_FILE:rocprofiler-sdk::rocprofv3>
        ${CMAKE_CURRENT_BINARY_DIR})

set_tests_properties(
    rocprofv3-test-attachment-without-env-execute
    PROPERTIES TIMEOUT 30 LABELS "integration-tests;attachment" ENVIRONMENT
               "${attachment-env}" DISABLED "${IS_DISABLED}")
