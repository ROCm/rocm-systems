#
#
#
cmake_minimum_required(VERSION 3.21.0 FATAL_ERROR)

if(NOT CMAKE_HIP_COMPILER)
    find_program(
        amdclangpp_EXECUTABLE
        NAMES amdclang++
        HINTS ${ROCM_PATH} ENV ROCM_PATH /opt/rocm
        PATHS ${ROCM_PATH} ENV ROCM_PATH /opt/rocm
        PATH_SUFFIXES bin llvm/bin NO_CACHE)
    mark_as_advanced(amdclangpp_EXECUTABLE)

    if(amdclangpp_EXECUTABLE)
        set(CMAKE_HIP_COMPILER "${amdclangpp_EXECUTABLE}")
    endif()
endif()

project(
    rocprofiler-sdk-tests-thread-trace
    LANGUAGES CXX HIP
    VERSION 0.0.0)

find_package(rocprofiler-sdk REQUIRED)
find_package(rocprofiler-sdk-roctx REQUIRED)

set(COMMON_LINK_LIBRARIES
    rocprofiler-sdk::rocprofiler-sdk rocprofiler-sdk::tests-build-flags
    rocprofiler-sdk::tests-common-library rocprofiler-sdk-roctx::rocprofiler-sdk-roctx)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_HIP_STANDARD 17)
set(CMAKE_HIP_EXTENSIONS OFF)
set(CMAKE_HIP_STANDARD_REQUIRED ON)

foreach(_TYPE DEBUG MINSIZEREL RELEASE RELWITHDEBINFO)
    if("${CMAKE_HIP_FLAGS_${_TYPE}}" STREQUAL "")
        set(CMAKE_HIP_FLAGS_${_TYPE} "${CMAKE_CXX_FLAGS_${_TYPE}}")
    endif()
endforeach()

set_source_files_properties(
    kernel_branch.cpp kernel_lds.cpp
    PROPERTIES LANGUAGE HIP COMPILE_FLAGS "${CMAKE_HIP_FLAGS_RELWITHDEBINFO}")
set_source_files_properties(main.cpp PROPERTIES LANGUAGE HIP)

# Single dispatch test
add_executable(thread-trace-api-single-test)
target_sources(
    thread-trace-api-single-test PRIVATE main.cpp trace_callbacks.cpp single_dispatch.cpp
                                         kernel_branch.cpp kernel_lds.cpp)
target_link_libraries(thread-trace-api-single-test PRIVATE ${COMMON_LINK_LIBRARIES})

add_test(NAME thread-trace-api-single-test
         COMMAND $<TARGET_FILE:thread-trace-api-single-test>)

set_tests_properties(
    thread-trace-api-single-test
    PROPERTIES TIMEOUT
               10
               LABELS
               "integration-tests"
               ENVIRONMENT
               "${ROCPROFILER_MEMCHECK_PRELOAD_ENV}"
               FAIL_REGULAR_EXPRESSION
               "${ROCPROFILER_DEFAULT_FAIL_REGEX}"
               DISABLED
               ${ROCPROFILER_DISABLE_UNSTABLE_CTESTS})

# Multi dispatch test
add_executable(thread-trace-api-multi-test)
target_sources(
    thread-trace-api-multi-test PRIVATE main.cpp trace_callbacks.cpp multi_dispatch.cpp
                                        kernel_branch.cpp kernel_lds.cpp)
target_link_libraries(thread-trace-api-multi-test PRIVATE ${COMMON_LINK_LIBRARIES})

add_test(NAME thread-trace-api-multi-test
         COMMAND $<TARGET_FILE:thread-trace-api-multi-test>)

set_tests_properties(
    thread-trace-api-multi-test
    PROPERTIES TIMEOUT
               10
               LABELS
               "integration-tests"
               ENVIRONMENT
               "${ROCPROFILER_MEMCHECK_PRELOAD_ENV}"
               FAIL_REGULAR_EXPRESSION
               "${ROCPROFILER_DEFAULT_FAIL_REGEX}"
               DISABLED
               ${ROCPROFILER_DISABLE_UNSTABLE_CTESTS})

# Agent profiling test
add_executable(thread-trace-api-agent-test)
target_sources(thread-trace-api-agent-test PRIVATE main.cpp trace_callbacks.cpp agent.cpp
                                                   kernel_branch.cpp kernel_lds.cpp)
target_link_libraries(thread-trace-api-agent-test PRIVATE ${COMMON_LINK_LIBRARIES})

add_test(NAME thread-trace-api-agent-test
         COMMAND $<TARGET_FILE:thread-trace-api-agent-test>)

set_tests_properties(
    thread-trace-api-agent-test
    PROPERTIES TIMEOUT
               10
               LABELS
               "integration-tests"
               ENVIRONMENT
               "${ROCPROFILER_MEMCHECK_PRELOAD_ENV}"
               FAIL_REGULAR_EXPRESSION
               "${ROCPROFILER_DEFAULT_FAIL_REGEX}"
               DISABLED
               ${ROCPROFILER_DISABLE_UNSTABLE_CTESTS})

# Test large buffer sizes. 5120 == 5GB
add_test(NAME thread-trace-api-large-buffer-test
         COMMAND $<TARGET_FILE:thread-trace-api-agent-test>)

set_tests_properties(
    thread-trace-api-large-buffer-test
    PROPERTIES TIMEOUT
               10
               LABELS
               "integration-tests"
               ENVIRONMENT
               "${PRELOAD_ENV};ATT_BUFFER_SIZE_MB=5120"
               FAIL_REGULAR_EXPRESSION
               "${ROCPROFILER_DEFAULT_FAIL_REGEX}"
               DISABLED
               ${ROCPROFILER_DISABLE_UNSTABLE_CTESTS})

# Test occupancy mode
add_test(NAME thread-trace-api-extra-args
         COMMAND $<TARGET_FILE:thread-trace-api-agent-test>)

set_tests_properties(
    thread-trace-api-extra-args
    PROPERTIES TIMEOUT
               10
               LABELS
               "integration-tests"
               ENVIRONMENT
               "${PRELOAD_ENV};ATT_NODETAIL=1"
               FAIL_REGULAR_EXPRESSION
               "${ROCPROFILER_DEFAULT_FAIL_REGEX}"
               DISABLED
               ${ROCPROFILER_DISABLE_UNSTABLE_CTESTS})

# Triple buffer tests
add_executable(thread-trace-api-triple-buffer-test)
target_sources(thread-trace-api-triple-buffer-test
               PRIVATE triple_buffer.cpp main.cpp kernel_branch.cpp kernel_lds.cpp)
target_link_libraries(thread-trace-api-triple-buffer-test
                      PRIVATE ${COMMON_LINK_LIBRARIES})

add_test(NAME thread-trace-api-triple-buffer-consistency-test
         COMMAND $<TARGET_FILE:thread-trace-api-triple-buffer-test>)

set_tests_properties(
    thread-trace-api-triple-buffer-consistency-test
    PROPERTIES
        TIMEOUT
        120
        LABELS
        "integration-tests"
        ENVIRONMENT
        "${ROCPROFILER_MEMCHECK_PRELOAD_ENV};TRIPLEBUFFER=1;ATT_NODETAIL=1;ATT_SLOW_CALLBACK=0"
        FAIL_REGULAR_EXPRESSION
        "${ROCPROFILER_DEFAULT_FAIL_REGEX}")

add_test(NAME thread-trace-api-triple-buffer-hammer-test
         COMMAND $<TARGET_FILE:thread-trace-api-triple-buffer-test>)

set_tests_properties(
    thread-trace-api-triple-buffer-hammer-test
    PROPERTIES
        TIMEOUT
        240
        LABELS
        "integration-tests"
        ENVIRONMENT
        "${ROCPROFILER_MEMCHECK_PRELOAD_ENV};TRIPLEBUFFER=1;ATT_NODETAIL=0;ATT_SLOW_CALLBACK=0"
        FAIL_REGULAR_EXPRESSION
        "${ROCPROFILER_DEFAULT_FAIL_REGEX}")

add_test(NAME thread-trace-api-triple-buffer-slow-test
         COMMAND $<TARGET_FILE:thread-trace-api-triple-buffer-test>)

set_tests_properties(
    thread-trace-api-triple-buffer-slow-test
    PROPERTIES
        TIMEOUT
        120
        LABELS
        "integration-tests"
        ENVIRONMENT
        "${ROCPROFILER_MEMCHECK_PRELOAD_ENV};TRIPLEBUFFER=1;ATT_NODETAIL=1;ATT_SLOW_CALLBACK=1"
        FAIL_REGULAR_EXPRESSION
        "${ROCPROFILER_DEFAULT_FAIL_REGEX}")
