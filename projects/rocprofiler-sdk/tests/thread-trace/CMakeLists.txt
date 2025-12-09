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
target_link_libraries(
    thread-trace-api-single-test
    PRIVATE rocprofiler-sdk::rocprofiler-sdk rocprofiler-sdk::tests-build-flags
            rocprofiler-sdk::tests-common-library)

rocprofiler_add_integration_execute_test(
    thread-trace-api-single-test
    COMMAND $<TARGET_FILE:thread-trace-api-single-test>
    DEPENDS thread-trace-api-single-test
    TIMEOUT 10
    LABELS "integration-tests"
    PRELOAD "${ROCPROFILER_MEMCHECK_PRELOAD_ENV_VALUE}")
# Multi dispatch test
add_executable(thread-trace-api-multi-test)
target_sources(
    thread-trace-api-multi-test PRIVATE main.cpp trace_callbacks.cpp multi_dispatch.cpp
                                        kernel_branch.cpp kernel_lds.cpp)
target_link_libraries(
    thread-trace-api-multi-test
    PRIVATE rocprofiler-sdk::rocprofiler-sdk rocprofiler-sdk::tests-build-flags
            rocprofiler-sdk::tests-common-library)

rocprofiler_add_integration_execute_test(
    thread-trace-api-multi-test
    COMMAND $<TARGET_FILE:thread-trace-api-multi-test>
    DEPENDS thread-trace-api-multi-test
    TIMEOUT 10
    LABELS "integration-tests"
    PRELOAD "${ROCPROFILER_MEMCHECK_PRELOAD_ENV_VALUE}")

# Agent profiling test
add_executable(thread-trace-api-agent-test)
target_sources(thread-trace-api-agent-test PRIVATE main.cpp trace_callbacks.cpp agent.cpp
                                                   kernel_branch.cpp kernel_lds.cpp)
target_link_libraries(
    thread-trace-api-agent-test
    PRIVATE rocprofiler-sdk::rocprofiler-sdk rocprofiler-sdk::tests-build-flags
            rocprofiler-sdk::tests-common-library)

rocprofiler_add_integration_execute_test(
    thread-trace-api-agent-test
    COMMAND $<TARGET_FILE:thread-trace-api-agent-test>
    DEPENDS thread-trace-api-agent-test
    TIMEOUT 10
    LABELS "integration-tests"
    PRELOAD "${ROCPROFILER_MEMCHECK_PRELOAD_ENV_VALUE}")

# Feature introduced in ROCm 7.1, however some pre-release CI builds for 7.1 dont have
# this feature in aqlprofile. Disable the next tests for ROCm < 7.2.
set(IS_DISABLED True)
find_package(hip CONFIG REQUIRED)
if(hip_FOUND)
    if(${hip_VERSION_MAJOR} EQUAL 7 AND ${hip_VERSION_MINOR} GREATER 1)
        set(IS_DISABLED False)
    elseif(${hip_VERSION_MAJOR} GREATER 7)
        set(IS_DISABLED False)
    endif()
endif()

# Test large buffer sizes. 5120 == 5GB
rocprofiler_add_integration_execute_test(
    thread-trace-api-large-buffer-test
    COMMAND $<TARGET_FILE:thread-trace-api-agent-test>
    DEPENDS thread-trace-api-agent-test
    TIMEOUT 30
    LABELS "integration-tests"
    PRELOAD "${PRELOAD_ENV}"
    ENVIRONMENT "ATT_LARGE_BUFFER_TEST=1"
    DISABLED ${IS_DISABLED})

# Test occupancy mode
rocprofiler_add_integration_execute_test(
    thread-trace-api-extra-args
    COMMAND $<TARGET_FILE:thread-trace-api-agent-test>
    DEPENDS thread-trace-api-agent-test
    TIMEOUT 10
    PRELOAD "${PRELOAD_ENV}"
    ENVIRONMENT "ATT_NODETAIL=1"
    DISABLED ${ROCPROFILER_DISABLE_UNSTABLE_CTESTS})
