#
# Range replay declines on hardware. Each test drives the application's range into one decline
# reason and checks the two things a decline promises: CLOSE names the reason before any pass
# runs, and the application's own execution of the range is unchanged.
#
# A test is one RR_DECLINE_MODE of the application paired with the status the tool expects.
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

project(rocprofiler-sdk-tests-range-replay-declines LANGUAGES CXX HIP)

foreach(_TYPE DEBUG MINSIZEREL RELEASE RELWITHDEBINFO)
    if("${CMAKE_HIP_FLAGS_${_TYPE}}" STREQUAL "")
        set(CMAKE_HIP_FLAGS_${_TYPE} "${CMAKE_CXX_FLAGS_${_TYPE}}")
    endif()
endforeach()

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_HIP_STANDARD 17)
set(CMAKE_HIP_EXTENSIONS OFF)
set(CMAKE_HIP_STANDARD_REQUIRED ON)

find_package(rocprofiler-sdk REQUIRED COMPONENTS headers shared-library)
find_package(Threads REQUIRED)
find_package(
    hsa-runtime64
    REQUIRED
    CONFIG
    HINTS
    ${rocm_version_DIR}
    ${ROCM_PATH}
    PATHS
    ${rocm_version_DIR}
    ${ROCM_PATH})

add_library(range-replay-declines-client SHARED)
target_sources(range-replay-declines-client PRIVATE client.cpp range.hpp)
target_include_directories(range-replay-declines-client
                           PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(
    range-replay-declines-client PRIVATE rocprofiler-sdk::rocprofiler-sdk
                                         rocprofiler-sdk::tests-build-flags)

# The application resolves the range replay entry points with dlsym instead of linking the
# SDK: linking rocprofiler-sdk into a HIP executable makes HIP report
# hipErrorInvalidDeviceFunction on the first launch on gfx942, the same reason the range
# replay samples do it that way. It links the HSA runtime for the batch copy case, and
# resolves hsa_amd_memory_async_batch_copy itself with dlsym so a runtime without it skips
# rather than failing to load.
set_source_files_properties(main.cpp PROPERTIES LANGUAGE HIP)
add_executable(range-replay-declines-testapp)
target_sources(range-replay-declines-testapp PRIVATE main.cpp range.hpp)
target_include_directories(range-replay-declines-testapp
                           PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(
    range-replay-declines-testapp
    PRIVATE rocprofiler-sdk::rocprofiler-sdk-headers rocprofiler-sdk::tests-build-flags
            hsa-runtime64::hsa-runtime64 Threads::Threads ${CMAKE_DL_LIBS})
add_dependencies(range-replay-declines-testapp range-replay-declines-client)

# The tool's verdict alone would let a run pass while the application reported a wrong
# result, so the application's own failure line fails the test too.
function(add_rr_decline_test NAME)
    cmake_parse_arguments(arg "" "MODE;EXPECT" "" ${ARGN})
    rocprofiler_add_integration_execute_test(
        ${NAME}
        COMMAND $<TARGET_FILE:range-replay-declines-testapp>
        DEPENDS range-replay-declines-client
        TIMEOUT 120
        LABELS "integration-tests;range-replay"
        PRELOAD "$<TARGET_FILE:range-replay-declines-client>"
        ENVIRONMENT "RR_DECLINE_MODE=${arg_MODE}" "RR_DECLINE_EXPECT=${arg_EXPECT}"
        PASS_REGULAR_EXPRESSION "\\[rr-decline\\] PASS"
        FAIL_REGULAR_EXPRESSION
            "\\[rr-decline\\] FAIL|\\[rr-decline\\] error|\\[app\\] FAIL|${ROCPROFILER_DEFAULT_FAIL_REGEX}"
        SKIP_REGULAR_EXPRESSION "\\[app\\] SKIP")
endfunction()

# C1: a host-to-device hipMemcpyAsync between the range's two kernels, with no memory-copy
# tracing configured anywhere in the process.
add_rr_decline_test(test-range-replay-decline-hip-memcpy-async MODE hip-memcpy-async
                    EXPECT MEMORY_COPY_IN_RANGE)

# C2: the same write through hsa_amd_memory_async_batch_copy, an HSA entry point
# intercepted separately from the other copy functions. Skipped where the runtime does not
# provide it.
add_rr_decline_test(test-range-replay-decline-hsa-batch-copy MODE hsa-batch-copy EXPECT
                    MEMORY_COPY_IN_RANGE)
