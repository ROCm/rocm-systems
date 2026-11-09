#
# Callback-driven host range replay (rocprofiler_range_replay_host_run) integration tests.
#
# main.cpp runs an order-dependent computation as one range through the C++ host_run
# wrapper: per step an affine update, a table mix and a block reduction on two streams
# joined by events, a device-to-host copy of the reduction, and a host computation that
# derives the next step's kernel arguments and mix table, copied back to the device. The
# application checks every pass and the state after the range bit for bit against a host
# reference of one execution; the tool checks the CLOSE status and the PASS sequence.
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

project(rocprofiler-sdk-tests-range-replay-host-api LANGUAGES CXX HIP)

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

add_library(range-replay-host-api-client SHARED)
target_sources(range-replay-host-api-client PRIVATE client.cpp range.hpp)
target_include_directories(range-replay-host-api-client
                           PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(
    range-replay-host-api-client PRIVATE rocprofiler-sdk::rocprofiler-sdk
                                         rocprofiler-sdk::tests-build-flags)

# The application does not link the SDK, for the same reason as the range replay samples
# (linking rocprofiler-sdk into a HIP executable breaks the first launch on gfx942).
# Unlike range-replay-host, it calls the range replay API by name, through the header's
# C++ host_run wrapper, so those references are left for the dynamic loader to resolve
# from the SDK the preloaded tool brings in. Every test of this executable preloads the
# tool.
set_source_files_properties(main.cpp PROPERTIES LANGUAGE HIP)
add_executable(range-replay-host-api-testapp)
target_sources(range-replay-host-api-testapp PRIVATE main.cpp range.hpp)
target_include_directories(range-replay-host-api-testapp
                           PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(
    range-replay-host-api-testapp PRIVATE rocprofiler-sdk::rocprofiler-sdk-headers
                                          rocprofiler-sdk::tests-build-flags)
target_link_options(range-replay-host-api-testapp PRIVATE
                    "LINKER:--unresolved-symbols=ignore-in-object-files")
add_dependencies(range-replay-host-api-testapp range-replay-host-api-client)

if(ROCPROFILER_MEMCHECK_PRELOAD_ENV_VALUE)
    set(PRELOAD_ENV
        "${ROCPROFILER_MEMCHECK_PRELOAD_ENV_VALUE}:$<TARGET_FILE:range-replay-host-api-client>"
        )
else()
    set(PRELOAD_ENV "$<TARGET_FILE:range-replay-host-api-client>")
endif()

function(add_rr_host_api_test NAME)
    cmake_parse_arguments(arg "" "" "ENVIRONMENT" ${ARGN})
    rocprofiler_add_integration_execute_test(
        ${NAME}
        COMMAND $<TARGET_FILE:range-replay-host-api-testapp>
        DEPENDS range-replay-host-api-client
        TIMEOUT 120
        LABELS "integration-tests;range-replay"
        PRELOAD "${PRELOAD_ENV}"
        ENVIRONMENT "ROCPROF_RANGE_REPLAY_VERIFY=1" ${arg_ENVIRONMENT}
        PASS_REGULAR_EXPRESSION "\\[rr-api\\] PASS"
        FAIL_REGULAR_EXPRESSION
            "\\[rr-api\\] FAIL|\\[rr-api\\] error|\\[app\\] FAIL|${ROCPROFILER_DEFAULT_FAIL_REGEX}"
        )
endfunction()

# Host replay through host_run, four passes.
add_rr_host_api_test(test-range-replay-host-api ENVIRONMENT
                     "RR_API_PASSES=4;RR_API_EXPECT_PASSES=4")

# The tool leaves host_replay unset: the body runs once and packet replay declines it.
add_rr_host_api_test(test-range-replay-host-api-packet ENVIRONMENT
                     "RR_API_TOOL_MODE=packet;RR_API_PASSES=4;RR_API_EXPECT_PASSES=1")

# The body throws on replayed pass 1: the range closes BODY_EXCEPTION and the exception
# reaches main.
add_rr_host_api_test(test-range-replay-host-api-throw ENVIRONMENT
                     "RR_API_PASSES=4;RR_API_THROW_PASS=1")
