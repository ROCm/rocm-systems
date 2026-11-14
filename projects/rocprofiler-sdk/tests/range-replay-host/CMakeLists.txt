#
# Host range replay (sigsetjmp / siglongjmp) integration tests.
#
# main.cpp runs a time-stepping solver inside one range opened with the host-begin
# protocol: per step an FFT / spectral filter / inverse FFT, a stencil, Jacobi sweeps, a
# device-to-device copy, a memset, a device-to-host residual the host reduces, and a
# host-computed boundary row copied back for the next step, on two streams joined by
# events. Packet replay cannot re-execute any of that; host replay re-runs the host code
# for every pass from the restored entry state. The application checks that every pass
# ends in the same device and host state and that registered host state was rewound; the
# tool checks the range's CLOSE status, the PASS sequence, the divergence count, and the
# exact number of counter records (with an optional local stop).
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

project(rocprofiler-sdk-tests-range-replay-host LANGUAGES CXX HIP)

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

add_library(range-replay-host-timestep-client SHARED)
target_sources(range-replay-host-timestep-client PRIVATE client.cpp range.hpp)
target_include_directories(range-replay-host-timestep-client
                           PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(
    range-replay-host-timestep-client PRIVATE rocprofiler-sdk::rocprofiler-sdk
                                              rocprofiler-sdk::tests-build-flags)

# The application resolves the range replay entry points with dlsym instead of linking the
# SDK, for the same reason as the range replay samples (linking rocprofiler-sdk into a HIP
# executable breaks the first launch on gfx942).
set_source_files_properties(main.cpp PROPERTIES LANGUAGE HIP)
add_executable(range-replay-host-timestep-testapp)
target_sources(range-replay-host-timestep-testapp PRIVATE main.cpp range.hpp)
target_include_directories(range-replay-host-timestep-testapp
                           PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(
    range-replay-host-timestep-testapp
    PRIVATE rocprofiler-sdk::rocprofiler-sdk-headers rocprofiler-sdk::tests-build-flags
            Threads::Threads ${CMAKE_DL_LIBS})
add_dependencies(range-replay-host-timestep-testapp range-replay-host-timestep-client)

if(ROCPROFILER_MEMCHECK_PRELOAD_ENV_VALUE)
    set(PRELOAD_ENV
        "${ROCPROFILER_MEMCHECK_PRELOAD_ENV_VALUE}:$<TARGET_FILE:range-replay-host-timestep-client>"
        )
else()
    set(PRELOAD_ENV "$<TARGET_FILE:range-replay-host-timestep-client>")
endif()

function(add_rr_host_test NAME)
    cmake_parse_arguments(arg "" "" "ENVIRONMENT" ${ARGN})
    rocprofiler_add_integration_execute_test(
        ${NAME}
        COMMAND $<TARGET_FILE:range-replay-host-timestep-testapp>
        DEPENDS range-replay-host-timestep-client
        TIMEOUT 180
        LABELS "integration-tests;range-replay"
        PRELOAD "${PRELOAD_ENV}"
        ENVIRONMENT "ROCPROF_RANGE_REPLAY_VERIFY=1" ${arg_ENVIRONMENT}
        PASS_REGULAR_EXPRESSION "\\[rr-host\\] PASS"
        FAIL_REGULAR_EXPRESSION
            "\\[rr-host\\] FAIL|\\[rr-host\\] error|\\[app\\] FAIL|${ROCPROFILER_DEFAULT_FAIL_REGEX}"
        )
endfunction()

# The time-stepping range, four passes, counters on every pass.
add_rr_host_test(test-range-replay-host-timestep ENVIRONMENT
                 "RR_HOST_PASSES=4;RR_HOST_EXPECT_PASSES=4")

# The same with a HIP graph launched inside every time step.
add_rr_host_test(test-range-replay-host-timestep-graph ENVIRONMENT
                 "RR_HOST_PASSES=4;RR_HOST_EXPECT_PASSES=4;RR_HOST_GRAPH=1")

# The same with a scratch buffer allocated and freed inside every time step.
add_rr_host_test(test-range-replay-host-timestep-scratch-alloc ENVIRONMENT
                 "RR_HOST_PASSES=4;RR_HOST_EXPECT_PASSES=4;RR_HOST_SCRATCH=1")

# Open-ended pass count, stopped by replay_continue_cb after five passes.
add_rr_host_test(test-range-replay-host-continue-cb ENVIRONMENT
                 "RR_HOST_PASSES=5;RR_HOST_EXPECT_PASSES=5;RR_HOST_CONTINUE=1")

# Counters locally stopped on replayed pass 2: passes 0 and 1 collect.
add_rr_host_test(test-range-replay-host-local-stop ENVIRONMENT
                 "RR_HOST_PASSES=4;RR_HOST_EXPECT_PASSES=4;RR_HOST_STOP_PASS=2")

# The same range under packet replay is declined and runs once.
add_rr_host_test(test-range-replay-host-packet-declines ENVIRONMENT
                 "RR_HOST_TOOL_MODE=packet;RR_HOST_PASSES=4;RR_HOST_EXPECT_PASSES=1")

# Host replay requested, but the application did not arm the jump buffer: declined, runs
# once.
add_rr_host_test(
    test-range-replay-host-unarmed-declines
    ENVIRONMENT
    "RR_HOST_TOOL_MODE=host-unarmed;RR_HOST_APP_ARM=0;RR_HOST_PASSES=4;RR_HOST_EXPECT_PASSES=1"
    )
