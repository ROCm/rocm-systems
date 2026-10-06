#
# Range replay with per-dispatch services: per-pass service selection integration tests.
#
# The range replay counterpart of kernel-replay-pass-selection. A range of three dispatches
# is replayed N times with a per-dispatch service enabled, and the tool's own dispatch
# callback for that service decides on which replayed passes it collects. Every replayed
# dispatch re-enters the queue interceptor, so these check that each service's enter and
# exit hooks consult the tool on every pass and attribute the result to that pass: the
# exact number of records per service and pass, that the range really was replayed, and
# that no replayed pass diverged from the application's own run.
#
# Pass 0 is the application's own execution and always collects; the selection applies to
# the replayed passes 1..N-1 only.
#
# Combinations not exercised here, for the same reasons as kernel-replay-pass-selection:
#   - counters + PC sampling: they cannot be configured together
#   - counters + SPM: both use SQ hardware on the same dispatch
#   - ATT + SPM: both inject AQL packets around the dispatch
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

project(rocprofiler-sdk-tests-range-replay-pass-selection LANGUAGES CXX HIP)

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

# The `headers` component lets the application place its range boundaries against the real
# types without linking the SDK; see the target_link_libraries comment below.
find_package(rocprofiler-sdk REQUIRED COMPONENTS headers shared-library)
find_package(Threads REQUIRED)

add_library(range-replay-pass-selection-client SHARED)
target_sources(range-replay-pass-selection-client PRIVATE client.cpp range.hpp)
target_include_directories(range-replay-pass-selection-client
                           PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(
    range-replay-pass-selection-client PRIVATE rocprofiler-sdk::rocprofiler-sdk
                                               rocprofiler-sdk::tests-build-flags)

# The application resolves the range replay entry points with dlsym instead of linking the
# SDK: linking rocprofiler-sdk into a HIP executable makes HIP report
# hipErrorInvalidDeviceFunction on the first launch on gfx942, the same reason the range
# replay samples do it that way.
set_source_files_properties(main.cpp PROPERTIES LANGUAGE HIP)
add_executable(range-replay-pass-selection-testapp)
target_sources(range-replay-pass-selection-testapp PRIVATE main.cpp range.hpp)
target_include_directories(range-replay-pass-selection-testapp
                           PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(
    range-replay-pass-selection-testapp
    PRIVATE rocprofiler-sdk::rocprofiler-sdk-headers rocprofiler-sdk::tests-build-flags
            Threads::Threads ${CMAKE_DL_LIBS})
add_dependencies(range-replay-pass-selection-testapp range-replay-pass-selection-client)

if(ROCPROFILER_MEMCHECK_PRELOAD_ENV_VALUE)
    set(PRELOAD_ENV
        "${ROCPROFILER_MEMCHECK_PRELOAD_ENV_VALUE}:$<TARGET_FILE:range-replay-pass-selection-client>"
        )
else()
    set(PRELOAD_ENV "$<TARGET_FILE:range-replay-pass-selection-client>")
endif()

include(rocprofiler-sdk-utilities)
rocprofiler_sdk_spm_disabled(IS_SPM_DISABLED)
rocprofiler_sdk_pc_sampling_disabled(IS_PC_SAMPLING_DISABLED)

# The pass expression alone would let a run pass on the tool's verdict while the
# application reported a wrong result, so the application's own failure line fails the
# test too.
function(add_rr_ps_test NAME)
    cmake_parse_arguments(arg "" "DISABLED" "ENVIRONMENT" ${ARGN})
    set(_disabled)
    if(arg_DISABLED)
        set(_disabled DISABLED ${arg_DISABLED})
    endif()
    rocprofiler_add_integration_execute_test(
        ${NAME}
        COMMAND $<TARGET_FILE:range-replay-pass-selection-testapp>
        DEPENDS range-replay-pass-selection-client
        TIMEOUT 120
        LABELS "integration-tests;range-replay"
        PRELOAD "${PRELOAD_ENV}"
        ENVIRONMENT "ROCPROF_RANGE_REPLAY_VERIFY=1" ${arg_ENVIRONMENT} ${_disabled}
        PASS_REGULAR_EXPRESSION "\\[rr-ps\\] PASS"
        FAIL_REGULAR_EXPRESSION
            "\\[rr-ps\\] FAIL|\\[rr-ps\\] error|\\[app\\] FAIL|${ROCPROFILER_DEFAULT_FAIL_REGEX}"
        SKIP_REGULAR_EXPRESSION "ATT unavailable|SPM unavailable|PC sampling unavailable")
endfunction()

# Counters on every pass: three dispatches x four passes.
add_rr_ps_test(test-range-replay-pass-selection-counters-all-passes ENVIRONMENT
               "RR_PS_SERVICES=counters;RR_PS_PASSES=4;RR_PS_STOP_PASS=-1")

# Counters stop from replayed pass 1: only the application's own run collects.
add_rr_ps_test(test-range-replay-pass-selection-counters-stop-after-0 ENVIRONMENT
               "RR_PS_SERVICES=counters;RR_PS_PASSES=4;RR_PS_STOP_PASS=1")

# Counters stop on pass 1 and collect again from pass 2: passes 0, 2 and 3 collect.
add_rr_ps_test(
    test-range-replay-pass-selection-counters-stop-then-restart ENVIRONMENT
    "RR_PS_SERVICES=counters;RR_PS_PASSES=4;RR_PS_STOP_PASS=1;RR_PS_START_PASS=2")

# ATT traces every dispatch of every pass.
add_rr_ps_test(test-range-replay-pass-selection-att-all-passes ENVIRONMENT
               "RR_PS_SERVICES=att;RR_PS_PASSES=4;RR_PS_STOP_PASS=-1")

# ATT stops from replayed pass 1: only the application's own run is traced.
add_rr_ps_test(test-range-replay-pass-selection-att-stop-after-0 ENVIRONMENT
               "RR_PS_SERVICES=att;RR_PS_PASSES=4;RR_PS_STOP_PASS=1")

# SPM: the same dispatch-scoped contract as counters.
add_rr_ps_test(
    test-range-replay-pass-selection-spm-all-passes
    ENVIRONMENT
    "RR_PS_SERVICES=spm;RR_PS_PASSES=4;RR_PS_STOP_PASS=-1;ROCPROFILER_SPM_BETA_ENABLED=True"
    DISABLED
    ${IS_SPM_DISABLED})

add_rr_ps_test(
    test-range-replay-pass-selection-spm-stop-after-0
    ENVIRONMENT
    "RR_PS_SERVICES=spm;RR_PS_PASSES=4;RR_PS_STOP_PASS=1;ROCPROFILER_SPM_BETA_ENABLED=True"
    DISABLED
    ${IS_SPM_DISABLED})

# PC sampling is agent-wide and has no dispatch callback, so it samples every pass; this
# checks that the range still replays under it.
add_rr_ps_test(
    test-range-replay-pass-selection-pc-sampling-every-pass ENVIRONMENT
    "RR_PS_SERVICES=pc-sampling;RR_PS_PASSES=4" DISABLED ${IS_PC_SAMPLING_DISABLED})

# Counters every pass alongside ATT on the application's own run only.
add_rr_ps_test(
    test-range-replay-pass-selection-counters-att ENVIRONMENT
    "RR_PS_SERVICES=counters,att;RR_PS_KEEP=counters;RR_PS_PASSES=4;RR_PS_STOP_PASS=1")
