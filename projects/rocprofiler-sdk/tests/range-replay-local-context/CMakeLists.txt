#
# Range replay with per-dispatch services: localized start/stop integration tests.
#
# The range replay counterpart of kernel-replay-local-context. A range of three dispatches
# is replayed N times with a per-dispatch service enabled, and the tool locally
# starts/stops that service's context across the replayed passes. Every replayed dispatch
# re-enters the queue interceptor, so these check that each service's enter and exit hooks,
# and the localized toggles they consult, hold up under range replay: the exact number of
# records per service, that the range really was replayed, and that no replayed pass
# diverged from the application's own run.
#
# Pass 0 is the application's own execution and always collects; toggles apply to the
# replayed passes 1..N-1 only.
#
# Combinations not exercised here, for the same reasons as kernel-replay-local-context:
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

project(rocprofiler-sdk-tests-range-replay-local-context LANGUAGES CXX HIP)

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

add_library(range-replay-local-context-client SHARED)
target_sources(range-replay-local-context-client PRIVATE client.cpp range.hpp)
target_include_directories(range-replay-local-context-client
                           PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(
    range-replay-local-context-client PRIVATE rocprofiler-sdk::rocprofiler-sdk
                                              rocprofiler-sdk::tests-build-flags)

# The application resolves the range replay entry points with dlsym instead of linking the
# SDK: linking rocprofiler-sdk into a HIP executable makes HIP report
# hipErrorInvalidDeviceFunction on the first launch on gfx942, the same reason the range
# replay samples do it that way.
set_source_files_properties(main.cpp PROPERTIES LANGUAGE HIP)
add_executable(range-replay-local-context-testapp)
target_sources(range-replay-local-context-testapp PRIVATE main.cpp range.hpp)
target_include_directories(range-replay-local-context-testapp
                           PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(
    range-replay-local-context-testapp
    PRIVATE rocprofiler-sdk::rocprofiler-sdk-headers rocprofiler-sdk::tests-build-flags
            Threads::Threads ${CMAKE_DL_LIBS})
add_dependencies(range-replay-local-context-testapp range-replay-local-context-client)

if(ROCPROFILER_MEMCHECK_PRELOAD_ENV_VALUE)
    set(PRELOAD_ENV
        "${ROCPROFILER_MEMCHECK_PRELOAD_ENV_VALUE}:$<TARGET_FILE:range-replay-local-context-client>"
        )
else()
    set(PRELOAD_ENV "$<TARGET_FILE:range-replay-local-context-client>")
endif()

include(rocprofiler-sdk-utilities)
rocprofiler_sdk_spm_disabled(IS_SPM_DISABLED)
rocprofiler_sdk_pc_sampling_disabled(IS_PC_SAMPLING_DISABLED)

# The pass expression alone would let a run pass on the tool's verdict while the
# application reported a wrong result, so the application's own failure line fails the
# test too.
function(add_rr_lc_test NAME)
    cmake_parse_arguments(arg "" "DISABLED" "ENVIRONMENT" ${ARGN})
    set(_disabled)
    if(arg_DISABLED)
        set(_disabled DISABLED ${arg_DISABLED})
    endif()
    rocprofiler_add_integration_execute_test(
        ${NAME}
        COMMAND $<TARGET_FILE:range-replay-local-context-testapp>
        DEPENDS range-replay-local-context-client
        TIMEOUT 120
        LABELS "integration-tests;range-replay"
        PRELOAD "${PRELOAD_ENV}"
        ENVIRONMENT "ROCPROF_RANGE_REPLAY_VERIFY=1" ${arg_ENVIRONMENT} ${_disabled}
        PASS_REGULAR_EXPRESSION "\\[rr-lc\\] PASS"
        FAIL_REGULAR_EXPRESSION
            "\\[rr-lc\\] FAIL|\\[rr-lc\\] error|\\[app\\] FAIL|${ROCPROFILER_DEFAULT_FAIL_REGEX}"
        SKIP_REGULAR_EXPRESSION "ATT unavailable|SPM unavailable|PC sampling unavailable")
endfunction()

# Counters on every pass: three dispatches x four passes.
add_rr_lc_test(test-range-replay-local-context-counters-all-passes ENVIRONMENT
               "RR_LC_SERVICES=counters;RR_LC_PASSES=4;RR_LC_STOP_PASS=-1")

# Counters stopped on replayed pass 1: only the application's own run collects.
add_rr_lc_test(test-range-replay-local-context-counters-stop-after-0 ENVIRONMENT
               "RR_LC_SERVICES=counters;RR_LC_PASSES=4;RR_LC_STOP_PASS=1")

# Counters stopped on pass 1, started again on pass 2: passes 0, 2 and 3 collect.
add_rr_lc_test(
    test-range-replay-local-context-counters-stop-then-restart ENVIRONMENT
    "RR_LC_SERVICES=counters;RR_LC_PASSES=4;RR_LC_STOP_PASS=1;RR_LC_START_PASS=2")

# ATT traces every dispatch of every pass.
add_rr_lc_test(test-range-replay-local-context-att-all-passes ENVIRONMENT
               "RR_LC_SERVICES=att;RR_LC_PASSES=4;RR_LC_STOP_PASS=-1")

# ATT stopped on replayed pass 1: only the application's own run is traced.
add_rr_lc_test(test-range-replay-local-context-att-stop-after-0 ENVIRONMENT
               "RR_LC_SERVICES=att;RR_LC_PASSES=4;RR_LC_STOP_PASS=1")

# SPM: the same dispatch-scoped contract as counters.
add_rr_lc_test(
    test-range-replay-local-context-spm-all-passes
    ENVIRONMENT
    "RR_LC_SERVICES=spm;RR_LC_PASSES=4;RR_LC_STOP_PASS=-1;ROCPROFILER_SPM_BETA_ENABLED=True"
    DISABLED
    ${IS_SPM_DISABLED})

add_rr_lc_test(
    test-range-replay-local-context-spm-stop-after-0
    ENVIRONMENT
    "RR_LC_SERVICES=spm;RR_LC_PASSES=4;RR_LC_STOP_PASS=1;ROCPROFILER_SPM_BETA_ENABLED=True"
    DISABLED
    ${IS_SPM_DISABLED})

# PC sampling: a local stop must succeed. Collection is agent-wide and ignores the
# override, as it does under kernel replay.
add_rr_lc_test(
    test-range-replay-local-context-pc-sampling-stop-after-0 ENVIRONMENT
    "RR_LC_SERVICES=pc-sampling;RR_LC_PASSES=4;RR_LC_STOP_PASS=1" DISABLED
    ${IS_PC_SAMPLING_DISABLED})

# Counters every pass alongside ATT on the application's own run only.
add_rr_lc_test(
    test-range-replay-local-context-counters-att ENVIRONMENT
    "RR_LC_SERVICES=counters,att;RR_LC_KEEP=counters;RR_LC_PASSES=4;RR_LC_STOP_PASS=1")
