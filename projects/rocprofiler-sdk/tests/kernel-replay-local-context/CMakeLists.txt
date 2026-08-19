#
# Kernel-replay localized context start/stop integration tests.
#
# A dedicated tool replays one distinctive HIP dispatch N times and locally
# starts/stops per-service contexts across passes. Individual services are the
# priority; one counters+ATT combination covers the design's main example
# (counters every pass, ATT once).
#
# Combinations that are not exercised here, by design:
#   - counters + PC sampling: Constraint5, they cannot be configured together
#   - counters + SPM: both use SQ hardware on the same dispatch
#   - ATT + SPM: both inject AQL packets around the dispatch
#   - 3-/4-way mixes inherit those conflicts
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

project(rocprofiler-sdk-tests-kernel-replay-local-context LANGUAGES CXX HIP)

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

find_package(rocprofiler-sdk REQUIRED)

add_library(kernel-replay-local-context-client SHARED)
target_sources(kernel-replay-local-context-client PRIVATE client.cpp)
target_link_libraries(
    kernel-replay-local-context-client PRIVATE rocprofiler-sdk::rocprofiler-sdk
                                               rocprofiler-sdk::tests-build-flags)

set_source_files_properties(main.cpp PROPERTIES LANGUAGE HIP)
add_executable(kernel-replay-local-context-testapp)
target_sources(kernel-replay-local-context-testapp PRIVATE main.cpp)
target_link_libraries(kernel-replay-local-context-testapp
                      PRIVATE rocprofiler-sdk::tests-build-flags)

if(ROCPROFILER_MEMCHECK_PRELOAD_ENV_VALUE)
    set(PRELOAD_ENV
        "${ROCPROFILER_MEMCHECK_PRELOAD_ENV_VALUE}:$<TARGET_FILE:kernel-replay-local-context-client>"
        )
else()
    set(PRELOAD_ENV "$<TARGET_FILE:kernel-replay-local-context-client>")
endif()

include(rocprofiler-sdk-utilities)
rocprofiler_sdk_spm_disabled(IS_SPM_DISABLED)
rocprofiler_sdk_pc_sampling_disabled(IS_PC_SAMPLING_DISABLED)

function(add_kr_lc_test NAME)
    cmake_parse_arguments(arg "" "DISABLED" "ENVIRONMENT" ${ARGN})
    set(_disabled)
    if(arg_DISABLED)
        set(_disabled DISABLED ${arg_DISABLED})
    endif()
    rocprofiler_add_integration_execute_test(
        ${NAME}
        COMMAND $<TARGET_FILE:kernel-replay-local-context-testapp>
        DEPENDS kernel-replay-local-context-client
        TIMEOUT 120
        LABELS "integration-tests;kernel-replay"
        PRELOAD "${PRELOAD_ENV}"
        ENVIRONMENT ${arg_ENVIRONMENT} ${_disabled}
        PASS_REGULAR_EXPRESSION "\\[lc\\] PASS"
        FAIL_REGULAR_EXPRESSION "\\[lc\\] FAIL|${ROCPROFILER_DEFAULT_FAIL_REGEX}"
        SKIP_REGULAR_EXPRESSION "ATT unavailable|SPM unavailable|PC sampling unavailable")
endfunction()

# Baseline: no local toggle, four replay passes, four counter records.
add_kr_lc_test(test-kernel-replay-local-context-counters-all-passes ENVIRONMENT
               "KR_LC_SERVICES=counters;KR_LC_PASSES=4;KR_LC_STOP_PASS=-1")

# Dispatch counters: stop on pass 0 enter → zero records from the replayed kernel.
add_kr_lc_test(test-kernel-replay-local-context-counters-stop-from-start ENVIRONMENT
               "KR_LC_SERVICES=counters;KR_LC_PASSES=4;KR_LC_STOP_PASS=0")

# Dispatch counters: collect pass 0, stop on pass 1 enter → one record (sticky).
add_kr_lc_test(test-kernel-replay-local-context-counters-stop-after-0 ENVIRONMENT
               "KR_LC_SERVICES=counters;KR_LC_PASSES=4;KR_LC_STOP_PASS=1")

# Dispatch counters: stop after pass 0, locally re-start on pass 2 → records on 0, 2, 3.
add_kr_lc_test(
    test-kernel-replay-local-context-counters-stop-then-restart ENVIRONMENT
    "KR_LC_SERVICES=counters;KR_LC_PASSES=4;KR_LC_STOP_PASS=1;KR_LC_START_PASS=2")

# ATT: locally stopped before pass 0 → no shader data for the replayed kernel.
add_kr_lc_test(test-kernel-replay-local-context-att-stop-from-start ENVIRONMENT
               "KR_LC_SERVICES=att;KR_LC_PASSES=4;KR_LC_STOP_PASS=0")

# ATT: collect pass 0 then sticky-stop → shader data from the enabled pass.
add_kr_lc_test(test-kernel-replay-local-context-att-stop-after-0 ENVIRONMENT
               "KR_LC_SERVICES=att;KR_LC_PASSES=4;KR_LC_STOP_PASS=1")

# SPM: same dispatch-scoped contract as counters.
add_kr_lc_test(
    test-kernel-replay-local-context-spm-stop-from-start
    ENVIRONMENT
    "KR_LC_SERVICES=spm;KR_LC_PASSES=4;KR_LC_STOP_PASS=0;ROCPROFILER_SPM_BETA_ENABLED=True"
    DISABLED
    ${IS_SPM_DISABLED})

add_kr_lc_test(
    test-kernel-replay-local-context-spm-stop-after-0
    ENVIRONMENT
    "KR_LC_SERVICES=spm;KR_LC_PASSES=4;KR_LC_STOP_PASS=1;ROCPROFILER_SPM_BETA_ENABLED=True"
    DISABLED
    ${IS_SPM_DISABLED})

# PC sampling: local stop must succeed. Collection is agent-wide and currently ignores the
# override (documented no-op).
add_kr_lc_test(
    test-kernel-replay-local-context-pc-sampling-stop-after-0
    ENVIRONMENT
    "KR_LC_SERVICES=pc-sampling;KR_LC_PASSES=4;KR_LC_STOP_PASS=1;ROCPROFILER_PC_SAMPLING_BETA_ENABLED=ON"
    DISABLED
    ${IS_PC_SAMPLING_DISABLED})

# Design example: counters every pass, ATT pass 0 only.
add_kr_lc_test(
    test-kernel-replay-local-context-counters-att ENVIRONMENT
    "KR_LC_SERVICES=counters,att;KR_LC_KEEP=counters;KR_LC_PASSES=4;KR_LC_STOP_PASS=1")
