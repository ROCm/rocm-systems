#
# Kernel-replay per-pass service selection integration tests.
#
# A dedicated tool replays one distinctive HIP dispatch N times and decides, in each
# service's own dispatch callback, on which passes that service collects. Individual
# services are the priority; one counters+ATT combination covers the design's main example
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

project(rocprofiler-sdk-tests-kernel-replay-pass-selection LANGUAGES CXX HIP)

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

add_library(kernel-replay-pass-selection-client SHARED)
target_sources(kernel-replay-pass-selection-client PRIVATE client.cpp)
target_link_libraries(
    kernel-replay-pass-selection-client PRIVATE rocprofiler-sdk::rocprofiler-sdk
                                                rocprofiler-sdk::tests-build-flags)

set_source_files_properties(main.cpp PROPERTIES LANGUAGE HIP)
add_executable(kernel-replay-pass-selection-testapp)
target_sources(kernel-replay-pass-selection-testapp PRIVATE main.cpp)
target_link_libraries(kernel-replay-pass-selection-testapp
                      PRIVATE rocprofiler-sdk::tests-build-flags)

if(ROCPROFILER_MEMCHECK_PRELOAD_ENV_VALUE)
    set(PRELOAD_ENV
        "${ROCPROFILER_MEMCHECK_PRELOAD_ENV_VALUE}:$<TARGET_FILE:kernel-replay-pass-selection-client>"
        )
else()
    set(PRELOAD_ENV "$<TARGET_FILE:kernel-replay-pass-selection-client>")
endif()

include(rocprofiler-sdk-utilities)
rocprofiler_sdk_spm_disabled(IS_SPM_DISABLED)
rocprofiler_sdk_pc_sampling_disabled(IS_PC_SAMPLING_DISABLED)

function(add_kr_ps_test NAME)
    cmake_parse_arguments(arg "" "DISABLED" "ENVIRONMENT" ${ARGN})
    set(_disabled)
    if(arg_DISABLED)
        set(_disabled DISABLED ${arg_DISABLED})
    endif()
    rocprofiler_add_integration_execute_test(
        ${NAME}
        COMMAND $<TARGET_FILE:kernel-replay-pass-selection-testapp>
        DEPENDS kernel-replay-pass-selection-client
        TIMEOUT 120
        LABELS "integration-tests;kernel-replay"
        PRELOAD "${PRELOAD_ENV}"
        ENVIRONMENT ${arg_ENVIRONMENT} ${_disabled}
        PASS_REGULAR_EXPRESSION "\\[kr-ps\\] PASS"
        FAIL_REGULAR_EXPRESSION "\\[kr-ps\\] FAIL|${ROCPROFILER_DEFAULT_FAIL_REGEX}"
        SKIP_REGULAR_EXPRESSION "ATT unavailable|SPM unavailable|PC sampling unavailable")
endfunction()

# Baseline: counters selected on every pass, four replay passes, four counter records.
add_kr_ps_test(test-kernel-replay-pass-selection-counters-all-passes ENVIRONMENT
               "KR_PS_SERVICES=counters;KR_PS_PASSES=4;KR_PS_STOP_PASS=-1")

# Dispatch counters: never selected → zero records from the replayed kernel.
add_kr_ps_test(test-kernel-replay-pass-selection-counters-stop-from-start ENVIRONMENT
               "KR_PS_SERVICES=counters;KR_PS_PASSES=4;KR_PS_STOP_PASS=0")

# Dispatch counters: selected on pass 0 only → one record.
add_kr_ps_test(test-kernel-replay-pass-selection-counters-stop-after-0 ENVIRONMENT
               "KR_PS_SERVICES=counters;KR_PS_PASSES=4;KR_PS_STOP_PASS=1")

# Dispatch counters: stop after pass 0, select again from pass 2 → records on 0, 2, 3.
add_kr_ps_test(
    test-kernel-replay-pass-selection-counters-stop-then-restart ENVIRONMENT
    "KR_PS_SERVICES=counters;KR_PS_PASSES=4;KR_PS_STOP_PASS=1;KR_PS_START_PASS=2")

# ATT: never selected → no shader data for the replayed kernel.
add_kr_ps_test(test-kernel-replay-pass-selection-att-stop-from-start ENVIRONMENT
               "KR_PS_SERVICES=att;KR_PS_PASSES=4;KR_PS_STOP_PASS=0")

# ATT: selected on pass 0 only → shader data from that pass.
add_kr_ps_test(test-kernel-replay-pass-selection-att-stop-after-0 ENVIRONMENT
               "KR_PS_SERVICES=att;KR_PS_PASSES=4;KR_PS_STOP_PASS=1")

# SPM: same dispatch-scoped contract as counters.
add_kr_ps_test(
    test-kernel-replay-pass-selection-spm-stop-from-start
    ENVIRONMENT
    "KR_PS_SERVICES=spm;KR_PS_PASSES=4;KR_PS_STOP_PASS=0;ROCPROFILER_SPM_BETA_ENABLED=True"
    DISABLED
    ${IS_SPM_DISABLED})

add_kr_ps_test(
    test-kernel-replay-pass-selection-spm-stop-after-0
    ENVIRONMENT
    "KR_PS_SERVICES=spm;KR_PS_PASSES=4;KR_PS_STOP_PASS=1;ROCPROFILER_SPM_BETA_ENABLED=True"
    DISABLED
    ${IS_SPM_DISABLED})

# PC sampling is agent-wide and has no dispatch callback, so it samples every pass; this
# checks that replay still completes under it.
add_kr_ps_test(
    test-kernel-replay-pass-selection-pc-sampling-every-pass ENVIRONMENT
    "KR_PS_SERVICES=pc-sampling;KR_PS_PASSES=4" DISABLED ${IS_PC_SAMPLING_DISABLED})

# Design example: counters every pass, ATT pass 0 only.
add_kr_ps_test(
    test-kernel-replay-pass-selection-counters-att ENVIRONMENT
    "KR_PS_SERVICES=counters,att;KR_PS_KEEP=counters;KR_PS_PASSES=4;KR_PS_STOP_PASS=1")
