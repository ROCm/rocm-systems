#
# Kernel replay performance regression tests (pass scaling + cost-model wall-time bounds).
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

project(rocprofiler-sdk-tests-kernel-replay-perf LANGUAGES CXX HIP)

foreach(_TYPE DEBUG MINSIZEREL RELEASE RELWITHDEBINFO)
    if("${CMAKE_HIP_FLAGS_${_TYPE}}" STREQUAL "")
        set(CMAKE_HIP_FLAGS_${_TYPE} "${CMAKE_CXX_FLAGS_${_TYPE}}")
    endif()
endforeach()

find_package(rocprofiler-sdk REQUIRED)
find_package(Python3 REQUIRED)

add_library(kernel-replay-perf-client SHARED)
target_sources(kernel-replay-perf-client PRIVATE client.cpp)
target_link_libraries(kernel-replay-perf-client PRIVATE rocprofiler-sdk::rocprofiler-sdk)

set_source_files_properties(main.cpp PROPERTIES LANGUAGE HIP)
add_executable(kernel-replay-perf-testapp)
target_sources(kernel-replay-perf-testapp PRIVATE main.cpp)
target_link_libraries(kernel-replay-perf-testapp PRIVATE Threads::Threads)

# 64 MB ballast, 8 replayed dispatches; compare P=2 vs P=5 wall time. The baseline is P=2
# rather than P=1 because a pass count of 1 is not replayed at all, so a P=1 baseline
# measures a bare dispatch and the ratio reports the cost of enabling replay instead of
# per-pass scaling. See the header comment in run_and_validate.py.
#
# Registered only for the nightly perf job. Replay wall time is dominated by fixed
# per-dispatch cost (agent drain, inventory discovery, writer lock) rather than by the
# pass count, so on a shared runner the ratio tracks contention rather than a regression.
# Gated in CMake rather than excluded in the workflow, matching the sweeps below.
if(ROCPROFILER_BUILD_NIGHTLY_PERF_CTESTS)
    add_test(
        NAME test-kernel-replay-perf-scaling
        COMMAND
            ${Python3_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/run_and_validate.py
            --testapp $<TARGET_FILE:kernel-replay-perf-testapp> --client
            $<TARGET_FILE:kernel-replay-perf-client> --ballast-mb 64 --launches 8
            --base-passes 2 --high-passes 5 --max-scaling-ratio 8.0 --repeat 3 --warmup 1)

    set_tests_properties(
        test-kernel-replay-perf-scaling
        PROPERTIES TIMEOUT
                   420
                   LABELS
                   "integration-tests;kernel-replay;kernel-replay-perf;perf-nightly"
                   RESOURCE_LOCK
                   rocprofiler_kernel_replay
                   DEPENDS
                   "kernel-replay-perf-client;kernel-replay-perf-testapp"
                   ENVIRONMENT
                   "KR_PERF_RESULTS_JSON=${CMAKE_CURRENT_BINARY_DIR}/kr_perf_scaling.json"
                   PASS_REGULAR_EXPRESSION
                   "\\[kr-perf-run\\] PASS"
                   FAIL_REGULAR_EXPRESSION
                   "${ROCPROFILER_DEFAULT_FAIL_REGEX}")
endif()

# Pass-count sweep {2,3,5,8} at smaller footprint — catches per-pass overhead regressions.
# Starts at 2 for the same reason as the scaling test above: P=1 is not replayed.
if(ROCPROFILER_BUILD_NIGHTLY_PERF_CTESTS)
    add_test(
        NAME test-kernel-replay-perf-pass-sweep
        COMMAND
            ${Python3_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/run_pass_sweep.py --testapp
            $<TARGET_FILE:kernel-replay-perf-testapp> --client
            $<TARGET_FILE:kernel-replay-perf-client> --ballast-mb 32 --launches 4
            --passes 2 3 5 8 --slack 2.5 --repeat 3 --warmup 1)

    set_tests_properties(
        test-kernel-replay-perf-pass-sweep
        PROPERTIES
            TIMEOUT
            900
            LABELS
            "integration-tests;kernel-replay;kernel-replay-perf;perf-nightly"
            RESOURCE_LOCK
            rocprofiler_kernel_replay
            DEPENDS
            "kernel-replay-perf-client;kernel-replay-perf-testapp"
            ENVIRONMENT
            "KR_PERF_SWEEP_JSON=${CMAKE_CURRENT_BINARY_DIR}/kr_perf_pass_sweep.json"
            PASS_REGULAR_EXPRESSION
            "\\[kr-perf-sweep\\] PASS overall"
            FAIL_REGULAR_EXPRESSION
            "${ROCPROFILER_DEFAULT_FAIL_REGEX}")
endif()

# Small-footprint P=2 vs P=3 — exercises older-GPU-friendly memory pressure. Nightly only,
# for the same reason as the scaling test above: at this footprint the measurement is
# mostly runner noise.
if(ROCPROFILER_BUILD_NIGHTLY_PERF_CTESTS)
    add_test(
        NAME test-kernel-replay-perf-small-footprint
        COMMAND
            ${Python3_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/run_and_validate.py
            --testapp $<TARGET_FILE:kernel-replay-perf-testapp> --client
            $<TARGET_FILE:kernel-replay-perf-client> --ballast-mb 16 --launches 6
            --base-passes 2 --high-passes 3 --max-scaling-ratio 6.0 --repeat 3 --warmup 1)

    set_tests_properties(
        test-kernel-replay-perf-small-footprint
        PROPERTIES TIMEOUT
                   300
                   LABELS
                   "integration-tests;kernel-replay;kernel-replay-perf;perf-nightly"
                   RESOURCE_LOCK
                   rocprofiler_kernel_replay
                   DEPENDS
                   "kernel-replay-perf-client;kernel-replay-perf-testapp"
                   PASS_REGULAR_EXPRESSION
                   "\\[kr-perf-run\\] PASS"
                   FAIL_REGULAR_EXPRESSION
                   "${ROCPROFILER_DEFAULT_FAIL_REGEX}")
endif()

# Higher launch count at moderate footprint — amortization / drain overhead guard.
if(ROCPROFILER_BUILD_NIGHTLY_PERF_CTESTS)
    add_test(
        NAME test-kernel-replay-perf-many-launches
        COMMAND
            ${Python3_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/run_and_validate.py
            --testapp $<TARGET_FILE:kernel-replay-perf-testapp> --client
            $<TARGET_FILE:kernel-replay-perf-client> --ballast-mb 32 --launches 16
            --base-passes 2 --high-passes 3 --max-scaling-ratio 6.0 --repeat 3 --warmup 1)

    set_tests_properties(
        test-kernel-replay-perf-many-launches
        PROPERTIES TIMEOUT
                   720
                   LABELS
                   "integration-tests;kernel-replay;kernel-replay-perf;perf-nightly"
                   RESOURCE_LOCK
                   rocprofiler_kernel_replay
                   DEPENDS
                   "kernel-replay-perf-client;kernel-replay-perf-testapp"
                   PASS_REGULAR_EXPRESSION
                   "\\[kr-perf-run\\] PASS"
                   FAIL_REGULAR_EXPRESSION
                   "${ROCPROFILER_DEFAULT_FAIL_REGEX}")
endif()
