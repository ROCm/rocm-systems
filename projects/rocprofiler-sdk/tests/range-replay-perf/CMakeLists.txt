#
# Range replay performance regression tests (pass scaling + fixed-cost amortization).
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

project(rocprofiler-sdk-tests-range-replay-perf LANGUAGES CXX HIP)

foreach(_TYPE DEBUG MINSIZEREL RELEASE RELWITHDEBINFO)
    if("${CMAKE_HIP_FLAGS_${_TYPE}}" STREQUAL "")
        set(CMAKE_HIP_FLAGS_${_TYPE} "${CMAKE_CXX_FLAGS_${_TYPE}}")
    endif()
endforeach()

# The `headers` component is what lets the application place its range boundaries against
# the real types without linking the SDK; see the target_link_libraries comment below.
find_package(rocprofiler-sdk REQUIRED COMPONENTS headers shared-library)
find_package(Python3 REQUIRED)
find_package(Threads REQUIRED)

add_library(range-replay-perf-client SHARED)
target_sources(range-replay-perf-client PRIVATE client.cpp)
target_include_directories(range-replay-perf-client PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(range-replay-perf-client PRIVATE rocprofiler-sdk::rocprofiler-sdk)

# The application resolves the range replay entry points with dlsym instead of linking the
# SDK: linking rocprofiler-sdk into a HIP executable makes HIP report
# hipErrorInvalidDeviceFunction on the first launch on gfx942, the same reason the range
# replay samples do it that way. It still needs the SDK headers for the declarations, so
# it takes the headers target only.
set_source_files_properties(main.cpp PROPERTIES LANGUAGE HIP)
add_executable(range-replay-perf-testapp)
target_sources(range-replay-perf-testapp PRIVATE main.cpp)
target_include_directories(range-replay-perf-testapp PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(
    range-replay-perf-testapp PRIVATE rocprofiler-sdk::rocprofiler-sdk-headers
                                      Threads::Threads ${CMAKE_DL_LIBS})
add_dependencies(range-replay-perf-testapp range-replay-perf-client)

# Pass-count scaling: 32 MB footprint, 4 dispatches per range, P=2 vs P=5.
#
# Registered only for the nightly perf job, matching kernel-replay-perf. Range replay wall
# time is dominated by the once-per-range snapshot and restore rather than by the pass
# count, so on a shared runner this ratio tracks contention more than it tracks a
# regression.
if(ROCPROFILER_BUILD_NIGHTLY_PERF_CTESTS)
    add_test(
        NAME test-range-replay-perf-scaling
        COMMAND
            ${Python3_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/run_and_validate.py
            --testapp $<TARGET_FILE:range-replay-perf-testapp> --client
            $<TARGET_FILE:range-replay-perf-client> --ballast-mb 32 --dispatches 4
            --ranges 2 --base-passes 2 --high-passes 5 --max-scaling-ratio 8.0 --repeat 3
            --warmup 1)

    set_tests_properties(
        test-range-replay-perf-scaling
        PROPERTIES TIMEOUT
                   420
                   LABELS
                   "integration-tests;range-replay;range-replay-perf;perf-nightly"
                   RESOURCE_LOCK
                   rocprofiler_range_replay
                   DEPENDS
                   "range-replay-perf-client;range-replay-perf-testapp"
                   ENVIRONMENT
                   "RR_PERF_RESULTS_JSON=${CMAKE_CURRENT_BINARY_DIR}/rr_perf_scaling.json"
                   PASS_REGULAR_EXPRESSION
                   "\\[rr-perf-run\\] PASS"
                   FAIL_REGULAR_EXPRESSION
                   "${ROCPROFILER_DEFAULT_FAIL_REGEX}")
endif()

# Amortization: 64 MB footprint, P=3, sweeping 1 -> 8 dispatches per range. This is the
# test that guards range replay's reason to exist -- that the fixed window cost is paid
# once per range rather than once per dispatch -- so 8x the dispatches must cost well
# under 8x the wall time.
if(ROCPROFILER_BUILD_NIGHTLY_PERF_CTESTS)
    add_test(
        NAME test-range-replay-perf-amortization
        COMMAND
            ${Python3_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/run_amortization.py
            --testapp $<TARGET_FILE:range-replay-perf-testapp> --client
            $<TARGET_FILE:range-replay-perf-client> --ballast-mb 64 --dispatches 1 2 4 8
            --ranges 2 --passes 3 --max-amortization-ratio 3.0 --repeat 3 --warmup 1)

    set_tests_properties(
        test-range-replay-perf-amortization
        PROPERTIES
            TIMEOUT
            900
            LABELS
            "integration-tests;range-replay;range-replay-perf;perf-nightly"
            RESOURCE_LOCK
            rocprofiler_range_replay
            DEPENDS
            "range-replay-perf-client;range-replay-perf-testapp"
            ENVIRONMENT
            "RR_PERF_AMORT_JSON=${CMAKE_CURRENT_BINARY_DIR}/rr_perf_amortization.json"
            PASS_REGULAR_EXPRESSION
            "\\[rr-perf-amort\\] PASS"
            FAIL_REGULAR_EXPRESSION
            "${ROCPROFILER_DEFAULT_FAIL_REGEX}")
endif()
