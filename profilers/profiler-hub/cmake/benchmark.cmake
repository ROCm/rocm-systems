# Copyright (c) Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

include_guard(DIRECTORY)

set(BENCHMARK_VERSION "1.8.3" CACHE STRING "Minimum Google Benchmark version")

find_package(benchmark QUIET)

set(_benchmark_reason "")
if(NOT benchmark_FOUND)
    set(_benchmark_reason "no package was found on CMAKE_PREFIX_PATH")
elseif(benchmark_VERSION VERSION_LESS BENCHMARK_VERSION)
    set(_benchmark_reason
        "version ${benchmark_VERSION} was found, but ${BENCHMARK_VERSION} or newer is required"
    )
endif()

if(_benchmark_reason STREQUAL "")
    message(
        STATUS
        "Using system Google Benchmark (version ${benchmark_VERSION})"
    )
elseif(NOT PROFILER_HUB_FETCH_DEPENDENCIES)
    message(
        FATAL_ERROR
        "profiler-hub requires Google Benchmark: ${_benchmark_reason}. Provide it on CMAKE_PREFIX_PATH, configure with -DPROFILER_HUB_BUILD_BENCHMARKS=OFF, or configure with -DPROFILER_HUB_FETCH_DEPENDENCIES=ON to download it."
    )
else()
    message(
        STATUS
        "Fetching Google Benchmark ${BENCHMARK_VERSION}: ${_benchmark_reason}"
    )
    include(FetchContent)

    FetchContent_Declare(
        googlebenchmark
        GIT_REPOSITORY https://github.com/google/benchmark.git
        GIT_TAG v${BENCHMARK_VERSION}
        GIT_SHALLOW TRUE
    )

    set(BENCHMARK_ENABLE_TESTING OFF CACHE BOOL "" FORCE)
    set(BENCHMARK_ENABLE_INSTALL OFF CACHE BOOL "" FORCE)
    set(BENCHMARK_ENABLE_GTEST_TESTS OFF CACHE BOOL "" FORCE)
    set(BENCHMARK_USE_BUNDLED_GTEST OFF CACHE BOOL "" FORCE)

    FetchContent_MakeAvailable(googlebenchmark)

    if(NOT TARGET benchmark::benchmark)
        add_library(benchmark::benchmark ALIAS benchmark)
    endif()

    if(NOT TARGET benchmark::benchmark_main)
        add_library(benchmark::benchmark_main ALIAS benchmark_main)
    endif()
endif()

unset(_benchmark_reason)
