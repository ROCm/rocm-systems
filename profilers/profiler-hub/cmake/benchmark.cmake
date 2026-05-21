# Copyright (c) Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

include_guard(DIRECTORY)

option(
    PROFILER_HUB_USE_SYSTEM_BENCHMARK
    "Use system-installed Google Benchmark if available"
    ON
)

set(BENCHMARK_VERSION "1.8.3" CACHE STRING "Google Benchmark version")

if(PROFILER_HUB_USE_SYSTEM_BENCHMARK)
    find_package(benchmark ${BENCHMARK_VERSION} QUIET)
endif()

if(benchmark_FOUND)
    message(
        STATUS
        "Using system Google Benchmark (version ${benchmark_VERSION})"
    )
else()
    message(
        STATUS
        "System Google Benchmark not found, fetching version ${BENCHMARK_VERSION}"
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
