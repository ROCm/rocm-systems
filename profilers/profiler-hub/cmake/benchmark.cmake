# Copyright (c) Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

include_guard(DIRECTORY)

set(BENCHMARK_VERSION "1.8.3" CACHE STRING "Google Benchmark version")

find_package(benchmark QUIET)

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
