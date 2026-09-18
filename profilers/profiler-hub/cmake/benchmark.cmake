# Copyright (c) Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

include_guard(DIRECTORY)

set(BENCHMARK_VERSION "1.8.3" CACHE STRING "Minimum Google Benchmark version")

# Fetching is On by default: this branch runs only under
# -DPROFILER_HUB_FETCH_DEPENDENCIES=OFF.
if(NOT PROFILER_HUB_FETCH_DEPENDENCIES)
    find_package(benchmark ${BENCHMARK_VERSION})

    if(NOT benchmark_FOUND)
        message(
            FATAL_ERROR
            "profiler-hub requires Google Benchmark ${BENCHMARK_VERSION} or newer on CMAKE_PREFIX_PATH. Configure with -DPROFILER_HUB_BUILD_BENCHMARKS=OFF to skip the benchmarks, or drop -DPROFILER_HUB_FETCH_DEPENDENCIES=OFF to download it instead."
        )
    endif()

    message(
        STATUS
        "Using system Google Benchmark (version ${benchmark_VERSION})"
    )
else()
    include(FetchContent)

    FetchContent_Declare(
        googlebenchmark
        GIT_REPOSITORY https://github.com/google/benchmark.git
        GIT_TAG v${BENCHMARK_VERSION}
        GIT_SHALLOW TRUE
        # Without this, the MakeAvailable below always fetches.
        # FetchContent derives the find_package call from the content name, which
        # here is not the name Google Benchmark installs itself under.
        FIND_PACKAGE_ARGS ${BENCHMARK_VERSION} NAMES benchmark
    )

    set(BENCHMARK_ENABLE_TESTING OFF CACHE BOOL "" FORCE)
    set(BENCHMARK_ENABLE_INSTALL OFF CACHE BOOL "" FORCE)
    set(BENCHMARK_ENABLE_GTEST_TESTS OFF CACHE BOOL "" FORCE)
    set(BENCHMARK_USE_BUNDLED_GTEST OFF CACHE BOOL "" FORCE)

    # Tries find_package() first, fetches only if that fails.
    FetchContent_MakeAvailable(googlebenchmark)

    if(NOT TARGET benchmark::benchmark)
        add_library(benchmark::benchmark ALIAS benchmark)
    endif()

    if(NOT TARGET benchmark::benchmark_main)
        add_library(benchmark::benchmark_main ALIAS benchmark_main)
    endif()
endif()
