# Copyright (c) Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

include(benchmark)

set(BENCHMARK_SOURCES writer_bench.cpp writer_parallel_bench.cpp)

add_executable(profiler-hub_benchmarks ${BENCHMARK_SOURCES})

set(THREADS_PREFER_PTHREAD_FLAG ON)
find_package(Threads REQUIRED)

target_link_libraries(
    profiler-hub_benchmarks
    PRIVATE
        profiler-hub
        benchmark::benchmark
        benchmark::benchmark_main
        spdlog::spdlog
        Threads::Threads
)

target_include_directories(
    profiler-hub_benchmarks
    PRIVATE ${CMAKE_SOURCE_DIR}/source ${CMAKE_SOURCE_DIR}/source/common
)
