# MIT License
#
# Copyright (c) 2025 Advanced Micro Devices, Inc. All Rights Reserved.
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

include(benchmark)

set(BENCHMARK_SOURCES writer_bench.cpp writer_parallel_bench.cpp)

add_executable(rocpdsna_benchmarks ${BENCHMARK_SOURCES})

target_link_libraries(
    rocpdsna_benchmarks
    PRIVATE
        rocpdsna
        benchmark::benchmark
        benchmark::benchmark_main
        pthread
        spdlog::spdlog
        rocpdsna-sqlite3
)

target_include_directories(
    rocpdsna_benchmarks
    PRIVATE ${CMAKE_SOURCE_DIR}/source ${CMAKE_SOURCE_DIR}/source/common
)
