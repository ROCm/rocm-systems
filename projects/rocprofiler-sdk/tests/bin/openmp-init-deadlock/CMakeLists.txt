# MIT License
#
# Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
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

#
cmake_minimum_required(VERSION 3.21.0 FATAL_ERROR)

project(rocprofiler-sdk-tests-bin-openmp-init-deadlock LANGUAGES CXX)

find_package(rocprofiler-sdk REQUIRED)
find_package(Threads REQUIRED)

# This regression test needs the LLVM OpenMP runtime (libomp), which performs OMPT tool
# discovery while holding its non-recursive init ticket lock. The GNU runtime (libgomp)
# does not reproduce the deadlock the same way, so the test is only registered when libomp
# is found.
find_library(
    openmp-init-deadlock-libomp
    NAMES omp libomp.so.5 libomp.so
    HINTS ${ROCM_PATH} ENV ROCM_PATH /opt/rocm
    PATH_SUFFIXES llvm/lib lib NO_CACHE)

if(NOT openmp-init-deadlock-libomp)
    message(
        STATUS
            "[openmp-init-deadlock] LLVM libomp not found; skipping OMPT init-deadlock regression app"
        )
    return()
endif()

# -fopenmp is used only to parse the '#pragma omp' directives at compile time; the LLVM
# libomp runtime is linked explicitly below (rather than relying on the compiler's default
# OpenMP runtime, which for GCC would be libgomp) so the OMPT/ticket-lock behavior under
# test is exercised regardless of the compiler.
function(openmp_init_deadlock_add_omp_library _NAME _SRC)
    add_library(${_NAME} SHARED)
    target_sources(${_NAME} PRIVATE ${_SRC})
    target_compile_options(${_NAME} PRIVATE -fopenmp)
    target_link_libraries(${_NAME} PRIVATE ${openmp-init-deadlock-libomp}
                                           rocprofiler-sdk::tests-build-flags)
endfunction()

# models OpenBLAS (USE_OPENMP=1): DT_INIT enters an omp parallel region
openmp_init_deadlock_add_omp_library(openmp-init-deadlock-blas-stub blas_stub.cpp)

# models a rocprofiler-sdk client tool: exports rocprofiler_configure and depends on the
# blas-stub so its own DT_INIT is still pending when discovery runs
openmp_init_deadlock_add_omp_library(openmp-init-deadlock-client client.cpp)
target_link_libraries(
    openmp-init-deadlock-client
    PRIVATE rocprofiler-sdk::rocprofiler-sdk
            # ensure blas-stub initializes first and remains a hard dependency
            "-Wl,--no-as-needed" openmp-init-deadlock-blas-stub "-Wl,--as-needed")

add_executable(openmp-init-deadlock)
target_sources(openmp-init-deadlock PRIVATE openmp-init-deadlock.cpp)
target_link_libraries(
    openmp-init-deadlock
    PRIVATE Threads::Threads rocprofiler-sdk::tests-build-flags
            # pulls in the client (and transitively blas-stub) as load-time dependencies
            "-Wl,--no-as-needed" openmp-init-deadlock-client "-Wl,--as-needed")
