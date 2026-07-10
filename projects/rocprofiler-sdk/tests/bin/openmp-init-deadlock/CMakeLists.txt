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

if(NOT OMP_TARGET_COMPILER)
    find_program(
        amdclangpp_EXECUTABLE
        NAMES amdclang++
        HINTS ${ROCM_PATH} ENV ROCM_PATH /opt/rocm
        PATHS ${ROCM_PATH} ENV ROCM_PATH /opt/rocm
        PATH_SUFFIXES bin llvm/bin NO_CACHE)
    mark_as_advanced(amdclangpp_EXECUTABLE)

    if(amdclangpp_EXECUTABLE)
        set(OMP_TARGET_COMPILER
            "${amdclangpp_EXECUTABLE}"
            CACHE FILEPATH "")
    endif()
endif()

project(rocprofiler-sdk-tests-bin-openmp-init-deadlock LANGUAGES CXX)

find_package(rocprofiler-sdk REQUIRED)
find_package(Threads REQUIRED)

include(rocprofiler-sdk-custom-compilation)

function(openmp_init_deadlock_add_omp_library _NAME _SRC)
    add_library(${_NAME} SHARED)
    target_sources(${_NAME} PRIVATE ${_SRC})
    target_compile_options(${_NAME} PRIVATE -fopenmp)
    target_link_options(${_NAME} PRIVATE -fopenmp)
    target_link_libraries(${_NAME} PRIVATE rocprofiler-sdk::tests-build-flags)
    if(OMP_TARGET_COMPILER)
        rocprofiler_sdk_custom_compilation(TARGET ${_NAME}
                                           COMPILER ${OMP_TARGET_COMPILER})
    endif()
endfunction()

# models OpenBLAS (USE_OPENMP=1): DT_INIT calls omp_get_num_places() (see blas_stub.cpp)
openmp_init_deadlock_add_omp_library(openmp-init-deadlock-blas-stub blas_stub.cpp)

# models an ordinary OpenMP-using application library: its DT_INIT calls into libomp and
# it exposes lib_get_num_places() so the executable can confirm the constructor ran
openmp_init_deadlock_add_omp_library(openmp-init-deadlock-consumer openmp_consumer.cpp)

# models a rocprofiler-sdk client tool: only exports rocprofiler_configure. It depends on
# the consumer (and blas-stub) so their DT_INIT constructors are pending when discovery
# runs
openmp_init_deadlock_add_omp_library(openmp-init-deadlock-client client.cpp)
target_link_libraries(
    openmp-init-deadlock-client
    PRIVATE rocprofiler-sdk::rocprofiler-sdk
            # ensure the consumer/blas-stub initialize first and remain hard dependencies
            "-Wl,--no-as-needed"
            openmp-init-deadlock-consumer
            openmp-init-deadlock-blas-stub
            "-Wl,--as-needed")

add_executable(openmp-init-deadlock)
target_sources(openmp-init-deadlock PRIVATE openmp-init-deadlock.cpp)
# main() calls omp_get_num_places() directly, so it needs -fopenmp and an OpenMP runtime
target_compile_options(openmp-init-deadlock PRIVATE -fopenmp)
target_link_options(openmp-init-deadlock PRIVATE -fopenmp)
target_link_libraries(
    openmp-init-deadlock
    PRIVATE Threads::Threads
            rocprofiler-sdk::tests-build-flags
            # pulls in the client (and transitively consumer + blas-stub) as load-time
            # deps
            "-Wl,--no-as-needed"
            openmp-init-deadlock-client
            openmp-init-deadlock-consumer
            "-Wl,--as-needed")
if(OMP_TARGET_COMPILER)
    rocprofiler_sdk_custom_compilation(TARGET openmp-init-deadlock
                                       COMPILER ${OMP_TARGET_COMPILER})
endif()
