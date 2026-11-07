# Copyright (c) Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

cmake_minimum_required(VERSION 3.16)
project(bench_workloads LANGUAGES C CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

option(BENCH_BUILD_TESTS "Build bench_hooks.h gtest tests" ON)

find_package(Threads REQUIRED)

add_library(bench_hooks INTERFACE)
target_include_directories(bench_hooks INTERFACE ${CMAKE_CURRENT_SOURCE_DIR}/include)

foreach(
    name
    noop
    cpu_compute
    call_heavy
    syscall_heavy
    mem_alloc
    io_files
    threads
    ai_hpc_cpu
)
    add_executable(${name} src/${name}.cpp)
    target_link_libraries(${name} PRIVATE bench_hooks Threads::Threads)
    target_compile_options(${name} PRIVATE -O2 -Wall -Wextra)
endforeach()

option(
    BENCH_BUILD_GPU
    "Build the HIP programs (gpu_kernels, gpu_long_kernel, ai_hpc_gpu) when HIP is available"
    ON
)

if(BENCH_BUILD_GPU)
    find_package(hip CONFIG QUIET)
    if(hip_FOUND)
        if(NOT CMAKE_HIP_COMPILER)
            find_program(BENCH_AMDCLANG amdclang++ REQUIRED)
            set(CMAKE_HIP_COMPILER ${BENCH_AMDCLANG})
        endif()
        if(NOT CMAKE_HIP_ARCHITECTURES)
            set(CMAKE_HIP_ARCHITECTURES native)
        endif()
        enable_language(HIP)
        foreach(name gpu_kernels gpu_long_kernel ai_hpc_gpu ai_hpc_hybrid)
            add_executable(${name} src/${name}.cpp)
            set_source_files_properties(src/${name}.cpp PROPERTIES LANGUAGE HIP)
            target_link_libraries(${name} PRIVATE bench_hooks hip::host)
            target_compile_options(${name} PRIVATE -O2)
        endforeach()
    else()
        message(STATUS "HIP not found: skipping the GPU programs")
    endif()
endif()

if(BENCH_BUILD_TESTS)
    find_package(GTest)
    if(GTest_FOUND)
        enable_testing()
        add_subdirectory(tests)
    else()
        message(STATUS "GTest not found: skipping bench_hooks tests")
    endif()
endif()
