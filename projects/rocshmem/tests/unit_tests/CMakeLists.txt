###############################################################################
# Copyright (c) Advanced Micro Devices, Inc. All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to
# deal in the Software without restriction, including without limitation the
# rights to use, copy, modify, merge, publish, distribute, sublicense, and/or
# sell copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
# FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS
# IN THE SOFTWARE.
###############################################################################

cmake_minimum_required(VERSION 3.16.3 FATAL_ERROR)

###############################################################################
# PROJECT
###############################################################################
include(${CMAKE_SOURCE_DIR}/cmake/setup_project.cmake)
project(rocshmem_unit_tests VERSION 1.0.0 LANGUAGES CXX)

###############################################################################
# SOURCES
###############################################################################
add_executable(${PROJECT_NAME} "")

target_include_directories(
  ${PROJECT_NAME}
  PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}
)

target_sources(
  ${PROJECT_NAME}
  PRIVATE
    shmem_gtest.cpp
    heap_memory_gtest.cpp
    hipmalloc_gtest.cpp
    bin_gtest.cpp
    binner_gtest.cpp
    #bitwise_gtest.cpp # Test is disabled because of compilation errors
    address_record_gtest.cpp
    index_strategy_gtest.cpp
    single_heap_gtest.cpp
    symmetric_heap_gtest.cpp
    pow2_bins_gtest.cpp
    dlmalloc_gtest.cpp
    remote_heap_info_gtest.cpp
    mpi_instance_gtest.cpp
    abql_block_mutex_gtest.cpp
    notifier_gtest.cpp
    free_list_gtest.cpp
    wavefront_size_gtest.cpp
    atomic_wf_queue_gtest.cpp
    envvar_gtest.cpp
)

if (USE_IPC)
  target_sources(
    ${PROJECT_NAME}
    PRIVATE
      ipc_impl_simple_coarse_gtest.cpp
      ipc_impl_simple_fine_gtest.cpp
      ipc_impl_tiled_fine_gtest.cpp
  )
endif()

###############################################################################
# ROCSHMEM DEPENDENCY
###############################################################################
find_package(hip REQUIRED PATHS /opt/rocm)
find_package(MPI REQUIRED)

if (BUILD_TESTS_ONLY)
  find_package(rocshmem REQUIRED PATHS /opt/rocm)

  target_include_directories(
    ${PROJECT_NAME}
    PRIVATE
      $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/..>
      $<BUILD_INTERFACE:${CMAKE_BINARY_DIR}>
  )
endif()

target_link_libraries(
  ${PROJECT_NAME}
  PRIVATE
    MPI::MPI_CXX
    roc::rocshmem
)

###############################################################################
# GTEST DEPENDENCY
###############################################################################
include(FetchContent)

FetchContent_Declare(
    googletest
    PREFIX extern/googletest
    GIT_REPOSITORY https://github.com/google/googletest.git
    GIT_TAG release-1.12.0
)

set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
set(BUILD_GMOCK OFF CACHE BOOL "" FORCE)
set(BUILD_GTEST ON CACHE BOOL "" FORCE)
set(INSTALL_GTEST OFF CACHE BOOL "" FORCE)

FetchContent_MakeAvailable(googletest)

target_link_libraries(
  ${PROJECT_NAME}
  PRIVATE
    gtest
    gtest_main
)
