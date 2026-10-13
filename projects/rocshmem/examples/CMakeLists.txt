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
if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/../cmake/setup_project.cmake")
  include(${CMAKE_CURRENT_SOURCE_DIR}/../cmake/setup_project.cmake)
endif()
project(rocshmem_examples VERSION 1.0.0 LANGUAGES CXX)

find_package(MPI)
find_package(hip REQUIRED)
if (NOT TARGET roc::rocshmem)
  find_package(rocshmem REQUIRED)
endif()
message(STATUS "Found rocshmem: ${rocshmem_DIR} (found version \"${rocshmem_VERSION}\")")

###############################################################################
# SOURCES
###############################################################################
set(EXAMPLE_SOURCES
  rocshmem_allreduce_test.cc
  rocshmem_alltoall_test.cc
  rocshmem_broadcast_test.cc
  rocshmem_getmem_test.cc
  rocshmem_put_signal_test.cc
  rocshmem_user_mem_test.cc
  tile_distributed_gemm.cc
  tile_halo_exchange.cc
)

if (MPI_CXX_FOUND)
  list(APPEND EXAMPLE_SOURCES
    rocshmem_init_attr_test.cc)

  # rocshmem_user_mem_symm_test.cc exercises HIP VMM APIs
  if(${ROCM_MAJOR_VERSION} GREATER 7 OR (${ROCM_MAJOR_VERSION} EQUAL 7 AND ${ROCM_MINOR_VERSION} GREATER_EQUAL 2))
    list(APPEND EXAMPLE_SOURCES rocshmem_user_mem_symm_test.cc)
  else()
    message(STATUS "Skipping rocshmem_user_mem_symm_test: requires ROCm 7.2 or newer (found ${ROCM_MAJOR_VERSION}.${ROCM_MINOR_VERSION})")
  endif()

  add_subdirectory(LL_MoE)
endif()

foreach(SOURCE_FILE IN LISTS EXAMPLE_SOURCES)
  get_filename_component(EXECUTABLE_NAME ${SOURCE_FILE} NAME_WE)

  add_executable(${EXECUTABLE_NAME} ${SOURCE_FILE})

  target_link_libraries(
    ${EXECUTABLE_NAME}
    PRIVATE
      roc::rocshmem
      $<TARGET_NAME_IF_EXISTS:MPI::MPI_CXX>
  )
endforeach()

###############################################################################
# tile_ck_gemm — optional example requiring AMD Composable Kernels (CK Tile)
#
# Pass -DCK_INSTALL_DIR=<path> at configure time to point at a CK install.
# ck_tile is header-only so no library link is required.
###############################################################################
set(CK_INSTALL_DIR "" CACHE PATH "Path to composablekernel install directory")

find_package(composable_kernel QUIET
  COMPONENTS utility
  HINTS ${CK_INSTALL_DIR}/lib/cmake/composable_kernel)
if(composable_kernel_FOUND)
  message(STATUS "Found composable_kernel ${composable_kernel_VERSION}: building tile_ck_gemm example")
  add_executable(tile_ck_gemm tile_ck_gemm.cc)
  target_link_libraries(tile_ck_gemm
    PRIVATE
      roc::rocshmem
      composable_kernel::utility
      $<TARGET_NAME_IF_EXISTS:MPI::MPI_CXX>
  )
  # Force CK_INSTALL_DIR/include before the ROCm system includes so hipcc
  # picks up the source-built ck_tile headers rather than /opt/rocm's copy.
  # Pass GPU target define so CK tile shape constexprs resolve correctly on
  # both the host and device compilation passes.
  target_compile_options(tile_ck_gemm PRIVATE
    -I${CK_INSTALL_DIR}/include
    -DCK_USE_GFX950
    -DCK_GFX950_SUPPORT
    -mllvm -enable-noalias-to-md-conversion=0
  )
else()
  message(STATUS "composable_kernel not found — skipping tile_ck_gemm example")
  message(STATUS "  Re-run cmake with -DCK_INSTALL_DIR=<path-to-ck-install>")
endif()

