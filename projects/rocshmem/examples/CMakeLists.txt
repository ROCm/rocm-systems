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
project(rocshmem_examples VERSION 1.0.0 LANGUAGES CXX)

find_package(MPI)
find_package(hip REQUIRED PATHS /opt/rocm)
if (NOT TARGET roc::rocshmem)
  find_package(rocshmem REQUIRED PATHS /opt/rocm)
endif()

###############################################################################
# SOURCES
###############################################################################
set(EXAMPLE_SOURCES
  rocshmem_allreduce_test.cc
  rocshmem_alltoall_test.cc
  rocshmem_broadcast_test.cc
  rocshmem_getmem_test.cc
  rocshmem_put_signal_test.cc
)

if (MPI_CXX_FOUND)
  list(APPEND EXAMPLE_SOURCES
    rocshmem_init_attr_test.cc)
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
