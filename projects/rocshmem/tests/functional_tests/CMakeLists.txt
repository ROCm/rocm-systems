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
project(rocshmem_functional_tests VERSION 1.0.0 LANGUAGES CXX)

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
    barrier_all_tester.cpp
    sync_all_tester.cpp
    test_driver.cpp
    tester.cpp
    tester_arguments.cpp
    ping_pong_tester.cpp
    ping_all_tester.cpp
    primitive_tester.cpp
    primitive_mr_tester.cpp
    default_ctx_primitive_tester.cpp
    team_ctx_primitive_tester.cpp
    team_ctx_infra_tester.cpp
    amo_bitwise_tester.cpp
    amo_extended_tester.cpp
    amo_standard_tester.cpp
    random_access_tester.cpp
    shmem_ptr_tester.cpp
    signaling_operations_tester.cpp
    signaling_operations_tester.hpp
    workgroup_primitives.cpp
    empty_tester.cpp
    wavefront_primitives.cpp
    flood_tester.cpp
)

###############################################################################
# ROCSHMEM
###############################################################################
if (BUILD_TESTS_ONLY)
  #TODO these find_packages should be performed as-needed in rocshmem-config.cmake
  find_package(hip REQUIRED PATHS /opt/rocm)
  find_package(MPI)
  find_package(Threads REQUIRED)
  find_package(rocshmem REQUIRED PATHS /opt/rocm)

  target_include_directories(
    ${PROJECT_NAME}
    PRIVATE
      $<BUILD_INTERFACE:${CMAKE_BINARY_DIR}>
      $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/../../src>
  )
endif()

find_package(PMIx)

target_compile_definitions(
  ${PROJECT_NAME}
  PRIVATE
    $<$<TARGET_EXISTS:PMIx::pmix>:HAVE_PMIX=1>
)

configure_file(../../scripts/functional_tests/driver.sh rocshmem_functional_driver.sh COPYONLY)
rocm_install(PROGRAMS ${CMAKE_CURRENT_BINARY_DIR}/rocshmem_functional_driver.sh COMPONENT tests DESTINATION ${CMAKE_INSTALL_DATADIR}/rocshmem)
rocm_install(TARGETS rocshmem_functional_tests COMPONENT tests)

target_link_libraries(
  ${PROJECT_NAME}
  PRIVATE
    roc::rocshmem
    $<TARGET_NAME_IF_EXISTS:MPI::MPI_CXX>
    $<TARGET_NAME_IF_EXISTS:PMIx::pmix>
)
