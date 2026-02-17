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
# Compile test kernel to HSACO for hipmodule_init test
###############################################################################
# Get GPU architectures - use a simplified list for HSACO generation
set(HSACO_GPU_TARGETS "gfx90a" "gfx942" "gfx950" "gfx1100")

# Compile test_kernel.hip to HSACO for each GPU target
foreach(GPU_TARGET ${HSACO_GPU_TARGETS})
  # Create safe filename by replacing special chars
  string(REPLACE ":" "_" SAFE_GPU_TARGET "${GPU_TARGET}")
  set(HSACO_TARGET "${CMAKE_CURRENT_BINARY_DIR}/test_kernel_${SAFE_GPU_TARGET}.hsaco")

  # Intermediate bitcode file
  set(BC_FILE "${CMAKE_CURRENT_BINARY_DIR}/test_kernel_${SAFE_GPU_TARGET}.bc")
  set(OBJ_FILE "${CMAKE_CURRENT_BINARY_DIR}/test_kernel_${SAFE_GPU_TARGET}.o")

  add_custom_command(
    OUTPUT ${BC_FILE}
    COMMAND ${ROCM_PATH}/lib/llvm/bin/clang++
      -x hip --cuda-device-only -std=c++17 -emit-llvm
      --offload-arch=${GPU_TARGET}
      -I${CMAKE_SOURCE_DIR}/include
      -I${CMAKE_BINARY_DIR}/include
      -I${CMAKE_SOURCE_DIR}/src
      -c ${CMAKE_CURRENT_SOURCE_DIR}/test_kernel.hip
      -o ${BC_FILE}
    DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/test_kernel.hip
    COMMENT "Compiling test_kernel.hip to bitcode for ${GPU_TARGET}"
    VERBATIM
  )

  add_custom_command(
    OUTPUT ${OBJ_FILE}
    COMMAND ${ROCM_PATH}/lib/llvm/bin/llc
      -march=amdgcn -mcpu=${GPU_TARGET} -filetype=obj
      ${BC_FILE} -o ${OBJ_FILE}
    DEPENDS ${BC_FILE}
    COMMENT "Compiling bitcode to object file for ${GPU_TARGET}"
    VERBATIM
  )

  add_custom_command(
    OUTPUT ${HSACO_TARGET}
    COMMAND ${ROCM_PATH}/lib/llvm/bin/ld.lld
      -shared ${OBJ_FILE} -o ${HSACO_TARGET}
    DEPENDS ${OBJ_FILE}
    COMMENT "Linking object file to HSACO for ${GPU_TARGET}"
    VERBATIM
  )
  list(APPEND TEST_KERNEL_HSACOS ${HSACO_TARGET})
endforeach()

# Add custom target to build all HSACOs
add_custom_target(test_kernels_hsaco ALL DEPENDS ${TEST_KERNEL_HSACOS})

# Install HSACO files
install(FILES ${TEST_KERNEL_HSACOS}
        DESTINATION ${CMAKE_INSTALL_DATADIR}/rocshmem/test_kernels
        COMPONENT tests)

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

# Option to use pre-compiled HSACO instead of HIPRTC
option(USE_PRECOMPILED_HSACO "Use pre-compiled HSACO for hipmodule_init test instead of HIPRTC" OFF)

target_compile_definitions(
  ${PROJECT_NAME}
  PRIVATE
    $<$<TARGET_EXISTS:PMIx::pmix>:HAVE_PMIX=1>
    CMAKE_SOURCE_DIR="${CMAKE_SOURCE_DIR}"
    CMAKE_BINARY_DIR="${CMAKE_BINARY_DIR}"
    $<$<BOOL:${USE_PRECOMPILED_HSACO}>:USE_PRECOMPILED_HSACO>
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
    $<IF:$<BOOL:${USE_PRECOMPILED_HSACO}>,,hiprtc>
)
