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
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
# FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS
# IN THE SOFTWARE.
###############################################################################

cmake_minimum_required(VERSION 3.20 FATAL_ERROR)

###############################################################################
# AVOID IN SOURCE BUILD
###############################################################################
if(CMAKE_SOURCE_DIR STREQUAL CMAKE_BINARY_DIR AND
   CMAKE_CURRENT_SOURCE_DIR STREQUAL CMAKE_SOURCE_DIR)
  set(MSG "")
  message(STATUS "Warning! Building from the source directory is not recommended")
  message(STATUS "If unintended, please remove 'CMakeCache.txt' and 'CMakeFiles'")
  message(STATUS "and build from a separate directory")
  message(FATAL_ERROR "In-source build")
endif()

###############################################################################
# CONFIGURATION OPTIONS
###############################################################################
option(DEBUG "Enable debug trace" OFF)
option(PROFILE "Enable statistics and timing support" OFF)
option(USE_RO "Enable RO conduit" ON)
option(USE_IPC "Enable IPC support (using HIP)" OFF)
option(USE_GDA "Enable GDA conduit" OFF)
option(USE_THREADS "Enable workgroup threads to share network queues" OFF)
option(USE_WF_COAL "Enable wavefront message coalescing" OFF)
option(USE_HEAP_DEVICE_FINEGRAIN "Heap uses GPU memory in finegrain mode" ON)
option(USE_HEAP_DEVICE_UNCACHED "Heap uses GPU memory in uncached mode" OFF)
option(USE_HEAP_DEVICE_COARSEGRAIN "Heap uses GPU memory in coarsegrain mode" OFF)
option(USE_HEAP_MANAGED "Heap uses managed memory" OFF)
option(USE_HEAP_HOST_HIP "Heap uses pinned host memory allocated with hip api" OFF)
option(USE_HEAP_HOST "Heap uses host memory allocated with malloc/free" OFF)
option(USE_ALLOC_DLMALLOC "Enable dlmalloc device memory allocator" ON)
option(USE_ALLOC_POW2BINS "Enable legacy Pow2Bins device memory allocator" OFF)
option(USE_FUNC_CALL "Force compiler to use function calls on library API" OFF)
option(USE_SHARED_CTX "Request support for shared ctx between WG" OFF)
option(USE_SINGLE_NODE "Enable single node support only." OFF)
option(USE_HDP_FLUSH "Force flush the HDP cache." OFF)
option(USE_HDP_FLUSH_HOST_SIDE "Use a polling thread to flush the HDP cache on the host." OFF)

option(BUILD_FUNCTIONAL_TESTS "Build the functional tests (Requires MPI)" OFF)
option(BUILD_EXAMPLES "Build the examples" ON)
option(BUILD_UNIT_TESTS "Build the unit tests (Requires MPI)" OFF)
option(BUILD_TESTS_ONLY "Build only tests. Used to link agains rocSHMEM in a ROCm Release" OFF)
option(BUILD_TOOLS "Build binary tools (e.g., rocshmem_info)" ON)

option(BUILD_LOCAL_GPU_TARGET_ONLY "Build only for GPUs detected on this machine" OFF)
option(BUILD_CODE_COVERAGE "Build with code coverage flags (gcc only)" OFF)

option(GDA_IONIC "Build for AMD Pensando IONIC RDMA provider" OFF)
option(GDA_BNXT "Build for Broadcom RDMA provider" OFF)
option(GDA_MLX5 "Build for Mellanox MLX5 RDMA provider" OFF)

set(USE_EXTERNAL_MPI AUTO CACHE STRING "Link with an external MPI (required if used MPI is ABI incompatible with Open MPI v5)")
set_property(CACHE USE_EXTERNAL_MPI PROPERTY STRINGS AUTO ON OFF)

###############################################################################
# PROJECT
###############################################################################
include(${CMAKE_SOURCE_DIR}/cmake/setup_project.cmake)

## Setup VERSION
file(READ include/rocshmem/rocshmem.hpp header_text)
if("${header_text}" MATCHES "constexpr char VERSION\\[\\] *= \"([0-9]+)\\.([0-9]+)\\.([0-9]+)\";")
  set(VERSION_STRING ${CMAKE_MATCH_1}.${CMAKE_MATCH_2}.${CMAKE_MATCH_3})
else()
  message(FATAL_ERROR "Failed to parse Version")
endif()
message(STATUS "rocSHMEM Version: " "${VERSION_STRING}")

project(rocshmem VERSION ${VERSION_STRING} LANGUAGES CXX)

find_package(ROCmCMakeBuildTools PATHS /opt/rocm)
include(ROCMCreatePackage)
include(ROCMInstallTargets)
include(ROCMCheckTargetIds)

rocm_setup_version(VERSION ${VERSION_STRING})


#############################################################################
# SET GPU ARCHITECTURES
#############################################################################
include(cmake/rocm_local_targets.cmake)

set(DEFAULT_GPUS
    gfx90a:xnack-;
    gfx90a:xnack+;
    gfx1100;
    gfx1201;
    gfx942)

if(${ROCM_MAJOR_VERSION} GREATER 6)
  list(APPEND DEFAULT_GPUS gfx950)
endif()

if($ENV{BUILD_LOCAL_GPU_TARGET_ONLY})
  set(BUILD_LOCAL_GPU_TARGET_ONLY ON)
endif()

if (BUILD_LOCAL_GPU_TARGET_ONLY)
  message(STATUS "Building only for local GPU target")
  if (COMMAND rocm_local_targets)
    rocm_local_targets(DEFAULT_GPUS)
  else()
    message(WARNING "Unable to determine local GPU targets. Falling back to default GPUs.")
  endif()
endif()

set(DEFAULT_GPU_TARGETS "${DEFAULT_GPUS}" CACHE STRING
    "Target default GPUs if GPU_TARGETS is not defined.")

if (COMMAND rocm_check_target_ids)
  message(STATUS "Checking for ROCm support for GPU targets: " "${DEFAULT_GPU_TARGETS}")
  rocm_check_target_ids(SUPPORTED_GPUS TARGETS ${DEFAULT_GPU_TARGETS})
else()
  message(WARNING "Unable to check for supported GPU targets.")
  set(SUPPORTED_GPUS ${DEFAULT_GPU_TARGETS})
endif()

set(GPU_TARGETS "${SUPPORTED_GPUS}" CACHE STRING "GPU architectures to compile for")

message(STATUS "Compiling for ${GPU_TARGETS}")

###############################################################################
# CREATE ROCSHMEM LIBRARY
###############################################################################
if (NOT BUILD_TESTS_ONLY)
  add_library(${PROJECT_NAME})
  add_library(roc::${PROJECT_NAME} ALIAS ${PROJECT_NAME})
  add_subdirectory(src)

  #############################################################################
  # PACKAGE DEPENDENCIES
  #############################################################################
  if (NOT USE_EXTERNAL_MPI STREQUAL "OFF")
    find_package(MPI)
  else()
    message ("-- External MPI detection disabled by user")
  endif()

  if (MPI_FOUND)
    set(HAVE_EXTERNAL_MPI ON)
  else()
    set(HAVE_EXTERNAL_MPI OFF)
    set(BUILD_UNIT_TESTS OFF)
  endif()

  if (USE_EXTERNAL_MPI STREQUAL "ON")
    if(NOT HAVE_EXTERNAL_MPI)
      message(FATAL_ERROR "External MPI support requested but MPI support not found. Build Aborted")
    endif()
  endif()

  find_package(hip REQUIRED PATHS /opt/rocm)
  find_package(hsa-runtime64 REQUIRED)

  set(CMAKE_THREAD_PREFER_PTHREAD TRUE)
  set(THREADS_PREFER_PTHREAD_FLAG TRUE)
  find_package(Threads REQUIRED)

  configure_file(cmake/rocshmem_config.h.in include/rocshmem/rocshmem_config.h)

  #############################################################################
  # LINKING AND INCLUDE DIRECTORIES
  #############################################################################
  target_compile_options(
    ${PROJECT_NAME}
    PUBLIC
      -fgpu-rdc
  )

  target_include_directories(
    ${PROJECT_NAME}
    PUBLIC
      $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
      $<BUILD_INTERFACE:${CMAKE_BINARY_DIR}/include>            # rocshmem_config.h
      $<BUILD_INTERFACE:${CMAKE_BINARY_DIR}/include/rocshmem>   # rocshmem_config.h from rocshmem.hpp
      $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/src>
      $<INSTALL_INTERFACE:include>
  )

  target_link_libraries(
    ${PROJECT_NAME}
    PUBLIC
      $<$<BOOL:${HAVE_EXTERNAL_MPI}>:MPI::MPI_CXX>
      Threads::Threads
      hip::device
      hip::host
      dl
      hsa-runtime64::hsa-runtime64
      -fgpu-rdc
  )

  if(${ROCM_MAJOR_VERSION} LESS 7)
    # ROCm 6.x requires us to explicitly enable warp sync builtins
    target_compile_definitions(${PROJECT_NAME} PRIVATE HIP_ENABLE_WARP_SYNC_BUILTINS=1)
  endif()

  #############################################################################
  # INSTALL
  #############################################################################
  include(ROCMInstallTargets)
  include(ROCMCreatePackage)

  rocm_install(TARGETS rocshmem)

  rocm_install(
    DIRECTORY ${CMAKE_SOURCE_DIR}/include/
    DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
  )

  rocm_install(
    FILES "${CMAKE_BINARY_DIR}/include/rocshmem/rocshmem_config.h"
    DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/rocshmem
  )

  if (BUILD_TOOLS)
    rocm_install(
      PROGRAMS "${CMAKE_BINARY_DIR}/src/tools/rocshmem_info"
      DESTINATION ${CMAKE_INSTALL_BINDIR}
    )
  endif()

  rocm_package_add_dependencies(
    DEPENDS
      hsa-rocr
      hip-runtime-amd
      rocm-dev
  )

  rocm_export_targets(
    TARGETS roc::rocshmem
    NAMESPACE roc::
  )
  include(ROCMPackageConfigHelpers)
  include(ROCMClients)
  rocm_package_setup_component(clients)
  rocm_package_setup_client_component(tests PACKAGE_NAME tests)

  rocm_create_package(
    NAME "rocSHMEM"
    DESCRIPTION "ROCm OpenSHMEM (rocSHMEM)"
    MAINTAINER "rocSHMEM Maintainer <rocshmem-maintainer@amd.com>"
  )
endif (NOT BUILD_TESTS_ONLY)

###############################################################################
# TEST SUBDIRECTORIES
###############################################################################
add_subdirectory(tests)

if (BUILD_EXAMPLES)
  add_subdirectory(examples)
endif()

