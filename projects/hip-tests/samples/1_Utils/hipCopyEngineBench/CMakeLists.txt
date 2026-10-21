# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
#
# SPDX-License-Identifier: MIT

cmake_minimum_required(VERSION 3.21)

if(UNIX)
  if(NOT DEFINED ROCM_PATH)
    if(DEFINED ENV{ROCM_PATH})
      set(ROCM_PATH $ENV{ROCM_PATH} CACHE STRING "ROCM Path")
    else()
      set(ROCM_PATH "/opt/rocm" CACHE STRING "Default ROCM installation directory.")
    endif()
  endif()
  # Search for rocm in common locations
  list(APPEND CMAKE_PREFIX_PATH ${ROCM_PATH})
endif()

project(hipCopyEngineBench LANGUAGES CXX HIP)

set(CMAKE_BUILD_TYPE Release)

# Detect GPU architectures and setup OFFLOAD_ARCH_FLAGS
include(${CMAKE_CURRENT_SOURCE_DIR}/../../common/DetectGPUArchs.cmake)

# hip_ext_copy_engine.h only exists in a HIP built with the kernel-side copy engine, so take HIP
# from the first CMAKE_PREFIX_PATH entry that has it, not from the compiler's ROCm root.
find_package(hip REQUIRED)

# A HIP without the copy-engine header cannot build these, and must not fail the samples build.
find_path(HIP_EXT_COPY_ENGINE_INCLUDE_DIR hip/hip_ext_copy_engine.h
  HINTS ${hip_DIR}/../../../include
  NO_DEFAULT_PATH)
if(NOT HIP_EXT_COPY_ENGINE_INCLUDE_DIR)
  message(STATUS "hipCopyEngineBench: hip/hip_ext_copy_engine.h not found in ${hip_DIR}/../../.., skipping")
  return()
endif()

if(TARGET build_utils)
  set(EXCLUDE_OPTION EXCLUDE_FROM_ALL)
else()
  set(EXCLUDE_OPTION)
endif()

set(COPY_ENGINE_BENCHMARKS copy-engine-bw copy-engine-latency)

foreach(benchmark ${COPY_ENGINE_BENCHMARKS})
  set_source_files_properties(${benchmark}.cpp PROPERTIES LANGUAGE HIP)
  add_executable(${benchmark} ${EXCLUDE_OPTION} ${benchmark}.cpp copy_engine_bench_common.hpp)
  set_target_properties(${benchmark} PROPERTIES HIP_STANDARD 17 HIP_STANDARD_REQUIRED ON)
  target_link_libraries(${benchmark} PRIVATE hip::device)
  target_compile_options(${benchmark} PRIVATE
    --hip-path=${hip_DIR}/../../..
    -Wall -Wextra)
  if(UNIX)
    # Set RPATH so executables can find HIP libraries at runtime
    set_target_properties(${benchmark} PROPERTIES
      BUILD_RPATH "${hip_DIR}/../.."
      INSTALL_RPATH "${hip_DIR}/../.."
    )
  endif()
endforeach()

if(TARGET build_utils)
  add_dependencies(build_utils ${COPY_ENGINE_BENCHMARKS})
endif()
