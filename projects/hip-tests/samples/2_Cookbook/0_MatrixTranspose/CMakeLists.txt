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

project(MatrixTranspose LANGUAGES CXX HIP)

set(CMAKE_BUILD_TYPE Release)

# Create the excutable
if(TARGET build_cookbook)
  set(EXCLUDE_OPTION EXCLUDE_FROM_ALL)
else()
  set(EXCLUDE_OPTION)
endif()

# Mark source file as HIP code (contains __global__ kernels)
set_source_files_properties(MatrixTranspose.cpp PROPERTIES LANGUAGE HIP)

add_executable(MatrixTranspose ${EXCLUDE_OPTION} MatrixTranspose.cpp)

target_include_directories(MatrixTranspose PRIVATE ../../common)

# Set RPATH so executable can find HIP libraries at runtime
if(UNIX)
  set_target_properties(MatrixTranspose PROPERTIES
    BUILD_RPATH "${ROCM_PATH}/lib"
    INSTALL_RPATH "${ROCM_PATH}/lib"
  )
endif()

if(TARGET build_cookbook)
  add_dependencies(build_cookbook MatrixTranspose)
endif()
