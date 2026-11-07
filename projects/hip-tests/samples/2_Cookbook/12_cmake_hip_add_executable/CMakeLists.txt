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

set(CMAKE_HIP_ARCHITECTURES OFF)

project(12_cmake_hip_add_executable LANGUAGES CXX)

# FindHIP.cmake compatibility shim — exercises the legacy find_package(HIP MODULE)
# path and HIP_ADD_EXECUTABLE macro, which internally delegate to
# find_package(hip CONFIG) and add_executable() + hip::device.
if(WIN32)
  set(CMAKE_MODULE_PATH "${HIP_PATH}/cmake" ${CMAKE_MODULE_PATH})
else()
  set(CMAKE_MODULE_PATH "${ROCM_PATH}/lib/cmake/hip" ${CMAKE_MODULE_PATH})
endif()

find_package(HIP MODULE REQUIRED)
message(STATUS "Found HIP: ${HIP_VERSION}")

if(TARGET build_cookbook)
  set(EXCLUDE_OPTION EXCLUDE_FROM_ALL)
else()
  set(EXCLUDE_OPTION)
endif()

hip_add_executable(MatrixTranspose1 ${EXCLUDE_OPTION} MatrixTranspose.cpp)

target_include_directories(MatrixTranspose1 PRIVATE ../../common)

# Set RPATH so executable can find HIP libraries at runtime
if(UNIX)
  set_target_properties(MatrixTranspose1 PROPERTIES
    BUILD_RPATH "${ROCM_PATH}/lib"
    INSTALL_RPATH "${ROCM_PATH}/lib"
  )
endif()

if(TARGET build_cookbook)
  add_dependencies(build_cookbook MatrixTranspose1)
endif()
