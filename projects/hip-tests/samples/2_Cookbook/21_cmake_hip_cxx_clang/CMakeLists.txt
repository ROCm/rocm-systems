# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
#
# SPDX-License-Identifier: MIT

cmake_minimum_required(VERSION 3.21.3)

# Path setup (must be before project())
if(WIN32)
  if(NOT DEFINED ROCM_PATH)
    if(DEFINED ENV{ROCM_PATH})
      set(ROCM_PATH $ENV{ROCM_PATH} CACHE STRING "ROCM Path")
    elseif(DEFINED ENV{HIP_PATH})
      set(ROCM_PATH $ENV{HIP_PATH} CACHE STRING "ROCM Path from HIP_PATH")
    endif()
  endif()
  if(DEFINED ROCM_PATH)
    list(APPEND CMAKE_PREFIX_PATH ${ROCM_PATH})
  endif()
  set(CMAKE_CXX_COMPILER "${ROCM_PATH}/lib/llvm/bin/clang++.exe")
elseif(UNIX)
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

project(cmake_cxx_amdclang++_test
        DESCRIPTION "Verifies CXX Language Support with amdclang++"
        LANGUAGES CXX)

# Set compiler after project() for Linux to avoid reconfiguration loop in subdirectory builds
if(UNIX)
  set(CMAKE_CXX_COMPILER "${ROCM_PATH}/llvm/bin/clang++")
endif()

# Find HIP
find_package(hip REQUIRED)

# Create the excutable
if(TARGET build_cookbook)
set(EXCLUDE_OPTION EXCLUDE_FROM_ALL)
else()
set(EXCLUDE_OPTION )
endif()
add_executable(square1 ${EXCLUDE_OPTION} square.cpp)

target_include_directories(square1 PRIVATE ../../common)

# Add GPU architecture for device code compilation (Windows)
if(WIN32 AND DEFINED CMAKE_HIP_ARCHITECTURES)
  foreach(_arch ${CMAKE_HIP_ARCHITECTURES})
    target_compile_options(square1 PRIVATE --offload-arch=${_arch})
  endforeach()
endif()

# Link with HIP
target_link_libraries(square1 hip::device)

if(TARGET build_cookbook)
add_dependencies(build_cookbook square1)
endif()
