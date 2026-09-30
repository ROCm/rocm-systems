# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
#
# SPDX-License-Identifier: MIT

cmake_minimum_required(VERSION 3.10.2)

# Windows ROCM_PATH setup
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

project(cmake_hip_device_test)

# Set compiler after project() for Linux to avoid reconfiguration loop in subdirectory builds
if(UNIX)
  set(CMAKE_CXX_COMPILER "${ROCM_PATH}/llvm/bin/clang++")
endif()

include_directories(../../common)

# Find hip
find_package(hip REQUIRED)

# Create the excutable
if(TARGET build_cookbook)
set(EXCLUDE_OPTION EXCLUDE_FROM_ALL)
else()
set(EXCLUDE_OPTION )
endif()
add_executable(test_cpp ${EXCLUDE_OPTION} square.cpp)

target_include_directories(test_cpp PRIVATE ../../common)

# Add GPU architecture for device code compilation
if(WIN32 AND DEFINED CMAKE_HIP_ARCHITECTURES)
  foreach(_arch ${CMAKE_HIP_ARCHITECTURES})
    target_compile_options(test_cpp PRIVATE --offload-arch=${_arch})
  endforeach()
endif()

# Link with HIP
target_link_libraries(test_cpp hip::device)

if(TARGET build_cookbook)
add_dependencies(build_cookbook test_cpp)
endif()
