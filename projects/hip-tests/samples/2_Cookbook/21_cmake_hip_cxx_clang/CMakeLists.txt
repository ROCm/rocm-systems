# Copyright (C) 2022 Advanced Micro Devices, Inc. All Rights Reserved.
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
# THE SOFTWARE.

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
