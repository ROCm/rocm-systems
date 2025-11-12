# Copyright (c) 2020 - 2023 Advanced Micro Devices, Inc. All Rights Reserved.
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
project(hipInfo)

# flag is set to ON in compute build for windows
option(HIPINFO_INTERNAL_BUILD "Enable building hipInfo from compute" OFF)

cmake_minimum_required(VERSION 3.10)
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

set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} --hip-path=${CMAKE_PREFIX_PATH}")
# Find hip
find_package(hip)

# Set compiler and linker
set(CMAKE_BUILD_TYPE Release)

# Create the excutable
if(TARGET build_utils)
set(EXCLUDE_OPTION EXCLUDE_FROM_ALL)
else()
set(EXCLUDE_OPTION )
endif()
add_executable(hipInfo ${EXCLUDE_OPTION} hipInfo.cpp)

# Link with HIP
target_link_libraries(hipInfo hip::host)

target_include_directories(hipInfo PRIVATE ../../common)

# Used only when make install is called
# when hipInfo is built as part of compute project
# hipInfo.exe will be installed to install/hip/bin path
if (WIN32 AND HIPINFO_INTERNAL_BUILD)
  install(FILES ${PROJECT_BINARY_DIR}/hipInfo.exe DESTINATION ${CMAKE_INSTALL_PREFIX})
endif()

if(TARGET build_utils)
add_dependencies(build_utils hipInfo)
endif()
