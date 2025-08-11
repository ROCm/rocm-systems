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

project(cmake_hiprtc_test CXX)

cmake_minimum_required(VERSION 3.10.2)

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

set(CMAKE_CXX_COMPILER "${ROCM_PATH}/llvm/bin/clang++")

# Find hiprtc
find_package(hiprtc REQUIRED)
# Find hip
find_package(hip REQUIRED)

# Create the excutable
if(TARGET build_cookbook)
set(EXCLUDE_OPTION EXCLUDE_FROM_ALL)
else()
set(EXCLUDE_OPTION )
endif()
add_executable(test ${EXCLUDE_OPTION} saxpy.cpp)

# Link with HIPRTC
target_link_libraries(test hiprtc)
# Link with HIP
target_link_libraries(test hip::device)

if(NOT BUILD_SHARED_LIBS)
  target_link_libraries(test hiprtc-builtins)
endif()

target_include_directories(test PRIVATE ../../common)

if(TARGET build_cookbook)
add_dependencies(build_cookbook test)
endif()
