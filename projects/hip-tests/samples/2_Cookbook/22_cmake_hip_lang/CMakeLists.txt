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

if(UNIX)
  if(NOT DEFINED ROCM_PATH)
    set(ROCM_PATH "/opt/rocm" CACHE STRING "Default ROCM installation directory.")
    list(APPEND CMAKE_PREFIX_PATH ${ROCM_PATH})
  endif()
endif()

if( NOT DEFINED CMAKE_HIP_COMPILER_ROCM)
 set(CMAKE_HIP_COMPILER_ROCM_ROOT ${ROCM_PATH})
endif()

set(CMAKE_HIP_COMPILER ${ROCM_PATH}/llvm/bin/clang++)

project(cmake_hip_lang_support VERSION 1.0
        DESCRIPTION "HIP Language Support with upstream CMake"
        LANGUAGES HIP)

# Create the executable
if(TARGET build_cookbook)
set(EXCLUDE_OPTION EXCLUDE_FROM_ALL)
else()
set(EXCLUDE_OPTION )
endif()
add_executable(square2 ${EXCLUDE_OPTION} square.hip)

target_include_directories(square2 PRIVATE ../../common)

if(TARGET build_cookbook)
add_dependencies(build_cookbook square2)
endif()
