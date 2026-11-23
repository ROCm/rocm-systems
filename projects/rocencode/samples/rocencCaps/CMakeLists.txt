# ##############################################################################
# Copyright (c) 2026 Advanced Micro Devices, Inc.
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
#
# ##############################################################################

cmake_minimum_required(VERSION 3.15)

project(rocenccaps LANGUAGES CXX)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

if(DEFINED ENV{ROCM_PATH})
  set(ROCM_PATH $ENV{ROCM_PATH} CACHE PATH "Default ROCm installation path")
elseif(NOT DEFINED ROCM_PATH)
  set(ROCM_PATH /opt/rocm CACHE PATH "Default ROCm installation path")
endif()

# No CMake package config is generated yet (that lands with the packaging
# work), so the installed library and header are located directly.
find_path(ROCENCODE_INCLUDE_DIR NAMES rocencode/rocencode.h HINTS ${ROCM_PATH}/include)
find_library(ROCENCODE_LIBRARY NAMES rocencode HINTS ${ROCM_PATH}/lib)

if(NOT ROCENCODE_INCLUDE_DIR OR NOT ROCENCODE_LIBRARY)
  message(FATAL_ERROR "rocencode not found under ${ROCM_PATH}; install it first.")
endif()

add_executable(rocenccaps rocenccaps.cpp)
target_compile_options(rocenccaps PRIVATE -Wall)
target_include_directories(rocenccaps PRIVATE ${ROCENCODE_INCLUDE_DIR})
target_link_libraries(rocenccaps ${ROCENCODE_LIBRARY})
