# MIT License
#
# Copyright (c) 2025 Advanced Micro Devices, Inc. All rights reserved.
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

cmake_minimum_required(VERSION 3.18.4 FATAL_ERROR)

project(rocprofiler-systems-examples LANGUAGES C CXX)

if("${CMAKE_BUILD_TYPE}" STREQUAL "")
    set(CMAKE_BUILD_TYPE "RelWithDebInfo" CACHE STRING "Build type" FORCE)
endif()

string(TOUPPER "${CMAKE_BUILD_TYPE}" BUILD_TYPE)

set(CMAKE_VISIBILITY_INLINES_HIDDEN OFF)
set(CMAKE_CXX_VISIBILITY_PRESET "default")
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_CLANG_TIDY)
set(CMAKE_INSTALL_DEFAULT_COMPONENT_NAME examples)

if(ROCPROFSYS_BUILD_DEBUG)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -g3 -fno-omit-frame-pointer")
endif()

option(BUILD_SHARED_LIBS "Build dynamic libraries" ON)

if(CMAKE_PROJECT_NAME STREQUAL "rocprofiler-systems")
    set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR})
    set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/${CMAKE_INSTALL_LIBDIR})
    set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/${CMAKE_INSTALL_LIBDIR})
    rocprofiler_systems_add_option(ROCPROFSYS_INSTALL_EXAMPLES
                                   "Install rocprofiler-systems examples" OFF
    )
else()
    option(ROCPROFSYS_INSTALL_EXAMPLES "Install rocprofiler-systems examples" ON)
endif()

if(ROCPROFSYS_INSTALL_EXAMPLES)
    include(GNUInstallDirs)
endif()

set(ROCPROFSYS_EXAMPLE_ROOT_DIR ${CMAKE_CURRENT_LIST_DIR} CACHE INTERNAL "")
# defines function for creating causal profiling exes
include(${CMAKE_CURRENT_LIST_DIR}/causal-helpers.cmake)

add_subdirectory(transpose)
add_subdirectory(parallel-overhead)
add_subdirectory(code-coverage)
add_subdirectory(user-api)
add_subdirectory(openmp)
add_subdirectory(mpi)
add_subdirectory(python)
add_subdirectory(lulesh)
add_subdirectory(rccl)
add_subdirectory(rewrite-caller)
add_subdirectory(causal)
add_subdirectory(trace-time-window)
add_subdirectory(fork)
add_subdirectory(videodecode)
add_subdirectory(jpegdecode)
add_subdirectory(roctx)
