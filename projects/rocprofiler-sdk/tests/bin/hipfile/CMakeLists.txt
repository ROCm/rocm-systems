# MIT License
#
# Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
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

cmake_minimum_required(VERSION 3.21.0 FATAL_ERROR)

project(
    rocprofiler-sdk-tests-bin-hipfile
    LANGUAGES CXX
    VERSION 0.0.0)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(hipfile QUIET CONFIG)

if(NOT TARGET hip::hipfile)
    return()
endif()

add_executable(hipfile-trace-client hipfile_trace.cpp)
target_link_libraries(hipfile-trace-client PRIVATE hip::hipfile)
target_compile_definitions(hipfile-trace-client PRIVATE __HIP_PLATFORM_AMD__)

if(DEFINED ROCM_PATH)
    set(_hipfile_sysdeps_library_dir "${ROCM_PATH}/lib/rocm_sysdeps/lib")
elseif(DEFINED ENV{ROCM_PATH})
    set(_hipfile_sysdeps_library_dir "$ENV{ROCM_PATH}/lib/rocm_sysdeps/lib")
else()
    set(_hipfile_sysdeps_library_dir "/opt/rocm/lib/rocm_sysdeps/lib")
endif()

if(EXISTS "${_hipfile_sysdeps_library_dir}")
    target_link_options(hipfile-trace-client PRIVATE
                        "LINKER:-rpath-link,${_hipfile_sysdeps_library_dir}")
endif()
