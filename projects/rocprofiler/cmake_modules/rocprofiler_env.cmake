################################################################################
# Copyright (c) 2018 Advanced Micro Devices, Inc. All rights reserved.
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
################################################################################

add_library(rocprofiler-build-flags INTERFACE)
add_library(rocprofiler::build-flags ALIAS rocprofiler-build-flags)

target_compile_options(
    rocprofiler-build-flags
    INTERFACE $<$<COMPILE_LANGUAGE:C,CXX>:-W -Wall -Wextra -Wno-unused-parameter>
              $<$<COMPILE_LANGUAGE:CXX>:-fms-extensions>
              $<$<COMPILE_LANGUAGE:CXX>:$<$<CXX_COMPILER_ID:Clang>:-ferror-limit=1000000>>
    )
target_compile_definitions(rocprofiler-build-flags INTERFACE NEW_TRACE_API=1)

# Enable debug trace
if(ROCPROFILER_DEBUG_TRACE)
    target_compile_definitions(rocprofiler-build-flags INTERFACE DEBUG_TRACE=1)
endif()

# Enable direct loading of AQL-profile HSA extension
if(ROCPROFILER_LD_AQLPROFILE)
    target_compile_definitions(rocprofiler-build-flags INTERFACE ROCP_LD_AQLPROFILE=1)
endif()

if(NOT DEFINED ROCM_PATH)
    set(ROCM_PATH "/opt/rocm"  CACHE STRING "Default ROCM installation directory")
endif()

# Find hsa-runtime
find_package(
    hsa-runtime64 CONFIG REQUIRED
    HINTS ${CMAKE_PREFIX_PATH}
    PATHS ${ROCM_PATH}
    PATH_SUFFIXES lib/cmake/hsa-runtime64)

# Include path: /opt/rocm-ver/include. Go up one level to get ROCm  path
get_filename_component(ROCM_ROOT_DIR ${ROCM_PATH}/include DIRECTORY)

# Basic Tool Chain Information
message("----------Build-Type: ${CMAKE_BUILD_TYPE}")
message("------------Compiler: ${CMAKE_CXX_COMPILER}")
message("----Compiler-Version: ${CMAKE_CXX_COMPILER_VERSION}")
message("-------ROCM_ROOT_DIR: ${ROCM_ROOT_DIR}")
message("-----CMAKE_CXX_FLAGS: ${CMAKE_CXX_FLAGS}")
message("---CMAKE_PREFIX_PATH: ${CMAKE_PREFIX_PATH}")
message("---------GPU_TARGETS: ${GPU_TARGETS}")

if("${ROCM_ROOT_DIR}" STREQUAL "")
    message(FATAL_ERROR "ROCM_ROOT_DIR is not found.")
endif()

find_library(
    HSA_AMD_AQLPROFILE_LIBRARY
    NAMES hsa-amd-aqlprofile64
    HINTS ${CMAKE_PREFIX_PATH}
    PATHS ${ROCM_ROOT_DIR}
    PATH_SUFFIXES lib REQUIRED)
