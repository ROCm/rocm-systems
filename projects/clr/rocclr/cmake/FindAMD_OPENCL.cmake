# Copyright (c) 2020 - 2021 Advanced Micro Devices, Inc. All rights reserved.
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

if(AMD_OPENCL_FOUND)
  return()
endif()

find_path(AMD_OPENCL_INCLUDE_DIR cl.h
  HINTS
    ${AMD_OPENCL_PATH}
  PATHS
    # gerrit repo name
    ${CMAKE_SOURCE_DIR}/opencl
    ${CMAKE_SOURCE_DIR}/../opencl
    ${CMAKE_SOURCE_DIR}/../../opencl
    # github repo name
    ${CMAKE_SOURCE_DIR}/ROCm-OpenCL-Runtime
    ${CMAKE_SOURCE_DIR}/../ROCm-OpenCL-Runtime
    ${CMAKE_SOURCE_DIR}/../../ROCm-OpenCL-Runtime
    # jenkins repo name
    ${CMAKE_SOURCE_DIR}/opencl-on-vdi
    ${CMAKE_SOURCE_DIR}/../opencl-on-vdi
    ${CMAKE_SOURCE_DIR}/../../opencl-on-vdi
    ${CMAKE_SOURCE_DIR}/opencl-on-rocclr
    ${CMAKE_SOURCE_DIR}/../opencl-on-rocclr
    ${CMAKE_SOURCE_DIR}/../../opencl-on-rocclr
  PATH_SUFFIXES
    khronos/headers/opencl2.2/CL
  NO_DEFAULT_PATH)

include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(AMD_OPENCL
  "\nAMD OpenCL not found"
  AMD_OPENCL_INCLUDE_DIR)
mark_as_advanced(AMD_OPENCL_INCLUDE_DIR)

set(AMD_OPENCL_DEFS
  -DHAVE_CL2_HPP
  -DOPENCL_MAJOR=2
  -DOPENCL_MINOR=1
  -DOPENCL_C_MAJOR=2
  -DOPENCL_C_MINOR=0
  -DCL_TARGET_OPENCL_VERSION=220
  -DCL_USE_DEPRECATED_OPENCL_1_0_APIS
  -DCL_USE_DEPRECATED_OPENCL_1_1_APIS
  -DCL_USE_DEPRECATED_OPENCL_1_2_APIS
  -DCL_USE_DEPRECATED_OPENCL_2_0_APIS)
mark_as_advanced(AMD_OPENCL_DEFS)

set(AMD_OPENCL_INCLUDE_DIRS
  ${AMD_OPENCL_INCLUDE_DIR}
  ${AMD_OPENCL_INCLUDE_DIR}/..
  ${AMD_OPENCL_INCLUDE_DIR}/../..
  ${AMD_OPENCL_INCLUDE_DIR}/../../..
  ${AMD_OPENCL_INCLUDE_DIR}/../../../..
  ${AMD_OPENCL_INCLUDE_DIR}/../../../../amdocl)
mark_as_advanced(AMD_OPENCL_INCLUDE_DIRS)
