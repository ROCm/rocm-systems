# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
#
# SPDX-License-Identifier: MIT

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
