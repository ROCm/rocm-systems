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

if(AMD_PAL_FOUND)
  return()
endif()

find_path(AMD_ASIC_REG_INCLUDE_DIR nv_id.h
  HINTS
    ${AMD_DRIVERS_PATH}
  PATHS
    # p4 repo layout
    ${CMAKE_SOURCE_DIR}/drivers
    ${CMAKE_SOURCE_DIR}/../drivers
    ${CMAKE_SOURCE_DIR}/../../drivers
    # github ent repo layout
    ${CMAKE_SOURCE_DIR}/drivers/drivers
    ${CMAKE_SOURCE_DIR}/../drivers/drivers
    ${CMAKE_SOURCE_DIR}/../../drivers/drivers
  PATH_SUFFIXES
    inc/asic_reg)

find_path(AMD_HSAIL_INCLUDE_DIR hsa.h
  HINTS
    ${AMD_SC_PATH}
  PATHS
    ${CMAKE_SOURCE_DIR}/sc
    ${CMAKE_SOURCE_DIR}/../sc
    ${CMAKE_SOURCE_DIR}/../../sc
  PATH_SUFFIXES
    HSAIL/include)

find_path(AMD_PAL_INCLUDE_DIR pal.h
  HINTS
    ${AMD_PAL_PATH}
  PATHS
    ${CMAKE_SOURCE_DIR}/pal
    ${CMAKE_SOURCE_DIR}/../pal
    ${CMAKE_SOURCE_DIR}/../../pal
  PATH_SUFFIXES
    inc/core)

include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(AMD_PAL
  "\nPAL not found"
  AMD_ASIC_REG_INCLUDE_DIR AMD_HSAIL_INCLUDE_DIR AMD_PAL_INCLUDE_DIR)
mark_as_advanced(AMD_ASIC_REG_INCLUDE_DIR AMD_HSAIL_INCLUDE_DIR AMD_PAL_INCLUDE_DIR)

set(GLOBAL_ROOT_SRC_DIR "${AMD_ASIC_REG_INCLUDE_DIR}/../../..")
set(PAL_SC_PATH "${AMD_HSAIL_INCLUDE_DIR}/../..")
add_subdirectory("${AMD_PAL_INCLUDE_DIR}/../.." ${CMAKE_CURRENT_BINARY_DIR}/pal)
