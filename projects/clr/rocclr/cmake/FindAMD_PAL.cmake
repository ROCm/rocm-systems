# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
#
# SPDX-License-Identifier: MIT

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
    ${CMAKE_SOURCE_DIR}/../../../drivers/drivers
  PATH_SUFFIXES
    inc/asic_reg)
if (NOT AMD_COMPUTE_WIN)
find_path(AMD_HSAIL_INCLUDE_DIR hsa.h
  HINTS
    ${AMD_SC_PATH}
  PATHS
    ${CMAKE_SOURCE_DIR}/sc
    ${CMAKE_SOURCE_DIR}/../sc
    ${CMAKE_SOURCE_DIR}/../../sc
  PATH_SUFFIXES
    HSAIL/include)
endif()

find_path(AMD_PAL_INCLUDE_DIR pal.h
  HINTS
    ${AMD_PAL_PATH}
  PATHS
    ${CMAKE_SOURCE_DIR}/pal
    ${CMAKE_SOURCE_DIR}/../pal
    ${CMAKE_SOURCE_DIR}/../../pal
    ${CMAKE_SOURCE_DIR}/../../../pal
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
