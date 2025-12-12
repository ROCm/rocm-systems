# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
#
# SPDX-License-Identifier: MIT

if(ROCCLR_FOUND)
  return()
endif()

find_path(ROCCLR_INCLUDE_DIR top.hpp
  HINTS
    ${ROCCLR_PATH}
  PATHS
    # gerrit repo name
    ${CMAKE_SOURCE_DIR}/vdi
    ${CMAKE_SOURCE_DIR}/../vdi
    ${CMAKE_SOURCE_DIR}/../../vdi
    # github repo name
    ${CMAKE_SOURCE_DIR}/ROCclr
    ${CMAKE_SOURCE_DIR}/../ROCclr
    ${CMAKE_SOURCE_DIR}/../../ROCclr
    # jenkins repo name
    ${CMAKE_SOURCE_DIR}/rocclr
    ${CMAKE_SOURCE_DIR}/../rocclr
    ${CMAKE_SOURCE_DIR}/../../rocclr
  PATH_SUFFIXES
    include)

include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(ROCclr
  "\nROCclr not found"
  ROCCLR_INCLUDE_DIR)
mark_as_advanced(ROCCLR_INCLUDE_DIR)

list(APPEND CMAKE_MODULE_PATH "${ROCCLR_INCLUDE_DIR}/../cmake")
include(ROCclr)
