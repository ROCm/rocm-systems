# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
#
# SPDX-License-Identifier: MIT

if(AMD_PAL_LIB_FOUND)
  return()
endif()

find_path(AMD_PAL_LIB_INCLUDE_DIR pal.h
  HINTS
    ${AMD_COMPUTE_WIN}/pal
  PATHS
    ${CMAKE_SOURCE_DIR}/pal
    ${CMAKE_SOURCE_DIR}/../pal
    ${CMAKE_SOURCE_DIR}/../../pal
    ${CMAKE_SOURCE_DIR}/../../../pal
  PATH_SUFFIXES
    inc/core)

include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(AMD_PAL_LIB
  "\nPAL not found"
  AMD_PAL_LIB_INCLUDE_DIR)
mark_as_advanced(AMD_PAL_LIB_INCLUDE_DIR)

# SYSTEM: these are vendored third-party sources under
# shared/amdgpu-windows-interop. CLR cannot fix warnings raised inside their
# headers, so their interface include directories are handed to the compiler as
# system includes (-external:I under MSVC, -isystem elsewhere) and silenced by
# the -external:W0 already on the command line.
add_subdirectory("${AMD_PAL_LIB_INCLUDE_DIR}/../.." ${CMAKE_CURRENT_BINARY_DIR}/pal SYSTEM)
