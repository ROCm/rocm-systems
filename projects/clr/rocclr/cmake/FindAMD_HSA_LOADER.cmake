# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
#
# SPDX-License-Identifier: MIT

if(AMD_HSA_LOADER_FOUND)
  return()
endif()

find_path(AMD_LIBELF_INCLUDE_DIR libelf.h
  HINTS
    ${AMD_LIBELF_PATH}
  PATHS
    ${CMAKE_SOURCE_DIR}/hsail-compiler/lib/loaders/elf/utils/libelf
    ${CMAKE_SOURCE_DIR}/../hsail-compiler/lib/loaders/elf/utils/libelf
    ${CMAKE_SOURCE_DIR}/../../hsail-compiler/lib/loaders/elf/utils/libelf
    ${CMAKE_SOURCE_DIR}/../../shared/amdgpu-windows-interop/hsail-compiler/lib/loaders/elf/utils/libelf
  NO_DEFAULT_PATH)
if (NOT AMD_COMPUTE_WIN OR ROCCLR_ENABLE_PAL)
find_path(AMD_HSAIL_INCLUDE_DIR hsa.h
  HINTS
    ${AMD_SC_PATH}
  PATHS
    ${CMAKE_SOURCE_DIR}/sc
    ${CMAKE_SOURCE_DIR}/../sc
    ${CMAKE_SOURCE_DIR}/../../sc
  PATH_SUFFIXES
    HSAIL/include)

include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(AMD_HSA_LOADER
    "\nHSA Loader not found"
    AMD_LIBELF_INCLUDE_DIR AMD_HSAIL_INCLUDE_DIR)
mark_as_advanced(AMD_LIBELF_INCLUDE_DIR AMD_HSAIL_INCLUDE_DIR)
endif()
set(USE_AMD_LIBELF "yes" CACHE FORCE "")
# TODO compiler team requested supporting sp3 disassembly
set(NO_SI_SP3 "yes" CACHE FORCE "")
set(HSAIL_COMPILER_SOURCE_DIR "${AMD_LIBELF_INCLUDE_DIR}/../../../../..")
set(HSAIL_ELFTOOLCHAIN_DIR ${HSAIL_COMPILER_SOURCE_DIR}/lib/loaders/elf/utils)
add_subdirectory("${AMD_LIBELF_INCLUDE_DIR}" ${CMAKE_CURRENT_BINARY_DIR}/libelf)
if (NOT AMD_COMPUTE_WIN OR ROCCLR_ENABLE_PAL)
add_subdirectory("${AMD_HSAIL_INCLUDE_DIR}/../ext/libamdhsacode" ${CMAKE_CURRENT_BINARY_DIR}/libamdhsacode)
add_subdirectory("${AMD_HSAIL_INCLUDE_DIR}/../ext/loader" ${CMAKE_CURRENT_BINARY_DIR}/loader)
endif()