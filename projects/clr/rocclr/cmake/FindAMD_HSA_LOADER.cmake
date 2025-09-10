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
  NO_DEFAULT_PATH)

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

set(USE_AMD_LIBELF "yes" CACHE FORCE "")
# TODO compiler team requested supporting sp3 disassembly
set(NO_SI_SP3 "yes" CACHE FORCE "")
set(HSAIL_COMPILER_SOURCE_DIR "${AMD_LIBELF_INCLUDE_DIR}/../../../../..")
set(HSAIL_ELFTOOLCHAIN_DIR ${HSAIL_COMPILER_SOURCE_DIR}/lib/loaders/elf/utils)
add_subdirectory("${AMD_LIBELF_INCLUDE_DIR}" ${CMAKE_CURRENT_BINARY_DIR}/libelf)
add_subdirectory("${AMD_HSAIL_INCLUDE_DIR}/../ext/libamdhsacode" ${CMAKE_CURRENT_BINARY_DIR}/libamdhsacode)
add_subdirectory("${AMD_HSAIL_INCLUDE_DIR}/../ext/loader" ${CMAKE_CURRENT_BINARY_DIR}/loader)
