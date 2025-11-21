# - Try to find libelf
# Once done this will define
#
#  LIBELF_FOUND - system has libelf
#  LIBELF_INCLUDE_DIRS - the libelf include directory
#  LIBELF_LIBRARIES - Link these to use libelf
#  LIBELF_DEFINITIONS - Compiler switches required for using libelf
#
#  Copyright (c) 2008 Bernhard Walle <bernhard.walle@gmx.de>
#
#  Redistribution and use is allowed according to the terms of the New
#  BSD license.
#  For details see the accompanying COPYING-CMAKE-SCRIPTS file.
#

if (LIBELF_FOUND)
  return()
endif (LIBELF_FOUND)

if (UNIX)
  find_path (LIBELF_INCLUDE_DIRS
    NAMES
      libelf.h
    PATHS
      /usr/include
      /usr/include/libelf
      /usr/local/include
      /usr/local/include/libelf
      /opt/local/include
      /opt/local/include/libelf
      ENV CPATH)
  
  find_library (LIBELF_LIBRARIES
    NAMES
      elf
    PATHS
      /usr/lib
      /usr/lib64
      /usr/local/lib
      /usr/local/lib64
      /opt/local/lib
      /opt/local/lib64
      ENV LIBRARY_PATH
      ENV LD_LIBRARY_PATH)
  
  include (FindPackageHandleStandardArgs)
  
  
  # handle the QUIETLY and REQUIRED arguments and set LIBELF_FOUND to TRUE if all listed variables are TRUE
  FIND_PACKAGE_HANDLE_STANDARD_ARGS(LibElf DEFAULT_MSG
    LIBELF_LIBRARIES
    LIBELF_INCLUDE_DIRS)
  
  SET(CMAKE_REQUIRED_LIBRARIES elf)
  if (CMAKE_CXX_COMPILER_LOADED)
    INCLUDE(CheckCXXSourceCompiles)
    CHECK_CXX_SOURCE_COMPILES("#include <libelf.h>
    int main() {
      Elf *e = (Elf*)0;
      size_t sz;
      elf_getshdrstrndx(e, &sz);
      return 0;
    }" ELF_GETSHDRSTRNDX)
  else()
  set ( ELF_GETSHDRSTRNDX "TRUE" )
  endif(CMAKE_CXX_COMPILER_LOADED)
  
  mark_as_advanced(LIBELF_INCLUDE_DIRS LIBELF_LIBRARIES ELF_GETSHDRSTRNDX)
  
  if(LIBELF_FOUND)
    add_library(elf::elf UNKNOWN IMPORTED)
    set_property(TARGET elf::elf PROPERTY IMPORTED_LOCATION ${LIBELF_LIBRARIES})
    set_property(TARGET elf::elf PROPERTY INTERFACE_INCLUDE_DIRECTORIES ${LIBELF_INCLUDE_DIRS})
  endif()
else()
  find_path(ROCR_LIBELF_INCLUDE_DIR libelf.h
      HINTS
        ${AMD_LIBELF_PATH}
      PATHS
        ${CMAKE_SOURCE_DIR}/hsail-compiler/lib/loaders/elf/utils/libelf
        ${CMAKE_SOURCE_DIR}/../hsail-compiler/lib/loaders/elf/utils/libelf
        ${CMAKE_SOURCE_DIR}/../../hsail-compiler/lib/loaders/elf/utils/libelf
      NO_DEFAULT_PATH)

  message("=> LibElf paths:" ${CMAKE_CURRENT_BINARY_DIR} ${ROCR_LIBELF_INCLUDE_DIR})
  if (${BUILD_SHARED_LIBS})
    mark_as_advanced(ROCR_LIBELF_INCLUDE_DIR)
    add_subdirectory("${ROCR_LIBELF_INCLUDE_DIR}" ${CMAKE_CURRENT_BINARY_DIR}/libelf)
  endif()
  set(USE_AMD_LIBELF "yes" CACHE FORCE "")
  set(AMD_ELFTOOLCHAIN_DIR ${ROCR_LIBELF_INCLUDE_DIR}/../..;${ROCR_LIBELF_INCLUDE_DIR}/../common/win32;${ROCR_LIBELF_INCLUDE_DIR}/../common)
  set(ROCR_LIBELF_INCLUDE_DIR ${ROCR_LIBELF_INCLUDE_DIR};${AMD_ELFTOOLCHAIN_DIR}) 
  set(LIBELF_INCLUDE_DIR ${ROCR_LIBELF_INCLUDE_DIR}) 
endif()
