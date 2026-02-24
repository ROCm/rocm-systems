################################################################################
##
## The University of Illinois/NCSA
## Open Source License (NCSA)
##
## Copyright (c) 2018-2026, Advanced Micro Devices, Inc. All rights reserved.
##
## Permission is hereby granted, free of charge, to any person obtaining a copy
## of this software and associated documentation files (the "Software"), to
## deal with the Software without restriction, including without limitation
## the rights to use, copy, modify, merge, publish, distribute, sublicense,
## and/or sell copies of the Software, and to permit persons to whom the
## Software is furnished to do so, subject to the following conditions:
##
##  - Redistributions of source code must retain the above copyright notice,
##    this list of conditions and the following disclaimers.
##  - Redistributions in binary form must reproduce the above copyright
##    notice, this list of conditions and the following disclaimers in
##    the documentation and/or other materials provided with the distribution.
##  - Neither the names of Advanced Micro Devices, Inc,
##    nor the names of its contributors may be used to endorse or promote
##    products derived from this Software without specific prior written
##    permission.
##
## THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
## IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
## FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
## THE CONTRIBUTORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR
## OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE,
## ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
## DEALINGS WITH THE SOFTWARE.
##
################################################################################

# Available defines to be used when building rocr-debug-agent:
#
# ENABLE_TESTS
#
# Enable building the rocr-debug-agent tests. Default is ON.
#
# The debug agent has a small set of validation tests that can be built
# with a suitable HIP compiler. This is mostly for development, so it isn't
# required to be built if users are only looking for the tool.
#
#
# rocm-debug-agent has the following dependencies:
#
# amd_dbgapi
# LibElf
# LibDW
# hsa-runtime64
# GIT (optional)
# Threads

cmake_minimum_required(VERSION 3.10)

project(rocm-debug-agent VERSION 2.1.0)

include(GNUInstallDirs)

# By default build the Debug Agent tests.
option(ENABLE_TESTS "Enable building Debug Agent tests" ON)

file(GLOB SOURCES "src/*.cpp")
add_library(rocm-debug-agent SHARED ${SOURCES})

set_target_properties(rocm-debug-agent PROPERTIES
  CXX_STANDARD 17
  CXX_STANDARD_REQUIRED ON
  CXX_EXTENSIONS OFF
  VERSION ${PROJECT_VERSION}
  SOVERSION ${PROJECT_VERSION_MAJOR}
  NO_SYSTEM_FROM_IMPORTED ON)

# Unified dependency discovery for both TheRock and standalone builds.
# Try CONFIG mode first (TheRock), then fall back to manual discovery (standalone).

# Find libdw (elfutils)
find_package(libdw CONFIG QUIET)
if(NOT libdw_FOUND)
  find_path(LIBDW_INCLUDE_DIR NAMES elfutils/libdw.h PATHS /usr/include /usr/local/include)
  find_library(LIBDW_LIBRARY NAMES dw PATHS /usr/lib /usr/local/lib)
  if(NOT LIBDW_INCLUDE_DIR OR NOT LIBDW_LIBRARY)
    message(FATAL_ERROR "Could not find libdw")
  endif()
  add_library(libdw::libdw UNKNOWN IMPORTED)
  set_target_properties(libdw::libdw PROPERTIES
    IMPORTED_LOCATION "${LIBDW_LIBRARY}"
    INTERFACE_INCLUDE_DIRECTORIES "${LIBDW_INCLUDE_DIR}")
  message(STATUS "Found libdw: ${LIBDW_LIBRARY}")
endif()

# Find hsa-runtime64 (ROCm Runtime)
find_package(hsa-runtime64 CONFIG QUIET)
if(NOT hsa-runtime64_FOUND)
  find_path(HSA_INCLUDE_DIR NAMES hsa/hsa.h PATHS /opt/rocm/include /opt/rocm/hsa/include)
  find_library(HSA_LIBRARY NAMES hsa-runtime64 PATHS /opt/rocm/lib /opt/rocm/hsa/lib)
  if(NOT HSA_INCLUDE_DIR OR NOT HSA_LIBRARY)
    message(FATAL_ERROR "Could not find hsa-runtime64")
  endif()
  add_library(hsa-runtime64::hsa-runtime64 UNKNOWN IMPORTED)
  set_target_properties(hsa-runtime64::hsa-runtime64 PROPERTIES
    IMPORTED_LOCATION "${HSA_LIBRARY}"
    INTERFACE_INCLUDE_DIRECTORIES "${HSA_INCLUDE_DIR}")
  message(STATUS "Found hsa-runtime64: ${HSA_LIBRARY}")
endif()

# Find libelf
find_package(LibElf CONFIG QUIET)
if(NOT LibElf_FOUND)
  find_path(LIBELF_INCLUDE_DIR NAMES libelf.h PATHS /usr/include /usr/local/include PATH_SUFFIXES libelf)
  find_library(LIBELF_LIBRARY NAMES elf PATHS /usr/lib /usr/local/lib)
  if(NOT LIBELF_INCLUDE_DIR OR NOT LIBELF_LIBRARY)
    message(FATAL_ERROR "Could not find libelf")
  endif()
  add_library(elf::elf UNKNOWN IMPORTED)
  set_target_properties(elf::elf PROPERTIES
    IMPORTED_LOCATION "${LIBELF_LIBRARY}"
    INTERFACE_INCLUDE_DIRECTORIES "${LIBELF_INCLUDE_DIR}")
  message(STATUS "Found libelf: ${LIBELF_LIBRARY}")
endif()

find_package(amd-dbgapi REQUIRED CONFIG
  PATHS
    /opt/rocm/
  PATH_SUFFIXES
    cmake/amd-dbgapi
    lib/cmake/amd-dbgapi
)

find_package(Threads REQUIRED)

target_link_libraries(rocm-debug-agent
  PRIVATE amd-dbgapi hsa-runtime64::hsa-runtime64 elf::elf libdw::libdw
  Threads::Threads ${CMAKE_DL_LIBS}
  -Wl,--version-script=${CMAKE_CURRENT_SOURCE_DIR}/src/exportmap -Wl,--no-undefined)
target_compile_options(rocm-debug-agent PRIVATE -Werror -Wall)

if(DEFINED ENV{ROCM_BUILD_ID})
  # ROCM_BUILD_ID is set by the ROCm-CI build environment.
  set(build_info $ENV{ROCM_BUILD_ID})
else()
  string(TIMESTAMP NOW "%Y%m%dT%H%M%S")
  set(build_info developer-build-${NOW})

  if(DEFINED ENV{USER})
    set(build_info ${build_info}-$ENV{USER})
  endif()

  find_package(Git)
  if(GIT_FOUND)
    execute_process(
      COMMAND ${GIT_EXECUTABLE} rev-parse --short HEAD
      WORKING_DIRECTORY "${PROJECT_SOURCE_DIR}"
      OUTPUT_VARIABLE build_revision
      ERROR_QUIET
      OUTPUT_STRIP_TRAILING_WHITESPACE
    )
  else()
    message(STATUS "GIT not found")
  endif()

  if(DEFINED build_revision)
    set(build_info ${build_info}-git-${build_revision})
  endif()
endif()

# Check for memfd_create headers and libs
include(CheckCSourceCompiles)
check_c_source_compiles("
  #define _GNU_SOURCE
  #include <sys/mman.h>
  int main() { return memfd_create (\"cmake_test\", 0); }
  " HAVE_MEMFD_CREATE)
if (HAVE_MEMFD_CREATE)
  add_definitions(-DHAVE_MEMFD_CREATE)
endif()


target_compile_options(rocm-debug-agent
  PRIVATE -fno-rtti -Wall -Wno-attributes -fvisibility=hidden)

target_compile_definitions(rocm-debug-agent
  PRIVATE AMD_INTERNAL_BUILD _GNU_SOURCE __STDC_LIMIT_MACROS __STDC_CONSTANT_MACROS)

install(TARGETS rocm-debug-agent
  LIBRARY
    DESTINATION ${CMAKE_INSTALL_LIBDIR}
  COMPONENT runtime)

install(FILES LICENSE.txt README.md
  DESTINATION ${CMAKE_INSTALL_DOCDIR}
  COMPONENT runtime)

if(ENABLE_TESTS)
  enable_testing()
  add_subdirectory(test)
endif()
