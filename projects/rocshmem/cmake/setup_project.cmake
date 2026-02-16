###############################################################################
# Copyright (c) Advanced Micro Devices, Inc. All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to
# deal in the Software without restriction, including without limitation the
# rights to use, copy, modify, merge, publish, distribute, sublicense, and/or
# sell copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
# FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS
# IN THE SOFTWARE.
###############################################################################

###############################################################################
# DEFAULT BUILD TYPE
###############################################################################
set(CMAKE_BUILD_TYPE "Release" CACHE STRING
      "build type: Release, Debug, RelWithDebInfo, MinSizeRel")
set_property(CACHE CMAKE_BUILD_TYPE PROPERTY STRINGS Release RelWithDebInfo Debug MinSizeRel)

###############################################################################
# DEPENDENCIES
###############################################################################

# Try to establish ROCM_PATH (for find_package)
#==================================================================================================
if (NOT DEFINED CACHE{ROCM_MAJOR_VERSION})
  find_file(rocm_version_file "version" PATH_SUFFIXES ".info"
    HINTS ${ROCM_PATH} ENV ROCM_PATH ${ROCM_ROOT} ENV ROCM_ROOT ${hip_ROOT} ENV hip_ROOT ${HIP_ROOT} ENV HIP_ROOT
    PATHS /opt/rocm
    REQUIRED)
  cmake_path(GET rocm_version_file PARENT_PATH version_file_dir)
  cmake_path(GET version_file_dir PARENT_PATH rocm_path)

  ## Check for ROCm version
  set(EXPLICIT_ROCM_VERSION "" CACHE STRING "Explicit ROCM version to compile to (auto detect if empty)")
  if(EXPLICIT_ROCM_VERSION)
    set(rocm_version_string "${EXPLICIT_ROCM_VERSION}")
  else()
    file(READ ${rocm_version_file} rocm_version_string)
  endif()
  string(REGEX MATCH "([0-9]+)\\.([0-9]+)\\.([0-9]+)" rocm_version_matches ${rocm_version_string})
  if (rocm_version_matches)
    set(ROCM_MAJOR_VERSION ${CMAKE_MATCH_1} CACHE INTERNAL "")
    set(ROCM_MINOR_VERSION ${CMAKE_MATCH_2} CACHE INTERNAL "")
    set(ROCM_PATCH_VERSION ${CMAKE_MATCH_3} CACHE INTERNAL "")
  else()
    message(FATAL_ERROR "Could not determine ROCm version (set EXPLICIT_ROCM_VERSION or set ROCM_PATH to a valid installation)")
  endif()

  set(ROCM_PATH "${rocm_path}" CACHE PATH "Location of the ROCm SDK")
  message(STATUS "ROCm version: ${ROCM_MAJOR_VERSION}.${ROCM_MINOR_VERSION}.${ROCM_PATCH_VERSION} from ${ROCM_PATH}")
endif()

# Always prefer the ROCm we found
list(PREPEND CMAKE_PREFIX_PATH ${ROCM_PATH})

# Use hipcc from our rocm install
if (NOT DEFINED CMAKE_CXX_COMPILER)
  find_program(CMAKE_CXX_COMPILER hipcc PATHS /opt/rocm)
endif()

###############################################################################
# GLOBAL COMPILE FLAGS
###############################################################################
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_FLAGS_DEBUG "-Og -ggdb")

list(APPEND CMAKE_MODULE_PATH ${CMAKE_CURRENT_SOURCE_DIR}/cmake)
