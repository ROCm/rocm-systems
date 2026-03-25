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

# Prefer the active package graph (for example, a super-project provided
# package) and refuse to build against the amd_comgr stub package, which
# injects headers from a potentially unrelated ROCm install.
if(DEFINED amd_comgr_DIR)
  get_filename_component(_amd_comgr_dir_name "${amd_comgr_DIR}" NAME)
  if(_amd_comgr_dir_name STREQUAL "amd_comgr_stub")
    unset(amd_comgr_DIR CACHE)
    unset(amd_comgr_DIR)
  endif()
endif()

if(THEROCK_USE_SAFE_DEPENDENCY_PROVIDER AND DEFINED THEROCK_PACKAGE_DIR_amd_comgr)
  set(amd_comgr_DIR "${THEROCK_PACKAGE_DIR_amd_comgr}" CACHE PATH
    "TheRock-provided amd_comgr package directory" FORCE)
endif()

if(THEROCK_USE_SAFE_DEPENDENCY_PROVIDER AND DEFINED THEROCK_PACKAGE_DIR_amd_comgr)
  include("${THEROCK_PACKAGE_DIR_amd_comgr}/amd_comgr-config.cmake")
  set(amd_comgr_FOUND TRUE)
elseif(THEROCK_USE_SAFE_DEPENDENCY_PROVIDER)
  find_package(amd_comgr 2.9 CONFIG QUIET)
else()
  find_package(amd_comgr 2.9 CONFIG QUIET
    PATHS
      /opt/rocm/
      ${ROCM_INSTALL_PATH}
    PATH_SUFFIXES
      cmake/amd_comgr
      lib/cmake/amd_comgr)
endif()

if (NOT amd_comgr_FOUND)
  if(THEROCK_USE_SAFE_DEPENDENCY_PROVIDER)
    find_package(amd_comgr 3.0 REQUIRED CONFIG)
  else()
    find_package(amd_comgr 3.0 REQUIRED CONFIG
      PATHS
        /opt/rocm/
        ${ROCM_INSTALL_PATH}
      PATH_SUFFIXES
        cmake/amd_comgr
        lib/cmake/amd_comgr)
  endif()
endif()

set(_amd_comgr_is_stub FALSE)
set(_amd_comgr_imported_location)
get_target_property(_amd_comgr_imported_configs amd_comgr IMPORTED_CONFIGURATIONS)
if(_amd_comgr_imported_configs)
  foreach(_amd_comgr_imported_config IN LISTS _amd_comgr_imported_configs)
    get_target_property(_amd_comgr_candidate_location amd_comgr
      "IMPORTED_LOCATION_${_amd_comgr_imported_config}")
    if(_amd_comgr_candidate_location)
      set(_amd_comgr_imported_location "${_amd_comgr_candidate_location}")
      if(_amd_comgr_candidate_location MATCHES "amd_comgr_stub")
        set(_amd_comgr_is_stub TRUE)
      endif()
      break()
    endif()
  endforeach()
endif()
if(NOT _amd_comgr_imported_location)
  get_target_property(_amd_comgr_imported_location amd_comgr IMPORTED_LOCATION)
  if(_amd_comgr_imported_location MATCHES "amd_comgr_stub")
    set(_amd_comgr_is_stub TRUE)
  endif()
endif()
if(NOT _amd_comgr_is_stub)
  get_target_property(_amd_comgr_link_libraries amd_comgr INTERFACE_LINK_LIBRARIES)
  if(_amd_comgr_link_libraries MATCHES "(^|;)\\$<LINK_ONLY:amd_comgr_loader>(;|$)"
      OR _amd_comgr_link_libraries MATCHES "(^|;)amd_comgr_loader(;|$)")
    set(_amd_comgr_is_stub TRUE)
  endif()
endif()
if(_amd_comgr_is_stub)
  message(FATAL_ERROR
    "rocclr requires the real amd_comgr package, but CMake resolved the "
    "amd_comgr stub package at '${_amd_comgr_imported_location}'. "
    "Configure with a real "
    "amd_comgr package on CMAKE_PREFIX_PATH or amd_comgr_DIR.")
endif()

get_target_property(_amd_comgr_lib_type amd_comgr TYPE)
target_compile_definitions(rocclr PUBLIC)
if(_amd_comgr_lib_type STREQUAL "SHARED_LIBRARY")
  target_compile_definitions(rocclr PUBLIC COMGR_DYN_DLL)
endif()
target_link_libraries(rocclr PUBLIC amd_comgr)

if(CLR_BUILD_HIP)
  # Temporary hack for versioned comgr needed by hiprtc
  file(STRINGS ${HIP_COMMON_DIR}/VERSION VERSION_LIST REGEX "^[0-9]+")
  list(GET VERSION_LIST 0 HIP_VERSION_MAJOR)
  list(GET VERSION_LIST 1 HIP_VERSION_MINOR)

  add_definitions(-DHIP_MAJOR_VERSION=${HIP_VERSION_MAJOR})
  add_definitions(-DHIP_MINOR_VERSION=${HIP_VERSION_MINOR})
endif()
