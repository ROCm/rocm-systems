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

find_package(amd_comgr 3.0 REQUIRED CONFIG)

get_filename_component(_amd_comgr_dir_name "${amd_comgr_DIR}" NAME)
if(_amd_comgr_dir_name STREQUAL "amd_comgr_stub")
  message(FATAL_ERROR
    "rocclr requires the real amd_comgr package, but CMake resolved the "
    "amd_comgr stub package at '${amd_comgr_DIR}'. Configure with a real "
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
