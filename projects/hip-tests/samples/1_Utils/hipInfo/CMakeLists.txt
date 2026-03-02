# Copyright (c) 2020 - 2023 Advanced Micro Devices, Inc. All Rights Reserved.
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

cmake_minimum_required(VERSION 3.21)

if(UNIX)
  if(NOT DEFINED ROCM_PATH)
    if(DEFINED ENV{ROCM_PATH})
      set(ROCM_PATH $ENV{ROCM_PATH} CACHE STRING "ROCM Path")
    else()
      set(ROCM_PATH "/opt/rocm" CACHE STRING "Default ROCM installation directory.")
    endif()
  endif()
  # Search for rocm in common locations
  list(APPEND CMAKE_PREFIX_PATH ${ROCM_PATH})
endif()

# flag is set to ON in compute build for windows
option(HIPINFO_INTERNAL_BUILD "Enable building hipInfo from compute" OFF)

# Windows internal build: Configure HIP compiler before project() to bypass CMake's auto-detection
# HIPINFO_INTERNAL_BUILD is set by amd_build.py when building as part of compute project
if(WIN32 AND HIPINFO_INTERNAL_BUILD)
  # Set HIP compiler to use same as CXX compiler (set by amd_build.py)
  if(CMAKE_CXX_COMPILER)
    set(CMAKE_HIP_COMPILER "${CMAKE_CXX_COMPILER}" CACHE FILEPATH "HIP compiler")
  endif()
  # Disable CMake's HIP architecture auto-detection (which fails on Windows)
  set(CMAKE_HIP_ARCHITECTURES OFF CACHE STRING "Disable HIP architecture detection")
  # Skip CMake's HIP compiler testing phase
  set(CMAKE_HIP_COMPILER_FORCED TRUE CACHE BOOL "Skip HIP compiler test")
endif()

project(hipInfo LANGUAGES CXX HIP)

set(CMAKE_BUILD_TYPE Release)

# Create the excutable
if(TARGET build_utils)
  set(EXCLUDE_OPTION EXCLUDE_FROM_ALL)
else()
  set(EXCLUDE_OPTION)
endif()

# Mark source file as HIP code
set_source_files_properties(hipInfo.cpp PROPERTIES LANGUAGE HIP)

add_executable(hipInfo ${EXCLUDE_OPTION} hipInfo.cpp)

target_include_directories(hipInfo PRIVATE ../../common)

# Windows: Add HIP compile options since CMAKE_HIP_ARCHITECTURES=OFF disables automatic flag handling
if(WIN32 AND HIPINFO_INTERNAL_BUILD)
  target_compile_options(hipInfo PRIVATE $<$<COMPILE_LANGUAGE:HIP>:-x hip --hip-path=${CMAKE_PREFIX_PATH}>)
  target_link_directories(hipInfo PRIVATE ${CMAKE_PREFIX_PATH}/lib)
  target_link_libraries(hipInfo PRIVATE amdhip64)
endif()

# Set RPATH so executable can find HIP libraries at runtime
if(UNIX)
  set_target_properties(hipInfo PROPERTIES
    BUILD_RPATH "${ROCM_PATH}/lib"
    INSTALL_RPATH "${ROCM_PATH}/lib"
  )
endif()

# Used only when make install is called
# when hipInfo is built as part of compute project
# hipInfo.exe will be installed to install/hip/bin path
if (WIN32 AND HIPINFO_INTERNAL_BUILD)
  install(FILES ${PROJECT_BINARY_DIR}/hipInfo.exe DESTINATION ${CMAKE_INSTALL_PREFIX})
endif()

if(TARGET build_utils)
  add_dependencies(build_utils hipInfo)
endif()
