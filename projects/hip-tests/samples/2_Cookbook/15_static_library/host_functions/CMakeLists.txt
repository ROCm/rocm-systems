# Copyright (c) 2025 Advanced Micro Devices, Inc. All Rights Reserved.
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

project(static_lib LANGUAGES CXX HIP)

# For windows, AR is MS Librarian and that is picked by Visual Studio's command prompt.
# If using LLVM/Clang toolchain, use llvm-ar instead.
if(WIN32)
  # Get the directory of the compiler
  get_filename_component(LLVM_BIN_DIR "${CMAKE_CXX_COMPILER}" DIRECTORY)
  
  # First try to find llvm-ar (for LLVM/Clang toolchain)
  # LLVM_BIN_DIR is derived from CMAKE_CXX_COMPILER path
  find_program(llvm_ar NAMES llvm-ar llvm-ar.exe
    HINTS
      "${LLVM_BIN_DIR}"
      "$ENV{HIP_PATH}/bin"
      "$ENV{ROCM_PATH}/llvm/bin"
  )
  if(llvm_ar)
    set(CMAKE_AR ${llvm_ar})
  else()
    # Fall back to MSVC lib.exe
    find_program(libpath NAMES lib.exe)
    set(CMAKE_AR ${libpath})
  endif()
  
  # Find lld-link for linking
  # LLVM_BIN_DIR is derived from CMAKE_CXX_COMPILER path
  find_program(lld_link NAMES lld-link lld-link.exe
    HINTS
      "${LLVM_BIN_DIR}"
      "$ENV{HIP_PATH}/bin"
      "$ENV{ROCM_PATH}/llvm/bin"
  )
  if(lld_link)
    set(CMAKE_LINKER ${lld_link})
    # Use --ld-path= to specify full path to linker (works better than -fuse-ld= with full paths)
    set(CMAKE_HIP_USING_LINKER_DEFAULT "--ld-path=${lld_link}")
    set(CMAKE_HIP_USING_LINKER_LLD "--ld-path=${lld_link}")
    set(CMAKE_CXX_USING_LINKER_DEFAULT "--ld-path=${lld_link}")
    set(CMAKE_CXX_USING_LINKER_LLD "--ld-path=${lld_link}")
  endif()
endif()

set(CMAKE_BUILD_TYPE Release)

# Turn static library generation ON
option(BUILD_SHARED_LIBS "Build as a shared library" OFF)

set(CPP_SOURCES ${CMAKE_CURRENT_SOURCE_DIR}/hipOptLibrary.cpp)

# For windows, We need to tell cmake how to create static library.
if(WIN32)
  if(llvm_ar)
    # LLVM ar uses different syntax
    set(CMAKE_CXX_CREATE_STATIC_LIBRARY "<CMAKE_AR> qc <TARGET> <OBJECTS>")
    set(CMAKE_HIP_CREATE_STATIC_LIBRARY "<CMAKE_AR> qc <TARGET> <OBJECTS>")
  else()
    set(CMAKE_CXX_CREATE_STATIC_LIBRARY "<CMAKE_AR> /out:<TARGET> <LINK_FLAGS> <OBJECTS>")
    set(CMAKE_HIP_CREATE_STATIC_LIBRARY "<CMAKE_AR> /out:<TARGET> <LINK_FLAGS> <OBJECTS>")
  endif()
endif()

if(TARGET build_cookbook)
  set(EXCLUDE_OPTION EXCLUDE_FROM_ALL)
else()
  set(EXCLUDE_OPTION)
endif()

set(TEST_SOURCES ${CMAKE_CURRENT_SOURCE_DIR}/hipMain1.cpp)

# Mark source files as HIP code
set_source_files_properties(hipOptLibrary.cpp PROPERTIES LANGUAGE HIP)
set_source_files_properties(hipMain1.cpp PROPERTIES LANGUAGE HIP)

# Generate static lib libHipOptLibrary.a
add_library(HipOptLibrary ${EXCLUDE_OPTION} STATIC ${CPP_SOURCES})

# Set-up the correct flags to generate the static library
target_link_options(HipOptLibrary PRIVATE --emit-static-lib)

# Include directory for HSA headers
if(WIN32)
  # On Windows, use path relative to LLVM_BIN_DIR or skip if not needed
  # HSA headers location varies; add if available
else()
  target_include_directories(HipOptLibrary PRIVATE ${ROCM_PATH}/hsa/include)
endif()

# Create test executable that uses libHipOptLibrary.a
add_executable(test_opt_static ${EXCLUDE_OPTION} ${TEST_SOURCES})
add_dependencies(test_opt_static HipOptLibrary)

# Get the ROCm lib directory (three levels up from llvm/bin, then into lib)
if(WIN32)
  get_filename_component(ROCM_LIB_DIR "${LLVM_BIN_DIR}/../../.." ABSOLUTE)
  set(ROCM_LIB_DIR "${ROCM_LIB_DIR}/lib")
endif()

# Link the static library
if(WIN32)
  # Add library search paths for ROCm libraries
  target_link_directories(test_opt_static PRIVATE 
    ${CMAKE_CURRENT_BINARY_DIR}
    "${ROCM_LIB_DIR}"
  )
  target_link_libraries(test_opt_static PRIVATE HipOptLibrary)
else()
  target_link_libraries(test_opt_static PRIVATE -lHipOptLibrary -L${CMAKE_CURRENT_BINARY_DIR})
endif()

# Link ROCm runtime libraries
if(WIN32)
  # Find the actual library files (amd_comgr may have version suffix like amd_comgr0702.lib)
  file(GLOB AMD_COMGR_LIB "${ROCM_LIB_DIR}/amd_comgr*.lib")
  if(NOT AMD_COMGR_LIB)
    message(WARNING "amd_comgr library not found in ${ROCM_LIB_DIR}")
  endif()
  target_link_libraries(test_opt_static PRIVATE 
    "${ROCM_LIB_DIR}/amdhip64.lib"
    "${AMD_COMGR_LIB}"
  )
else()
  target_link_libraries(test_opt_static PRIVATE amdhip64 amd_comgr hsa-runtime64)
endif()

# Set RPATH so executable can find HIP libraries at runtime
if(UNIX)
  set_target_properties(test_opt_static PROPERTIES
    BUILD_RPATH "${ROCM_PATH}/lib"
    INSTALL_RPATH "${ROCM_PATH}/lib"
  )
endif()

if(TARGET build_cookbook)
add_dependencies(build_cookbook HipOptLibrary test_opt_static)
endif()
