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
if(WIN32)
  find_program(libpath NAMES lib.exe)
  set(CMAKE_AR ${libpath})
endif()

set(CMAKE_BUILD_TYPE Release)

# Turn static library generation ON
option(BUILD_SHARED_LIBS "Build as a shared library" OFF)

set(CPP_SOURCES ${CMAKE_CURRENT_SOURCE_DIR}/hipOptLibrary.cpp)

# For windows, We need to tell cmake how to create static library.
if(WIN32)
  set(CMAKE_CXX_CREATE_STATIC_LIBRARY "<CMAKE_AR> /out:<TARGET> <LINK_FLAGS> <OBJECTS>")
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
target_include_directories(HipOptLibrary PRIVATE /opt/rocm/hsa/include)

# Create test executable that uses libHipOptLibrary.a
add_executable(test_opt_static ${EXCLUDE_OPTION} ${TEST_SOURCES})
add_dependencies(test_opt_static HipOptLibrary)
target_link_libraries(test_opt_static PRIVATE -lHipOptLibrary -L${CMAKE_CURRENT_BINARY_DIR})

if(WIN32)
  target_link_libraries(test_opt_static PRIVATE amdhip64 amd_comgr)
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
