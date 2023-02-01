################################################################################
# Copyright (c) 2018 Advanced Micro Devices, Inc. All rights reserved.
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
################################################################################

## Linux Compiler options
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fms-extensions")

add_definitions ( -DNEW_TRACE_API=1 )

## CLANG options
if("$ENV{CXX}" STREQUAL "/usr/bin/clang++")
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -ferror-limit=1000000")
endif()

## Enable debug trace
if ( DEFINED ENV{CMAKE_DEBUG_TRACE} )
  add_definitions ( -DDEBUG_TRACE=1 )
endif()

## Enable AQL-profile new API
if ( NOT DEFINED ENV{CMAKE_CURR_API} )
  add_definitions ( -DAQLPROF_NEW_API=1 )
endif()

## Enable direct loading of AQL-profile HSA extension
if ( DEFINED ENV{CMAKE_LD_AQLPROFILE} )
  add_definitions ( -DROCP_LD_AQLPROFILE=1 )
endif()

## Make env vars
if ( NOT DEFINED CMAKE_BUILD_TYPE OR "${CMAKE_BUILD_TYPE}" STREQUAL "" )
  if ( DEFINED ENV{CMAKE_BUILD_TYPE} )
    set ( CMAKE_BUILD_TYPE $ENV{CMAKE_BUILD_TYPE} )
  endif()
endif()
if ( NOT DEFINED CMAKE_PREFIX_PATH AND DEFINED ENV{CMAKE_PREFIX_PATH} )
  set ( CMAKE_PREFIX_PATH $ENV{CMAKE_PREFIX_PATH} )
endif()

## Extend Compiler flags based on build type
string ( TOLOWER "${CMAKE_BUILD_TYPE}" CMAKE_BUILD_TYPE )
if ( "${CMAKE_BUILD_TYPE}" STREQUAL debug )
  set ( CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -ggdb" )
  set ( CMAKE_BUILD_TYPE "debug" )
else ()
  set ( CMAKE_BUILD_TYPE "release" )
endif ()

## Find hsa-runtime
find_package(hsa-runtime64 CONFIG REQUIRED HINTS ${CMAKE_INSTALL_PREFIX} PATHS /opt/rocm PATH_SUFFIXES lib/cmake/hsa-runtime64 )

# find KFD thunk
find_package(hsakmt CONFIG REQUIRED HINTS ${CMAKE_INSTALL_PREFIX} PATHS /opt/rocm PATH_SUFFIXES lib/cmake/hsakmt )

## Find ROCm
## TODO: Need a better method to find the ROCm path
find_path ( HSA_KMT_INC_PATH "hsakmt/hsakmt.h" )
if ( "${HSA_KMT_INC_PATH}" STREQUAL "" )
  get_target_property(HSA_KMT_INC_PATH hsakmt::hsakmt INTERFACE_INCLUDE_DIRECTORIES)
endif()
## Include path: /opt/rocm-ver/include. Go up one level to get ROCm  path
get_filename_component ( ROCM_ROOT_DIR "${HSA_KMT_INC_PATH}" DIRECTORY )

## Basic Tool Chain Information
message ( "----------Build-Type: ${CMAKE_BUILD_TYPE}" )
message ( "------------Compiler: ${CMAKE_CXX_COMPILER}" )
message ( "----Compiler-Version: ${CMAKE_CXX_COMPILER_VERSION}" )
message ( "-------ROCM_ROOT_DIR: ${ROCM_ROOT_DIR}" )
message ( "-----CMAKE_CXX_FLAGS: ${CMAKE_CXX_FLAGS}" )
message ( "---CMAKE_PREFIX_PATH: ${CMAKE_PREFIX_PATH}" )
message ( "---------GPU_TARGETS: ${GPU_TARGETS}" )

if ( "${ROCM_ROOT_DIR}" STREQUAL "" )
  message ( FATAL_ERROR "ROCM_ROOT_DIR is not found." )
endif ()

find_library ( FIND_AQL_PROFILE_LIB "libhsa-amd-aqlprofile64.so" HINTS ${CMAKE_INSTALL_PREFIX} PATHS ${ROCM_ROOT_DIR})
if (  NOT FIND_AQL_PROFILE_LIB )
  message ( FATAL_ERROR "AQL_PROFILE not installed. Please install AQL_PROFILE" )
endif()
