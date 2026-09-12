#
# Copyright (C) 2018 Advanced Micro Devices, Inc. All Rights Reserved.
#
# Permission is hereby granted, free of charge, to any person obtaining a
# copy of this software and associated documentation files (the "Software"),
# to deal in the Software without restriction, including without limitation
# the rights to use, copy, modify, merge, publish, distribute, sublicense,
# and/or sell copies of the Software, and to permit persons to whom the
# Software is furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
# THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR
# OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE,
# ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR
# OTHER DEALINGS IN THE SOFTWARE.
#
#

# If environment variable DRM_DIR or LIBHSAKMT_PATH is set, the script
# will pick up the corresponding libraries from those pathes.

cmake_minimum_required(VERSION 3.25 FATAL_ERROR)

project(KFDTest)

# For DEB/RPM generation
set ( CPACK_PACKAGE_NAME "kfdtest" )
set ( CPACK_PACKAGE_CONTACT "Advanced Micro Devices Inc." )
set ( CPACK_PACKAGE_DESCRIPTION "This package includes kfdtest, the list of excluded tests for each ASIC, and a convenience script to run the test suite" )
set ( CPACK_PACKAGE_DESCRIPTION_SUMMARY "Test suite for ROCK/KFD" )

# Make proper version for appending
# Default Value is 99999, setting it first
set(ROCM_VERSION_FOR_PACKAGE "99999")
if(DEFINED ENV{ROCM_LIBPATCH_VERSION})
  set(ROCM_VERSION_FOR_PACKAGE $ENV{ROCM_LIBPATCH_VERSION})
endif()

set ( CPACK_PACKAGE_VERSION_MAJOR "1" )
set ( CPACK_PACKAGE_VERSION_MINOR "0" )
set ( CPACK_PACKAGE_VERSION_PATCH "0" )
set ( CPACK_PACKAGE_HOMEPAGE_URL "https://github.com/ROCm/ROCR-Runtime/" )
set ( CPACK_DEBIAN_FILE_NAME "DEB-DEFAULT")
set ( CPACK_RPM_FILE_NAME "RPM-DEFAULT")

## Debian package values
set ( CPACK_DEBIAN_PACKAGE_RELEASE "local" )
if( DEFINED ENV{CPACK_DEBIAN_PACKAGE_RELEASE} )
  set ( CPACK_DEBIAN_PACKAGE_RELEASE $ENV{CPACK_DEBIAN_PACKAGE_RELEASE} )
endif()
## RPM package variables
set ( CPACK_RPM_PACKAGE_RELEASE "local" )
if( DEFINED ENV{CPACK_RPM_PACKAGE_RELEASE} )
  set ( CPACK_RPM_PACKAGE_RELEASE $ENV{CPACK_RPM_PACKAGE_RELEASE} )
endif()

## Note: rpm --eval %{?dist} will evaluate to NULL in Debian
## So Debian distros won't append dist tag to CPACK_RPM_PACKAGE_RELEASE.
## Also for debian package name , the dist tag is added from build env
execute_process( COMMAND rpm --eval %{?dist}
                 RESULT_VARIABLE PROC_RESULT
                 OUTPUT_VARIABLE EVAL_RESULT
                 OUTPUT_STRIP_TRAILING_WHITESPACE )
message("RESULT_VARIABLE ${PROC_RESULT} OUTPUT_VARIABLE: ${EVAL_RESULT}")
## Add distribution tag to rpm package name
if ( PROC_RESULT EQUAL "0" AND NOT EVAL_RESULT STREQUAL "" )
  string ( APPEND CPACK_RPM_PACKAGE_RELEASE "%{?dist}" )
endif()

set(PACKAGE_VERSION_STR "${CPACK_PACKAGE_VERSION_MAJOR}.${CPACK_PACKAGE_VERSION_MINOR}.${CPACK_PACKAGE_VERSION_PATCH}.${ROCM_VERSION_FOR_PACKAGE}")
set(CPACK_PACKAGE_VERSION "${PACKAGE_VERSION_STR}")

## Define default variable and variables for the optional build target hsakmt-dev
set ( SOURCE_DIR ${CMAKE_CURRENT_SOURCE_DIR} CACHE STRING "Location of hsakmt source code." )
set ( CMAKE_INSTALL_PREFIX "/opt/rocm"  CACHE STRING "Default installation directory." )
set ( CPACK_PACKAGING_INSTALL_PREFIX "${CMAKE_INSTALL_PREFIX}"  CACHE STRING "Default packaging prefix." )
set ( CPACK_GENERATOR "DEB;RPM"  CACHE STRING "Default packaging generators." )

# Debian package specific variables
set ( CPACK_DEBIAN_PACKAGE_HOMEPAGE "https://github.com/ROCm/ROCR-Runtime/" )
set ( CPACK_DEBIAN_PACKAGE_DEPENDS "rocm-core" )

# RPM package specific variables
set (CPACK_RPM_PACKAGE_REQUIRES "rocm-core")

#set ( CMAKE_VERBOSE_MAKEFILE on )

find_package(PkgConfig)

# Prepend DRM_DIR to CMAKE_PREFIX_PATH if set
if(DRM_DIR)
    list(PREPEND CMAKE_PREFIX_PATH "${DRM_DIR}")
endif()
# The module name passed to pkg_check_modules() is determined by the
# name of file *.pc
pkg_check_modules(DRM REQUIRED libdrm)
pkg_check_modules(DRM_AMDGPU REQUIRED libdrm_amdgpu)
include_directories(${DRM_AMDGPU_INCLUDE_DIRS})

# CMP0074: make find_package() use <PackageName>_ROOT variables (CMake 3.12+)
# This allows NUMA_ROOT, hsakmt_ROOT, etc. to guide package discovery in both
# TheRock (via CMake cache variables) and standalone builds (via environment).
# Without this policy, find_package() ignores _ROOT hints, breaking custom paths.
cmake_policy( SET CMP0074 NEW )

# Find NUMA library
# Try CONFIG mode first (TheRock), fall back to manual discovery (standalone)
find_package(NUMA CONFIG QUIET)
if(NUMA_FOUND)
    # CONFIG mode provides numa::numa (lowercase)
    # Use it directly if it exists, otherwise create NUMA::NUMA from variables
    if(NOT TARGET numa::numa AND NOT TARGET NUMA::NUMA)
        # CONFIG provided variables but no target - forward to NUMA_LIBRARIES
        add_library(NUMA::NUMA INTERFACE IMPORTED)
        set_target_properties(NUMA::NUMA PROPERTIES
            INTERFACE_LINK_LIBRARIES "${NUMA_LIBRARIES}")
    elseif(TARGET numa::numa AND NOT TARGET NUMA::NUMA)
        # Create NUMA::NUMA as an alias for compatibility
        # Note: We use the lowercase target directly in target_link_libraries
        add_library(NUMA::NUMA INTERFACE IMPORTED)
        target_link_libraries(NUMA::NUMA INTERFACE numa::numa)
    endif()
else()
    find_library(NUMA_LIBRARY NAMES numa PATHS /usr/lib /usr/local/lib)
    if(NOT NUMA_LIBRARY)
        message(FATAL_ERROR "Could not find NUMA library")
    endif()
    add_library(NUMA::NUMA UNKNOWN IMPORTED)
    set_target_properties(NUMA::NUMA PROPERTIES
        IMPORTED_LOCATION "${NUMA_LIBRARY}")
    message(STATUS "Found NUMA: ${NUMA_LIBRARY}")
endif()

# Find libhsakmt library
# Try CONFIG mode first (TheRock), fall back to manual discovery (standalone)
find_package(hsakmt CONFIG QUIET)
if(NOT hsakmt_FOUND)
    # Manual discovery with multiple fallback strategies
    # kfdtest requires static library for libhsakmt
    if(LIBHSAKMT_PATH)
        message(STATUS "Using LIBHSAKMT_PATH: ${LIBHSAKMT_PATH}")
        find_library(HSAKMT_LIBRARY NAMES libhsakmt.a PATHS "${LIBHSAKMT_PATH}" NO_DEFAULT_PATH)
        # Find include directory relative to library
        get_filename_component(HSAKMT_LIB_DIR "${LIBHSAKMT_PATH}" ABSOLUTE)
        find_path(HSAKMT_INCLUDE_DIR NAMES hsakmt.h
            PATHS "${HSAKMT_LIB_DIR}/../include" "${HSAKMT_LIB_DIR}/../../include"
            NO_DEFAULT_PATH)
    elseif(DEFINED ENV{LIBHSAKMT_PATH})
        message(STATUS "Using LIBHSAKMT_PATH from environment: $ENV{LIBHSAKMT_PATH}")
        find_library(HSAKMT_LIBRARY NAMES libhsakmt.a PATHS "$ENV{LIBHSAKMT_PATH}" NO_DEFAULT_PATH)
        get_filename_component(HSAKMT_LIB_DIR "$ENV{LIBHSAKMT_PATH}" ABSOLUTE)
        find_path(HSAKMT_INCLUDE_DIR NAMES hsakmt.h
            PATHS "${HSAKMT_LIB_DIR}/../include" "${HSAKMT_LIB_DIR}/../../include"
            NO_DEFAULT_PATH)
    else()
        # Try pkg-config
        if(ROCM_INSTALL_PATH)
            set(ENV{PKG_CONFIG_PATH} ${ROCM_INSTALL_PATH}/share/pkgconfig)
        else()
            set(ENV{PKG_CONFIG_PATH} /opt/rocm/share/pkgconfig)
        endif()

        pkg_check_modules(HSAKMT libhsakmt QUIET)

        if(HSAKMT_FOUND)
            find_library(HSAKMT_LIBRARY NAMES libhsakmt.a PATHS ${HSAKMT_LIBRARY_DIRS} NO_DEFAULT_PATH)
            find_path(HSAKMT_INCLUDE_DIR NAMES hsakmt.h PATHS ${HSAKMT_INCLUDE_DIRS} NO_DEFAULT_PATH)
        else()
            # Fall back to standard search paths
            find_library(HSAKMT_LIBRARY NAMES libhsakmt.a
                PATHS /opt/rocm/lib /usr/lib/x86_64-linux-gnu /usr/lib /usr/local/lib)
            find_path(HSAKMT_INCLUDE_DIR NAMES hsakmt.h
                PATHS /opt/rocm/include /usr/include /usr/local/include)
        endif()
    endif()

    if(NOT HSAKMT_LIBRARY)
        message(FATAL_ERROR "Could not find libhsakmt. Set LIBHSAKMT_PATH to the directory containing libhsakmt.a")
    endif()

    # Create imported target
    add_library(hsakmt::hsakmt STATIC IMPORTED)
    set_target_properties(hsakmt::hsakmt PROPERTIES
        IMPORTED_LOCATION "${HSAKMT_LIBRARY}")

    # Add include directory if found
    if(HSAKMT_INCLUDE_DIR)
        set_target_properties(hsakmt::hsakmt PROPERTIES
            INTERFACE_INCLUDE_DIRECTORIES "${HSAKMT_INCLUDE_DIR}")
        message(STATUS "Found libhsakmt: ${HSAKMT_LIBRARY} (include: ${HSAKMT_INCLUDE_DIR})")
    else()
        message(STATUS "Found libhsakmt: ${HSAKMT_LIBRARY}")
    endif()
endif()

# Find LLVM library
# Uses LLVM_DIR (CMake/env variable) or CMAKE_PREFIX_PATH if set, otherwise system paths
find_package( LLVM REQUIRED CONFIG )

if( ${LLVM_PACKAGE_VERSION} VERSION_LESS "7.0" )
    message( FATAL_ERROR "Requires LLVM 7.0 or greater "
        "(found ${LLVM_PACKAGE_VERSION})" )
elseif( ${LLVM_PACKAGE_VERSION} VERSION_LESS "14.0" )
    message( WARNING "Not using latest LLVM version. "
        "Some ASIC targets may not work!" )
endif()

message( STATUS "Found LLVM ${LLVM_PACKAGE_VERSION}" )
message( STATUS "Using LLVMConfig.cmake in: ${LLVM_DIR}" )

include_directories(${LLVM_INCLUDE_DIRS})
separate_arguments(LLVM_DEFINITIONS_LIST NATIVE_COMMAND ${LLVM_DEFINITIONS})
add_definitions(${LLVM_DEFINITIONS_LIST})

if (LLVM_LINK_LLVM_DYLIB)
  set(llvm_libs LLVM)
else()
  llvm_map_components_to_libnames(llvm_libs AMDGPUAsmParser Core Support)
endif()

include_directories(${PROJECT_SOURCE_DIR}/gtest-1.6.0)
include_directories(${PROJECT_SOURCE_DIR}/include)
include_directories(${PROJECT_SOURCE_DIR}/../../include)
include_directories(${PROJECT_SOURCE_DIR}/../../libhsakmt/include)

include_directories(${DRM_INCLUDE_DIRS})

# Platform specific includes (i.e kfd_ioct.h on Linux)
if (CMAKE_SYSTEM_NAME MATCHES "Linux")
      include_directories(${PROJECT_SOURCE_DIR}/../../include/hsakmt/linux)
endif()

set (SRC_FILES gtest-1.6.0/gtest-all.cpp

  src/AqlQueue.cpp
  src/BasePacket.cpp
  src/BaseDebug.cpp
  src/BaseQueue.cpp
  src/Dispatch.cpp
  src/GoogleTestExtension.cpp
  src/IndirectBuffer.cpp
  src/Assemble.cpp
  src/ShaderStore.cpp
  src/LinuxOSWrapper.cpp
  src/PM4Packet.cpp
  src/PM4Queue.cpp
  src/RDMAUtil.cpp
  src/SDMAPacket.cpp
  src/SDMAQueue.cpp
  src/KFDBaseComponentTest.cpp
  src/KFDMultiProcessTest.cpp
  src/KFDTestMain.cpp
  src/KFDTestUtil.cpp
  src/KFDTestUtilQueue.cpp

  src/KFDOpenCloseKFDTest.cpp
  src/KFDTopologyTest.cpp
  src/KFDMemoryTest.cpp
  src/KFDLocalMemoryTest.cpp
  src/KFDEventTest.cpp
  src/KFDQMTest.cpp
  src/KFDCWSRTest.cpp
  src/KFDExceptionTest.cpp
  src/KFDGraphicsInterop.cpp
  src/KFDPerfCounters.cpp
  src/KFDDBGTest.cpp
  src/KFDGWSTest.cpp
  src/KFDIPCTest.cpp
  src/KFDASMTest.cpp

  src/KFDEvictTest.cpp
  src/KFDHWSTest.cpp
  src/KFDPerformanceTest.cpp
  src/KFDPMTest.cpp
  src/KFDSVMRangeTest.cpp
  src/KFDSVMRangeReGrant_test.cpp
  src/KFDSVMRangePartialPage_test.cpp
  src/KFDSVMEvictTest.cpp
  src/KFDRASTest.cpp
  src/KFDPCSamplingTest.cpp
  src/KFDNegativeTest.cpp
  src/RDMATest.cpp
  src/KFDAISTest.cpp)

message( STATUS "PROJECT_SOURCE_DIR:" ${PROJECT_SOURCE_DIR} )
#message( STATUS "SRC_FILES: ")
#foreach(file ${SRC_FILES})
#  message(STATUS "${file}")
#endforeach()

#add_definitions(-Wall -std=c++11)

if ( "${CMAKE_C_COMPILER_VERSION}" STRGREATER "4.8.0")
## Add --enable-new-dtags to generate DT_RUNPATH
set ( CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=gnu++17 -Wl,--enable-new-dtags" )
endif()
if ( "${CMAKE_BUILD_TYPE}" STREQUAL Release )
    set ( CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O2" )
else ()
    set ( CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -g" )
endif ()

## Address Sanitize Flag
if ( ADDRESS_SANITIZER OR THEROCK_SANITIZER STREQUAL "ASAN" OR THEROCK_SANITIZER STREQUAL "HOST_ASAN" )
    set ( CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fsanitize=address" )
    set ( CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -fsanitize=address" )
endif ()

add_executable(kfdtest ${SRC_FILES})

# Context-version test support
if( ${HSAKMT_CTX} )
    target_compile_definitions(kfdtest PRIVATE -DHSAKMT_CTX=1)
    # Secondary context test support
    if( ${HSAKMT_SECONDARY_CTX} )
        target_compile_definitions(kfdtest PRIVATE -DHSAKMT_SECONDARY_CTX=1)
    endif()
endif()

target_link_libraries(kfdtest
    PRIVATE
        hsakmt::hsakmt
        NUMA::NUMA
        ${DRM_LDFLAGS}
        ${DRM_AMDGPU_LDFLAGS}
        ${llvm_libs}
        pthread
        m
        stdc++
        rt
        ${CMAKE_DL_LIBS})

configure_file ( scripts/kfdtest.exclude kfdtest.exclude COPYONLY )
configure_file ( scripts/run_kfdtest.sh run_kfdtest.sh COPYONLY )

install( PROGRAMS ${CMAKE_CURRENT_BINARY_DIR}/kfdtest ${CMAKE_CURRENT_BINARY_DIR}/run_kfdtest.sh
	DESTINATION bin )
install( FILES ${CMAKE_CURRENT_BINARY_DIR}/kfdtest.exclude
	DESTINATION share/kfdtest )
# Remove dependency on rocm-core if -DROCM_DEP_ROCMCORE=ON not given to cmake
if(NOT ROCM_DEP_ROCMCORE)
    string(REGEX REPLACE ",? ?rocm-core" "" CPACK_RPM_PACKAGE_REQUIRES ${CPACK_RPM_PACKAGE_REQUIRES})
    string(REGEX REPLACE ",? ?rocm-core" "" CPACK_DEBIAN_PACKAGE_DEPENDS ${CPACK_DEBIAN_PACKAGE_DEPENDS})
endif()
include ( CPack )
