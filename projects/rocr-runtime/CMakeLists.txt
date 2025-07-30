################################################################################
##
## The University of Illinois/NCSA
## Open Source License (NCSA)
##
## Copyright (c) 2014-2025, Advanced Micro Devices, Inc. All rights reserved.
##
## Developed by:
##
##                 AMD Research and AMD HSA Software Development
##
##                 Advanced Micro Devices, Inc.
##
##                 www.amd.com
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


cmake_minimum_required(VERSION 3.7)

# Set the project name
project("rocr")

set(CMAKE_VERBOSE_MAKEFILE ON)
## Expose static library option
if ( NOT DEFINED BUILD_SHARED_LIBS )
  set ( BUILD_SHARED_LIBS ON )
endif()
set ( BUILD_SHARED_LIBS ${BUILD_SHARED_LIBS} CACHE BOOL "Build shared library (.so) or not.")

if (NOT DEFINED BUILD_ROCR)
  set(BUILD_ROCR ON)
endif()

function(add_rocm_subdir subdir subdir_assigns)
    message("add_rocm_subdir() -- " ${subdir})
    # message("  subdir_assigns before:" ${subdir_assigns} "EOM")
    string(STRIP "${subdir_assigns}" subdir_assigns)
    message("  subdir_assigns:" ${subdir_assigns} "EOM")

    # if the subdir_assigns is defined and  non-empty, then..

    if(NOT "${subdir_assigns}" STREQUAL "")
        foreach(assignment IN LISTS subdir_assigns)
            # The format of each var should be VARNAME=VALUE
            message("assignment: " ${assignment})
            string(REPLACE "=" ";" pair ${assignment})
            list(GET pair 0 var_name)
            list(GET pair 1 var_value)

            # Set variable locally for this function and for the subdirectory
            set(${var_name} "${var_value}")
            message("The value of ${var_name} is: ${${var_name}}")
        endforeach()
    endif()
    add_subdirectory(${subdir})
endfunction()

list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake_modules")
include(utils)


## Get version strings
get_version("1.18.0")
if (${ROCM_PATCH_VERSION})
  set(VERSION_PATCH ${ROCM_PATCH_VERSION})
endif()
set(SO_VERSION_STRING "${VERSION_MAJOR}.${VERSION_MINOR}.${VERSION_PATCH}")
set(PACKAGE_VERSION_STRING "${VERSION_MAJOR}.${VERSION_MINOR}.${VERSION_COMMIT_COUNT}")

if (NOT DEFINED BUILD_SHARED_LIBS)
	set(BUILD_SHARED_LIBS ON)
endif()

# Set hsa pkg dependency with rocprofiler-register package
# for Shared Library Only.
if (BUILD_SHARED_LIBS)
  set(HSA_DEP_ROCPROFILER_REGISTER ON CACHE INTERNAL "")
endif()

if (HSA_DEP_ROCPROFILER_REGISTER)
  string(APPEND CPACK_DEBIAN_BINARY_PACKAGE_DEPENDS ", rocprofiler-register")
  string(APPEND CPACK_RPM_BINARY_PACKAGE_REQUIRES " rocprofiler-register")
endif()

if (NOT DEFINED BUILD_THUNK_VIRTIO)
  set(BUILD_THUNK_VIRTIO OFF)
endif()

add_rocm_subdir(libhsakmt "${THUNK_DEFINITIONS}")
set_target_properties(hsakmt PROPERTIES
                                ARCHIVE_OUTPUT_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}/libhsakmt/archive"
                                LIBRARY_OUTPUT_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}/libhsakmt/lib"
                                RUNTIME_OUTPUT_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}/libhsakmt/runtime")

if (BUILD_THUNK_VIRTIO)
  add_rocm_subdir(libhsakmt/src/virtio "${THUNK_VIRTIO_DEFINITIONS}")
endif()

if (BUILD_ROCR)
  add_rocm_subdir(runtime/hsa-runtime "${ROCR_DEFINITIONS}")
  set_target_properties(hsa-runtime64 PROPERTIES
                               ARCHIVE_OUTPUT_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}/rocr/archive"
                               LIBRARY_OUTPUT_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}/rocr/lib"
                               RUNTIME_OUTPUT_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}/rocr/runtime")

  if (BUILD_SHARED_LIBS)
    add_dependencies(hsa-runtime64 hsakmt)
    if (BUILD_THUNK_VIRTIO)
      add_dependencies(hsa-runtime64 hsakmt_virtio)
    endif()
  else()
    add_dependencies(hsa-runtime64 hsakmt-staticdrm)
  endif()
endif()

# Optionally record the package's find module in the user's package cache.
if ( NOT DEFINED EXPORT_TO_USER_PACKAGE_REGISTRY )
  set ( EXPORT_TO_USER_PACKAGE_REGISTRY "off")
endif()
set ( EXPORT_TO_USER_PACKAGE_REGISTRY ${EXPORT_TO_USER_PACKAGE_REGISTRY} CACHE BOOL "Add cmake package config location to the user's cmake package registry.")
if(${EXPORT_TO_USER_PACKAGE_REGISTRY})
  # Enable writing to the registry
  set(CMAKE_EXPORT_PACKAGE_REGISTRY ON)
  # Generate a target file for the build
  export(TARGETS ${CORE_RUNTIME_NAME} NAMESPACE ${CORE_RUNTIME_NAME}:: FILE ${CORE_RUNTIME_NAME}Targets.cmake)
  # Record the package in the user's cache.
  export(PACKAGE ${CORE_RUNTIME_NAME})
endif()

## Packaging directives
set(CPACK_VERBOSE 1)
set(CPACK_GENERATOR "DEB;RPM" CACHE STRING "Package types to build")
set(ENABLE_LDCONFIG ON CACHE BOOL "Set library links and caches using ldconfig.")

# From libhsakmt:
set(CPACK_PACKAGING_INSTALL_PREFIX "${CMAKE_INSTALL_PREFIX}"  CACHE STRING "Default packaging prefix.")

if(DEFINED CPACK_PACKAGING_INSTALL_PREFIX)
  set(CPACK_RPM_EXCLUDE_FROM_AUTO_FILELIST_ADDITION "${CPACK_PACKAGING_INSTALL_PREFIX} ${CPACK_PACKAGING_INSTALL_PREFIX}/${CMAKE_INSTALL_BINDIR}")
endif()

# ASAN Package will have libraries and license file
if (ENABLE_ASAN_PACKAGING)
  # ASAN Package requires only asan component with libraries and license file
  set(CPACK_COMPONENTS_ALL asan)
else()
  set(CPACK_COMPONENTS_ALL binary dev)
endif()
set(CPACK_DEB_COMPONENT_INSTALL ON)
set(CPACK_RPM_COMPONENT_INSTALL ON)
set(CPACK_PACKAGE_VENDOR "Advanced Micro Devices, Inc.")
set(CPACK_PACKAGE_VERSION ${PACKAGE_VERSION_STRING})
set(CPACK_PACKAGE_CONTACT "AMD HSA Support <dl.HSA-Runtime-Support@amd.com>")
set(CPACK_COMPONENT_DESCRIPTION "AMD Heterogeneous System Architecture HSA - Linux HSA Runtime for Boltzmann (ROCm) platforms\nIncludes HSAKMT, the user-mode API interfaces used to interact with the ROCk driver.\n Contains the headers, pkgonfig and\n cmake files for ROCT.")
set(CPACK_COMPONENT_BINARY_DESCRIPTION "AMD Heterogeneous System Architecture HSA - Linux HSA Runtime for Boltzmann (ROCm) platforms")
set(CPACK_COMPONENT_DEV_DESCRIPTION "AMD Heterogeneous System Architecture HSA development package.\n This package contains the headers and cmake files for the rocr-runtime package.")
set(CPACK_COMPONENT_ASAN_DESCRIPTION "AMD Heterogeneous System Architecture HSA - Linux HSA instrumented libraries for Boltzmann (ROCm) platforms")

if (DEFINED ENV{ROCM_LIBPATCH_VERSION})
  set(CPACK_PACKAGE_VERSION "${CPACK_PACKAGE_VERSION}.$ENV{ROCM_LIBPATCH_VERSION}")
  message("Using CPACK_PACKAGE_VERSION ${CPACK_PACKAGE_VERSION}")
endif()

# Debian package specific variables
set(CPACK_DEBIAN_BINARY_PACKAGE_NAME "hsa-rocr")
set(CPACK_DEBIAN_DEV_PACKAGE_NAME "hsa-rocr-dev")
set(CPACK_DEBIAN_ASAN_PACKAGE_NAME "hsa-rocr-asan")
if (DEFINED ENV{CPACK_DEBIAN_PACKAGE_RELEASE})
  set(CPACK_DEBIAN_PACKAGE_RELEASE $ENV{CPACK_DEBIAN_PACKAGE_RELEASE})
else()
  set(CPACK_DEBIAN_PACKAGE_RELEASE "local")
endif()
message("Using CPACK_DEBIAN_PACKAGE_RELEASE ${CPACK_DEBIAN_PACKAGE_RELEASE}")
set(CPACK_DEBIAN_FILE_NAME "DEB-DEFAULT")
set(CPACK_DEBIAN_PACKAGE_HOMEPAGE "https://github.com/RadeonOpenCompute/ROCR-Runtime")

## Process the Debian install/remove scripts to update the CPACK variables
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/DEBIAN/Binary/postinst.in DEBIAN/Binary/postinst @ONLY)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/DEBIAN/Binary/prerm.in DEBIAN/Binary/prerm @ONLY)
file(COPY ${CMAKE_CURRENT_SOURCE_DIR}/DEBIAN/preinst DESTINATION ${CMAKE_CURRENT_BINARY_DIR}/DEBIAN)
set (CPACK_DEBIAN_BINARY_PACKAGE_CONTROL_EXTRA "DEBIAN/preinst;DEBIAN/Binary/postinst;DEBIAN/Binary/prerm")
# Needed since some packages still say they need hsakmt-roct
set(CPACK_DEBIAN_DEV_PACKAGE_REPLACES "hsakmt-roct,hsakmt-roct-dev,hsa-ext-rocr-dev")
set(CPACK_DEBIAN_DEV_PACKAGE_PROVIDES "hsakmt-roct,hsakmt-roct-dev,hsa-ext-rocr-dev")
#TODO: hsa-ext-rocr-dev can be added to conflicts list and remove CPACK_DEBIAN_DEV_PACKAGE_BREAKS
set(CPACK_DEBIAN_DEV_PACKAGE_CONFLICTS "hsakmt-roct,hsakmt-roct-dev")
# package dependencies
set(CPACK_DEBIAN_PACKAGE_DEPENDS "libdrm-amdgpu-dev | libdrm-dev, rocm-core")
set(CPACK_DEBIAN_PACKAGE_RECOMMENDS "libdrm-amdgpu-dev")
# Setting devel package dependendent version
set(CPACK_DEBIAN_DEV_PACKAGE_DEPENDS "libdrm-amdgpu-dev | libdrm-dev, rocm-core, hsa-rocr")

set(CPACK_DEBIAN_DEV_PACKAGE_RECOMMENDS "libdrm-amdgpu-dev")

set(CPACK_DEBIAN_BINARY_PACKAGE_DEPENDS "libdrm-amdgpu-amdgpu1 | libdrm-amdgpu1, libnuma1, libelf1")
set(CPACK_DEBIAN_ASAN_PACKAGE_DEPENDS "libdrm-amdgpu-dev | libdrm-dev, rocm-core-asan, libdrm-amdgpu-amdgpu1 | libdrm-amdgpu1, libnuma1, libelf1")
set(CPACK_DEBIAN_ASAN_PACKAGE_RECOMMENDS "libdrm-amdgpu-dev")

set(CPACK_DEBIAN_BINARY_PACKAGE_RECOMMENDS "libdrm-amdgpu-amdgpu1")
if (ROCM_DEP_ROCMCORE)
  string(APPEND CPACK_DEBIAN_BINARY_PACKAGE_DEPENDS ", rocm-core")
  string(APPEND CPACK_DEBIAN_ASAN_PACKAGE_DEPENDS ", rocm-core-asan")
endif()
if (HSA_DEP_ROCPROFILER_REGISTER)
  string(APPEND CPACK_DEBIAN_BINARY_PACKAGE_DEPENDS ", rocprofiler-register")
endif()
# Declare package relationships (hsa-ext-rocr-dev is a legacy package that we subsume)
set(CPACK_DEBIAN_DEV_PACKAGE_BREAKS "hsa-ext-rocr-dev")

# RPM package specific variables
set(EL7_DISTRO "FALSE")
Checksetel7(EL7_DISTRO)
set(CPACK_RPM_BINARY_PACKAGE_NAME "hsa-rocr")
# Since we changed the package name to match RPM specs, take care of older builds that had -dev installed
# Also cover the fact that this now replaces the old binary package hsakmt-roct
set(CPACK_RPM_DEV_PACKAGE_PROVIDES "hsakmt-roct,hsakmt-roct-devel,hsakmt-roct-dev,hsa-ext-rocr-dev")
set(CPACK_RPM_DEV_PACKAGE_OBSOLETES "hsakmt-roct,hsakmt-roct-devel,hsakmt-roct-dev,hsa-ext-rocr-dev")

set(CPACK_RPM_DEV_PACKAGE_NAME "hsa-rocr-devel")
set(CPACK_RPM_ASAN_PACKAGE_NAME "hsa-rocr-asan")
if (DEFINED ENV{CPACK_RPM_PACKAGE_RELEASE})
  set(CPACK_RPM_PACKAGE_RELEASE $ENV{CPACK_RPM_PACKAGE_RELEASE})
else()
  set(CPACK_RPM_PACKAGE_RELEASE "local")
endif()

string(APPEND CPACK_RPM_PACKAGE_RELEASE "%{?dist}")
set(CPACK_RPM_FILE_NAME "RPM-DEFAULT")
message("CPACK_RPM_PACKAGE_RELEASE: ${CPACK_RPM_PACKAGE_RELEASE}")
set(CPACK_RPM_PACKAGE_LICENSE "NCSA")

## Process the Rpm install/remove scripts to update the CPACK variables
configure_file("${CMAKE_CURRENT_SOURCE_DIR}/RPM/Binary/post.in" RPM/Binary/post @ONLY)
configure_file("${CMAKE_CURRENT_SOURCE_DIR}/RPM/Binary/postun.in" RPM/Binary/postun @ONLY)
file(COPY ${CMAKE_CURRENT_SOURCE_DIR}/RPM/preinst DESTINATION ${CMAKE_CURRENT_BINARY_DIR}/RPM)
set (CPACK_RPM_PRE_INSTALL_SCRIPT_FILE "${CMAKE_CURRENT_BINARY_DIR}/RPM/preinst")

set(CPACK_RPM_BINARY_POST_INSTALL_SCRIPT_FILE "${CMAKE_CURRENT_BINARY_DIR}/RPM/Binary/post")
set(CPACK_RPM_BINARY_POST_UNINSTALL_SCRIPT_FILE "${CMAKE_CURRENT_BINARY_DIR}/RPM/Binary/postun")

# package dependencies
set(CPACK_RPM_DEV_PACKAGE_REQUIRES "rocm-core , hsa-rocr")

#
if (${EL7_DISTRO} STREQUAL "TRUE")
  set(CPACK_RPM_BINARY_PACKAGE_REQUIRES "libdrm-amdgpu, numactl-libs")
  set(CPACK_RPM_ASAN_PACKAGE_REQUIRES "libdrm-amdgpu, numactl-libs, libdrm-amdgpu-devel")
  set(CPACK_RPM_PACKAGE_REQUIRES "libdrm-amdgpu-devel")
  string(APPEND CPACK_RPM_DEV_PACKAGE_REQUIRES ", libdrm-amdgpu-devel")
else()
  set(CPACK_RPM_BINARY_PACKAGE_REQUIRES "(libdrm-amdgpu or libdrm or libdrm_amdgpu1), (libnuma1 or numactl-libs)")
  set(CPACK_RPM_ASAN_PACKAGE_REQUIRES "(libdrm-amdgpu or libdrm or libdrm_amdgpu1), (libnuma1 or numactl-libs), (libdrm-amdgpu-devel or libdrm-devel)")
  set(CPACK_RPM_USER_BINARY_SPECFILE "${CMAKE_CURRENT_SOURCE_DIR}/RPM/hsa-rocr.spec.in")
  set(CPACK_RPM_PACKAGE_RECOMMENDS "libdrm-amdgpu, libdrm-amdgpu-devel")

  set(CPACK_RPM_PACKAGE_REQUIRES "(libdrm-amdgpu-devel or libdrm-devel)")
  string(APPEND CPACK_RPM_DEV_PACKAGE_REQUIRES ", (libdrm-amdgpu-devel or libdrm-devel)")
  set(CPACK_RPM_DEV_PACKAGE_RECOMMENDS "libdrm-amdgpu-devel")
  set(CPACK_RPM_ASAN_PACKAGE_RECOMMENDS "libdrm-amdgpu-devel")

endif()

if (ROCM_DEP_ROCMCORE)
  string(APPEND CPACK_RPM_BINARY_PACKAGE_REQUIRES " rocm-core")
  string(APPEND CPACK_RPM_ASAN_PACKAGE_REQUIRES " rocm-core-asan")
else()
  string(REGEX REPLACE ",? ?rocm-core" "" CPACK_RPM_PACKAGE_REQUIRES ${CPACK_RPM_PACKAGE_REQUIRES})
  string(REGEX REPLACE ",? ?rocm-core" "" CPACK_DEBIAN_PACKAGE_DEPENDS ${CPACK_DEBIAN_PACKAGE_DEPENDS})
  string(REGEX REPLACE ",? ?rocm-core" "" CPACK_RPM_DEV_PACKAGE_REQUIRES ${CPACK_RPM_DEV_PACKAGE_REQUIRES})
  string(REGEX REPLACE ",? ?rocm-core" "" CPACK_DEBIAN_DEV_PACKAGE_DEPENDS ${CPACK_DEBIAN_DEV_PACKAGE_DEPENDS})
  string(REGEX REPLACE ",? ?rocm-core-asan" "" CPACK_RPM_ASAN_PACKAGE_REQUIRES ${CPACK_RPM_ASAN_PACKAGE_REQUIRES})
  string(REGEX REPLACE ",? ?rocm-core-asan" "" CPACK_DEBIAN_ASAN_PACKAGE_DEPENDS ${CPACK_DEBIAN_ASAN_PACKAGE_DEPENDS})
endif()
if (HSA_DEP_ROCPROFILER_REGISTER)
  string(APPEND CPACK_RPM_BINARY_PACKAGE_REQUIRES " rocprofiler-register")
endif()

if(NOT BUILD_SHARED_LIBS)
  # Suffix package name with static
  set(CPACK_RPM_STATIC_PACKAGE_NAME "hsa-rocr-static-devel")
  set(CPACK_DEBIAN_STATIC_PACKAGE_NAME "hsa-rocr-static-dev")
  set(CPACK_COMPONENT_STATIC_DESCRIPTION "HSA (Heterogenous System Architecture) core runtime - Linux static libraries")
  set(CPACK_RPM_STATIC_PACKAGE_REQUIRES "${CPACK_RPM_BINARY_PACKAGE_REQUIRES}")
  set(CPACK_DEBIAN_STATIC_PACKAGE_DEPENDS "${CPACK_DEBIAN_BINARY_PACKAGE_DEPENDS}")
endif()

## Include packaging
include(CPack)

# static package generation
# Group binary and dev component to single package
if(NOT BUILD_SHARED_LIBS)
    cpack_add_component_group("static")
    cpack_add_component(binary  GROUP static)
    cpack_add_component(dev GROUP static)
endif()

cpack_add_component(asan
  DISPLAY_NAME "ASAN"
  DESCRIPTION "ASAN libraries for rocr-runtime")
