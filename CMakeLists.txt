################################################################################
##
## Copyright (c) 2016 Advanced Micro Devices, Inc. All rights reserved.
##
## MIT LICENSE:
## Permission is hereby granted, free of charge, to any person obtaining a copy of
## this software and associated documentation files (the "Software"), to deal in
## the Software without restriction, including without limitation the rights to
## use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies
## of the Software, and to permit persons to whom the Software is furnished to do
## so, subject to the following conditions:
##
## The above copyright notice and this permission notice shall be included in all
## copies or substantial portions of the Software.
##
## THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
## IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
## FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL THE
## AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
## LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
## OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
## SOFTWARE.
##
################################################################################

cmake_minimum_required ( VERSION 3.6.3 )

set ( HSAKMT "hsakmt" )
set ( HSAKMT_PACKAGE "hsakmt-roct" )
set ( HSAKMT_COMPONENT "lib${HSAKMT}" )
set ( HSAKMT_TARGET "${HSAKMT}" )

project ( ${HSAKMT_TARGET} VERSION 1.9.0)

# Build thunk as static by default
if ( NOT DEFINED BUILD_SHARED_LIBS )
  set ( BUILD_SHARED_LIBS "on" )
endif()
set ( BUILD_SHARED_LIBS ${BUILD_SHARED_LIBS} CACHE BOOL "Build shared library (.so) or not.")

# Optionally, build HSAKMT with ccache.
set(ROCM_CCACHE_BUILD OFF CACHE BOOL "Set to ON for a ccache enabled build")
if (ROCM_CCACHE_BUILD)
  find_program(CCACHE_PROGRAM ccache)
  if (CCACHE_PROGRAM)
    set_property(GLOBAL PROPERTY RULE_LAUNCH_COMPILE ${CCACHE_PROGRAM})
  else()
    message(WARNING "Unable to find ccache. Falling back to real compiler")
  endif() # if (CCACHE_PROGRAM)
endif() # if (ROCM_CCACHE_BUILD)

list( APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake_modules" )

## Include common cmake modules
include ( utils )
include ( GNUInstallDirs )

## Setup the package version.
get_version ( "1.0.0" )

set ( BUILD_VERSION_MAJOR ${VERSION_MAJOR} )
set ( BUILD_VERSION_MINOR ${VERSION_MINOR} )
set ( BUILD_VERSION_PATCH ${VERSION_PATCH} )

set ( LIB_VERSION_MAJOR ${VERSION_MAJOR})
set ( LIB_VERSION_MINOR ${VERSION_MINOR})
set ( LIB_VERSION_PATCH ${VERSION_PATCH})

set ( LIB_VERSION_STRING "${LIB_VERSION_MAJOR}.${LIB_VERSION_MINOR}.${LIB_VERSION_PATCH}" )

if ( DEFINED VERSION_BUILD AND NOT ${VERSION_BUILD} STREQUAL "" )
    message ( "VERSION BUILD DEFINED ${VERSION_BUILD}" )
    set ( BUILD_VERSION_PATCH "${BUILD_VERSION_PATCH}-${VERSION_BUILD}" )
endif ()
set ( BUILD_VERSION_STRING "${BUILD_VERSION_MAJOR}.${BUILD_VERSION_MINOR}.${BUILD_VERSION_PATCH}" )

## Compiler flags
set (HSAKMT_CXX_FLAGS -Wno-conversion-null )

set ( HSAKMT_LINKER_SCRIPT "${CMAKE_CURRENT_SOURCE_DIR}/src/libhsakmt.ver" )

## Linker Flags
## Add --enable-new-dtags to generate DT_RUNPATH
set (HSAKMT_LINK_FLAGS "${HSAKMT_LINK_FLAGS} -Wl,--enable-new-dtags -Wl,--version-script=${HSAKMT_LINKER_SCRIPT} -Wl,-soname=${HSAKMT_COMPONENT}.so.${LIB_VERSION_MAJOR} -Wl,-z,nodelete")

## Address Sanitize Flag
if ( ${ADDRESS_SANITIZER} )
    set ( HSAKMT_CXX_FLAGS "${HSAKMT_CXX_FLAGS}" -fsanitize=address )
    set ( HSAKMT_LINK_FLAGS "${HSAKMT_LINK_FLAGS} -fsanitize=address" )
    if ( BUILD_SHARED_LIBS )
        set ( HSAKMT_LINK_FLAGS "${HSAKMT_LINK_FLAGS} -shared-libsan" )
    else ()
        set ( HSAKMT_LINK_FLAGS "${HSAKMT_LINK_FLAGS} -static-libsan" )
    endif ()
endif ()

## Source files
set ( HSAKMT_SRC "src/debug.cpp"
                 "src/events.cpp"
                 "src/globals.cpp"
                 "src/memory.cpp"
                 "src/libdrm.cpp"
                 "src/openclose.cpp"
                 "src/perfctr.cpp"
                 "src/queues.cpp"
                 "src/time.cpp"
                 "src/topology.cpp"
                 "src/spm.cpp"
                 "src/version.cpp"
                 "src/svm.cpp"
                 "src/pc_sampling.cpp"
                 "src/rocr_proxy/rocr_proxy.cpp"
                 "src/wddm/device.cpp"
                 "src/wddm/gpu_memory.cpp"
                 "src/wddm/va_mgr.cpp"
                 "src/wddm/queue.cpp"
                 "src/wddm/cmd_util.cpp" )

## Declare the library target name
add_library ( ${HSAKMT_TARGET} "")

## Add sources
target_sources ( ${HSAKMT_TARGET} PRIVATE ${HSAKMT_SRC} )

## Add headers.  The public headers need to point at their location in both build and install
## directory layouts.  This declaration allows publishing library use data to downstream clients.
target_include_directories( ${HSAKMT_TARGET}
  PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>
  PRIVATE
  ${DK_HEADER}/ms_sdk/include/shared
  ${DK_HEADER}/inc/shared
  ${DK_HEADER}/inc/asic_reg
  ${CMAKE_CURRENT_SOURCE_DIR}/src
  ${CMAKE_CURRENT_SOURCE_DIR}/../runtime )

add_compile_definitions(LINUX __AMD64__ LITTLEENDIAN_CPU HSA_LARGE_MODEL)
target_link_directories(${HSAKMT_TARGET} PRIVATE ${DXCORE_LIB_PATH})
target_link_libraries(${HSAKMT_TARGET} PRIVATE dxcore)

set_property(TARGET ${HSAKMT_TARGET} PROPERTY LINK_FLAGS ${HSAKMT_LINK_FLAGS})

## Set the VERSION and SOVERSION values
set_property ( TARGET ${HSAKMT_TARGET} PROPERTY VERSION "${LIB_VERSION_STRING}" )
set_property ( TARGET ${HSAKMT_TARGET} PROPERTY SOVERSION "${LIB_VERSION_MAJOR}" )

find_package(PkgConfig)
# get OS-info for OS-specific build dependencies
get_os_info()

find_package(PkgConfig)
# Check for libraries required for building
find_library(LIBC NAMES c REQUIRED)
find_library(NUMA NAMES numa REQUIRED)
message(STATUS "LIBC:" ${LIBC})
message(STATUS "NUMA:" ${NUMA})

## If environment variable DRM_DIR is set, the script
## will pick up the corresponding libraries from that path.
list (PREPEND CMAKE_PREFIX_PATH "${DRM_DIR}")

# The module name passed to pkg_check_modules() is determined by the
# name of file *.pc
pkg_check_modules(DRM REQUIRED IMPORTED_TARGET libdrm)
pkg_check_modules(DRM_AMDGPU REQUIRED IMPORTED_TARGET libdrm_amdgpu)
include_directories(${DRM_AMDGPU_INCLUDE_DIRS})
include_directories(${DRM_INCLUDE_DIRS})

target_link_libraries ( ${HSAKMT_TARGET}
  PRIVATE ${DRM_LDFLAGS} ${DRM_AMDGPU_LDFLAGS} pthread rt c numa ${CMAKE_DL_LIBS}
)

target_compile_options(${HSAKMT_TARGET} PRIVATE ${DRM_CFLAGS} ${HSAKMT_CXX_FLAGS})

## Define default paths and packages.
if( CMAKE_INSTALL_PREFIX_INITIALIZED_TO_DEFAULT )
  set ( CMAKE_INSTALL_PREFIX "/opt/rocm" )
endif()
set ( CMAKE_INSTALL_PREFIX ${CMAKE_INSTALL_PREFIX} CACHE STRING "Default installation directory." FORCE )
set ( CPACK_PACKAGING_INSTALL_PREFIX "${CMAKE_INSTALL_PREFIX}"  CACHE STRING "Default packaging prefix." )
set ( CPACK_GENERATOR "DEB;RPM"  CACHE STRING "Default packaging generators." )

## set components
if ( ENABLE_ASAN_PACKAGING )
  # ASAN Package requires only asan component with libraries and license file
  set ( CPACK_COMPONENTS_ALL asan )
else()
  set ( CPACK_COMPONENTS_ALL devel )
endif()

# Installs binaries and exports the library usage data to ${HSAKMT_TARGET}Targets
install ( TARGETS ${HSAKMT_TARGET} EXPORT ${HSAKMT_TARGET}Targets
    ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR} COMPONENT asan
    LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR} COMPONENT asan )
install ( TARGETS ${HSAKMT_TARGET}
    ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR} COMPONENT devel
    LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR} COMPONENT devel )

#install ( FILES ${CMAKE_CURRENT_SOURCE_DIR}/LICENSE.md DESTINATION ${CMAKE_INSTALL_DOCDIR} COMPONENT devel )

# Install public headers
install ( DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/include/${HSAKMT_TARGET} DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
  COMPONENT devel PATTERN "linux" EXCLUDE )

# Option to build header path migration helpers.
option(INCLUDE_PATH_COMPATIBILITY "Generate backward compatible headers and include paths.  Use of these headers will warn when included." OFF)

if(INCLUDE_PATH_COMPATIBILITY)
  # To enable/disable #error in wrapper header files
  if(NOT DEFINED ROCM_HEADER_WRAPPER_WERROR)
      if(DEFINED ENV{ROCM_HEADER_WRAPPER_WERROR})
          set(ROCM_HEADER_WRAPPER_WERROR "$ENV{ROCM_HEADER_WRAPPER_WERROR}"
                  CACHE STRING "Header wrapper warnings as errors.")
      else()
          set(ROCM_HEADER_WRAPPER_WERROR "OFF" CACHE STRING "Header wrapper warnings as errors.")
      endif()
  endif()
  if(ROCM_HEADER_WRAPPER_WERROR)
      set(deprecated_error 1)
  else()
      set(deprecated_error 0)
  endif()
  include(hsakmt-backward-compat.cmake)
endif()
# Record our usage data for clients find_package calls.
install ( EXPORT ${HSAKMT_TARGET}Targets
  FILE ${HSAKMT_TARGET}Targets.cmake
  NAMESPACE ${HSAKMT_TARGET}::
  DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/${HSAKMT_TARGET}
  COMPONENT devel)

# Adds the target alias hsakmt::hsakmt to the local cmake cache.
# This isn't necessary today.  It's harmless preparation for some
# hypothetical future in which the we might be included by add_subdirectory()
# in some other project's cmake file.  It allows uniform use of find_package
# and target_link_library() without regard to whether a target is external or
# a subdirectory of the current build.
add_library( ${HSAKMT_TARGET}::${HSAKMT_TARGET} ALIAS ${HSAKMT_TARGET} )

# Create cmake configuration files
include(CMakePackageConfigHelpers)

configure_package_config_file(${HSAKMT_TARGET}-config.cmake.in
                            ${HSAKMT_TARGET}-config.cmake
                            INSTALL_DESTINATION
                            ${CMAKE_INSTALL_LIBDIR}/cmake/${HSAKMT_TARGET} )

write_basic_package_version_file(${HSAKMT_TARGET}-config-version.cmake
                 VERSION ${BUILD_VERSION_STRING}
                 COMPATIBILITY
                 AnyNewerVersion)

install(FILES
        ${CMAKE_CURRENT_BINARY_DIR}/${HSAKMT_TARGET}-config.cmake
        ${CMAKE_CURRENT_BINARY_DIR}/${HSAKMT_TARGET}-config-version.cmake
        DESTINATION
        ${CMAKE_INSTALL_LIBDIR}/cmake/${HSAKMT_TARGET}
        COMPONENT devel)

# Optionally record the package's find module in the user's package cache.
if ( NOT DEFINED EXPORT_TO_USER_PACKAGE_REGISTRY )
  set ( EXPORT_TO_USER_PACKAGE_REGISTRY "off" )
endif()
set ( EXPORT_TO_USER_PACKAGE_REGISTRY ${EXPORT_TO_USER_PACKAGE_REGISTRY}
             CACHE BOOL "Add cmake package config location to the user's cmake package registry.")
if(${EXPORT_TO_USER_PACKAGE_REGISTRY})
  # Enable writing to the registry
  set(CMAKE_EXPORT_PACKAGE_REGISTRY ON)
  # Generate a target file for the build
  export(TARGETS ${HSAKMT_TARGET} NAMESPACE ${HSAKMT_TARGET}:: FILE ${HSAKMT_TARGET}Targets.cmake)
  # Record the package in the user's cache.
  export(PACKAGE ${HSAKMT_TARGET})
endif()

# CPACK_PACKAGING_INSTALL_PREFIX is needed in libhsakmt.pc.in
# TODO: Add support for relocatable packages.
configure_file ( libhsakmt.pc.in libhsakmt.pc @ONLY )

install ( FILES ${CMAKE_CURRENT_BINARY_DIR}/libhsakmt.pc DESTINATION ${CMAKE_INSTALL_LIBDIR}/pkgconfig COMPONENT devel)

###########################
# Packaging directives
###########################
# Use component packaging
set(CPACK_COMPONENTS_GROUPING IGNORE)
set(CPACK_DEB_COMPONENT_INSTALL ON)
set(CPACK_RPM_COMPONENT_INSTALL ON)
set(CPACK_PACKAGE_VENDOR "Advanced Micro Devices, Inc.")
set(CPACK_PACKAGE_VERSION_MAJOR ${VERSION_MAJOR})
set(CPACK_PACKAGE_VERSION_MINOR ${VERSION_MINOR})
set(CPACK_PACKAGE_VERSION_PATCH ${VERSION_PATCH})
set(CPACK_PACKAGE_CONTACT "AMD GFX mailing list <amd-gfx@lists.freedesktop.org>")
set(CPACK_RESOURCE_FILE_LICENSE "${CMAKE_CURRENT_SOURCE_DIR}/LICENSE.md")
set(CPACK_COMPONENT_DESCRIPTION "HSAKMT development package.\n This package includes the user-mode API interfaces\nused to interact with the ROCk driver.\n This package contains the headers, pkgonfig and\n cmake files for the ROCT package.")
set ( ENABLE_LDCONFIG ON CACHE BOOL "Set library links and caches using ldconfig.")

# Install License file
install ( FILES ${CPACK_RESOURCE_FILE_LICENSE} DESTINATION ${CMAKE_INSTALL_DOCDIR} COMPONENT devel)
if ( ENABLE_ASAN_PACKAGING )
  install ( FILES ${CPACK_RESOURCE_FILE_LICENSE} DESTINATION ${CMAKE_INSTALL_DOCDIR}-asan COMPONENT asan)
endif()

# Make proper version for appending
# Default Value is 99999, setting it first
set(ROCM_VERSION_FOR_PACKAGE "99999")
if(DEFINED ENV{ROCM_LIBPATCH_VERSION})
  set(ROCM_VERSION_FOR_PACKAGE $ENV{ROCM_LIBPATCH_VERSION})
endif()

# Prepare final version for the CPACK use
set(PACKAGE_VERSION_STR "${CPACK_PACKAGE_VERSION_MAJOR}.${CPACK_PACKAGE_VERSION_MINOR}.${CPACK_PACKAGE_VERSION_PATCH}.${ROCM_VERSION_FOR_PACKAGE}")
set(CPACK_PACKAGE_VERSION "${PACKAGE_VERSION_STR}")

# Debian package specific variables
set(CPACK_DEBIAN_PACKAGE_HOMEPAGE "https://github.com/ROCm/ROCT-Thunk-Interface")
set(CPACK_DEBIAN_PACKAGE_NAME "hsakmt-roct-dev")
if (DEFINED ENV{CPACK_DEBIAN_PACKAGE_RELEASE})
   set(CPACK_DEBIAN_PACKAGE_RELEASE $ENV{CPACK_DEBIAN_PACKAGE_RELEASE})
else()
   set(CPACK_DEBIAN_PACKAGE_RELEASE "local")
endif()
## Process the Debian install/remove scripts to update the CPACK variables
configure_file ( ${CMAKE_CURRENT_SOURCE_DIR}/DEBIAN/postinst.in DEBIAN/postinst @ONLY )
configure_file ( ${CMAKE_CURRENT_SOURCE_DIR}/DEBIAN/prerm.in DEBIAN/prerm @ONLY )
set(CPACK_DEBIAN_PACKAGE_CONTROL_EXTRA "DEBIAN/postinst;DEBIAN/prerm")
# Needed since some packages still say they need hsakmt-roct
set(CPACK_DEBIAN_PACKAGE_REPLACES "hsakmt-roct")
set(CPACK_DEBIAN_PACKAGE_PROVIDES "hsakmt-roct")
set(CPACK_RPM_PACKAGE_OBSOLETES "hsakmt-roct")
set(CPACK_RPM_PACKAGE_PROVIDES "hsakmt-roct")

# Debian devel component specific flags
set(CPACK_DEBIAN_DEVEL_PACKAGE_NAME "hsakmt-roct-dev")
set(CPACK_DEBIAN_DEVEL_PACKAGE_REPLACES "hsakmt-roct")
set(CPACK_DEBIAN_DEVEL_PACKAGE_PROVIDES "hsakmt-roct")
set(CPACK_RPM_DEVEL_PACKAGE_PROVIDES "hsakmt-roct")

# Debian package specific variable for ASAN
set(CPACK_DEBIAN_ASAN_PACKAGE_NAME "hsakmt-roct-asan" )

## Process the Rpm install/remove scripts to update the CPACK variables
configure_file ( "${CMAKE_CURRENT_SOURCE_DIR}/RPM/post.in" RPM/post @ONLY )
configure_file ( "${CMAKE_CURRENT_SOURCE_DIR}/RPM/postun.in" RPM/postun @ONLY )
## RPM package specific variables
set ( CPACK_RPM_PRE_INSTALL_SCRIPT_FILE "${CMAKE_CURRENT_BINARY_DIR}/RPM/post" )
set ( CPACK_RPM_POST_UNINSTALL_SCRIPT_FILE "${CMAKE_CURRENT_BINARY_DIR}/RPM/postun" )

if(DEFINED CPACK_PACKAGING_INSTALL_PREFIX)
  set(CPACK_RPM_EXCLUDE_FROM_AUTO_FILELIST_ADDITION "${CPACK_PACKAGING_INSTALL_PREFIX} ${CPACK_PACKAGING_INSTALL_PREFIX}/${CMAKE_INSTALL_BINDIR}")
endif()
set(CPACK_RPM_PACKAGE_NAME "hsakmt-roct-devel")
set(CPACK_RPM_PACKAGE_LICENSE "MIT")

# RPM package specific variable for ASAN
set(CPACK_RPM_ASAN_PACKAGE_NAME "hsakmt-roct-asan" )

# RPM package devel specific variable
set(CPACK_RPM_DEVEL_PACKAGE_NAME "hsakmt-roct-devel")
# Since we changed the package name to match RPM specs, take care of older builds that had -dev installed
# Also cover the fact that this now replaces the old binary package hsakmt-roct
set(CPACK_RPM_PACKAGE_OBSOLETES "hsakmt-roct-dev hsakmt-roct")
set(CPACK_RPM_DEVEL_PACKAGE_OBSOLETES "hsakmt-roct-dev hsakmt-roct")

if(DEFINED ENV{CPACK_RPM_PACKAGE_RELEASE})
  set(CPACK_RPM_PACKAGE_RELEASE $ENV{CPACK_RPM_PACKAGE_RELEASE})
else()
  set(CPACK_RPM_PACKAGE_RELEASE "local")
endif()
## Specify a RPM SPEC template
# This template was generated by using "CPACK_RPM_GENERATE_USER_BINARY_SPECFILE_TEMPLATE"
# but I've modified it to support CPACK_RPM_PACKAGE_RECOMMENDS for EL8+ and SLE
set ( CPACK_RPM_USER_BINARY_SPECFILE "${CMAKE_CURRENT_SOURCE_DIR}/RPM/${CPACK_RPM_PACKAGE_NAME}.spec.in" )

# 'dist' breaks manual builds on debian systems due to empty Provides
execute_process( COMMAND rpm --eval %{?dist}
                 RESULT_VARIABLE PROC_RESULT
                 OUTPUT_VARIABLE EVAL_RESULT
                 OUTPUT_STRIP_TRAILING_WHITESPACE )
message("RESULT_VARIABLE ${PROC_RESULT} OUTPUT_VARIABLE: ${EVAL_RESULT}")

if ( PROC_RESULT EQUAL "0" AND NOT EVAL_RESULT STREQUAL "" )
  string ( APPEND CPACK_RPM_PACKAGE_RELEASE "%{?dist}" )
endif()

# Setting devel package dependendent version
set(CPACK_DEBIAN_PACKAGE_DEPENDS "libdrm-amdgpu-dev | libdrm-dev, rocm-core")
set(CPACK_DEBIAN_PACKAGE_RECOMMENDS "libdrm-amdgpu-dev")
set(CPACK_DEBIAN_PACKAGE_REPLACES "hsakmt-roct")
set(CPACK_DEBIAN_DEVEL_PACKAGE_DEPENDS "libdrm-amdgpu-dev | libdrm-dev, rocm-core")
set(CPACK_DEBIAN_DEVEL_PACKAGE_RECOMMENDS "libdrm-amdgpu-dev")
set(CPACK_DEBIAN_DEVEL_PACKAGE_REPLACES "hsakmt-roct")

# Debian package specific variable for ASAN
set(CPACK_DEBIAN_ASAN_PACKAGE_DEPENDS "libdrm-amdgpu-dev | libdrm-dev, rocm-core-asan")
set(CPACK_DEBIAN_ASAN_PACKAGE_RECOMMENDS "libdrm-amdgpu-dev")

set(CPACK_RPM_PACKAGE_REQUIRES "rocm-core")
set(CPACK_RPM_DEVEL_PACKAGE_REQUIRES "rocm-core")
# RPM package specific variable for ASAN
set(CPACK_RPM_ASAN_PACKAGE_REQUIRES "rocm-core-asan")

# NOTE: CentOS/RHEL <8 and SUSE <10 do not have support for rpm recommends,
# so HSAKMT_REQUIRES_LIBDRM should be defined when building EL7 to avoid
# producing binaries which do not install on CentOS7 or RHEL7
if(HSAKMT_REQUIRES_LIBDRM)
  string ( APPEND CPACK_RPM_PACKAGE_REQUIRES ", libdrm-amdgpu-devel" )
  string ( APPEND CPACK_RPM_DEVEL_PACKAGE_REQUIRES ", libdrm-amdgpu-devel" )
  string ( APPEND CPACK_RPM_ASAN_PACKAGE_REQUIRES ", libdrm-amdgpu-devel" )
else()
  string ( APPEND CPACK_RPM_PACKAGE_REQUIRES ", (libdrm-amdgpu-devel or libdrm-devel)" )
  string ( APPEND CPACK_RPM_DEVEL_PACKAGE_REQUIRES ", (libdrm-amdgpu-devel or libdrm-devel)" )
  string ( APPEND CPACK_RPM_ASAN_PACKAGE_REQUIRES ", (libdrm-amdgpu-devel or libdrm-devel)" )
  set(CPACK_RPM_PACKAGE_RECOMMENDS "libdrm-amdgpu-devel" )
  set(CPACK_RPM_DEVEL_PACKAGE_RECOMMENDS "libdrm-amdgpu-devel" )
  set(CPACK_RPM_ASAN_PACKAGE_RECOMMENDS "libdrm-amdgpu-devel" )
endif()

# Set the names now using CPACK utility
set(CPACK_DEBIAN_FILE_NAME "DEB-DEFAULT")
set(CPACK_RPM_FILE_NAME "RPM-DEFAULT")

# Remove dependency on rocm-core if -DROCM_DEP_ROCMCORE=ON not given to cmake
if(NOT ROCM_DEP_ROCMCORE)
    string(REGEX REPLACE ",? ?rocm-core" "" CPACK_RPM_PACKAGE_REQUIRES ${CPACK_RPM_PACKAGE_REQUIRES})
    string(REGEX REPLACE ",? ?rocm-core" "" CPACK_DEBIAN_PACKAGE_DEPENDS ${CPACK_DEBIAN_PACKAGE_DEPENDS})
    string(REGEX REPLACE ",? ?rocm-core" "" CPACK_RPM_DEVEL_PACKAGE_REQUIRES ${CPACK_RPM_DEVEL_PACKAGE_REQUIRES})
    string(REGEX REPLACE ",? ?rocm-core" "" CPACK_DEBIAN_DEVEL_PACKAGE_DEPENDS ${CPACK_DEBIAN_DEVEL_PACKAGE_DEPENDS})
    string(REGEX REPLACE ",? ?rocm-core-asan" "" CPACK_RPM_ASAN_PACKAGE_REQUIRES ${CPACK_RPM_ASAN_PACKAGE_REQUIRES})
    string(REGEX REPLACE ",? ?rocm-core-asan" "" CPACK_DEBIAN_ASAN_PACKAGE_DEPENDS ${CPACK_DEBIAN_ASAN_PACKAGE_DEPENDS})
endif()

include(CPack)

cpack_add_component(asan
  DISPLAY_NAME "ASAN"
  DESCRIPTION "ASAN libraries for the LIBHSAKMT")
