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

set(CMAKE_VERBOSE_MAKEFILE ON)

set ( HSAKMT "hsakmt" )
set ( HSAKMT_PACKAGE "hsakmt-roct" )
set ( HSAKMT_COMPONENT "lib${HSAKMT}" )
set ( HSAKMT_TARGET "${HSAKMT}" )
set(HSAKMT_STATIC_DRM_TARGET "${HSAKMT_TARGET}-staticdrm")

project ( ${HSAKMT_TARGET} VERSION 1.9.0)

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

list( PREPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake_modules" )

## Include common cmake modules
include ( utils )
include ( GNUInstallDirs )

## Setup the package version.
get_version ( "1.0.0" )

set ( BUILD_VERSION_MAJOR ${VERSION_MAJOR} )
set ( BUILD_VERSION_MINOR ${VERSION_MINOR} )
set ( BUILD_VERSION_PATCH ${VERSION_PATCH} )

set ( LIB_VERSION_MAJOR 1)
set ( LIB_VERSION_MINOR 0)
if (${ROCM_PATCH_VERSION})
    set ( LIB_VERSION_PATCH ${ROCM_PATCH_VERSION} )
else ()
    set ( LIB_VERSION_PATCH 6)
endif ()
set ( LIB_VERSION_STRING "${LIB_VERSION_MAJOR}.${LIB_VERSION_MINOR}.${LIB_VERSION_PATCH}" )

if ( DEFINED VERSION_BUILD AND NOT ${VERSION_BUILD} STREQUAL "" )
    message ( "VERSION BUILD DEFINED ${VERSION_BUILD}" )
    set ( BUILD_VERSION_PATCH "${BUILD_VERSION_PATCH}-${VERSION_BUILD}" )
endif ()
set ( BUILD_VERSION_STRING "${BUILD_VERSION_MAJOR}.${BUILD_VERSION_MINOR}.${BUILD_VERSION_PATCH}" )

## Compiler flags
set (HSAKMT_C_FLAGS -fPIC -W -Wall -Wextra -Wno-unused-parameter -Wformat-security -Wswitch-default -Wundef -Wshadow -Wpointer-arith -Wbad-function-cast -Wcast-qual -Wstrict-prototypes -Wmissing-prototypes -Wmissing-declarations -Wredundant-decls -Wunreachable-code -std=gnu99 -fvisibility=hidden)
if ( CMAKE_COMPILER_IS_GNUCC )
    set ( HSAKMT_C_FLAGS "${HSAKMT_C_FLAGS}" -Wlogical-op)
endif ()
if ( ${HSAKMT_WERROR} )
    set ( HSAKMT_C_FLAGS "${HSAKMT_C_FLAGS}" -Werror )
endif ()
if ( "${CMAKE_BUILD_TYPE}" STREQUAL Release )
    set ( HSAKMT_C_FLAGS "${HSAKMT_C_FLAGS}" -O2 )
else ()
    set ( HSAKMT_C_FLAGS "${HSAKMT_C_FLAGS}" -g )
endif ()

set ( HSAKMT_LINKER_SCRIPT "${CMAKE_CURRENT_SOURCE_DIR}/src/libhsakmt.ver" )

## Linker Flags
## Add --enable-new-dtags to generate DT_RUNPATH
set (HSAKMT_LINK_FLAGS "${HSAKMT_LINK_FLAGS} -Wl,--enable-new-dtags -Wl,--version-script=${HSAKMT_LINKER_SCRIPT} -Wl,-soname=${HSAKMT_COMPONENT}.so.${LIB_VERSION_MAJOR} -Wl,-z,nodelete")

## Address Sanitize Flag
if ( ${ADDRESS_SANITIZER} )
    set ( HSAKMT_C_FLAGS "${HSAKMT_C_FLAGS}" -fsanitize=address )
    set ( HSAKMT_LINK_FLAGS "${HSAKMT_LINK_FLAGS} -fsanitize=address" )
    if ( BUILD_SHARED_LIBS )
        set ( HSAKMT_LINK_FLAGS "${HSAKMT_LINK_FLAGS} -shared-libsan" )
    else ()
        set ( HSAKMT_LINK_FLAGS "${HSAKMT_LINK_FLAGS} -static-libsan" )
    endif ()
else ()
    if ( CMAKE_COMPILER_IS_GNUCC )
        set ( HSAKMT_LINK_FLAGS "${HSAKMT_LINK_FLAGS} -Wl,-no-undefined" )
    else ()
        set ( HSAKMT_LINK_FLAGS "${HSAKMT_LINK_FLAGS} -Wl,-undefined,error" )
    endif ()
endif ()

## Source files
set ( HSAKMT_SRC "src/debug.c"
                 "src/events.c"
                 "src/fmm.c"
                 "src/globals.c"
                 "src/hsakmtmodel.c"
                 "src/libhsakmt.c"
                 "src/memory.c"
                 "src/openclose.c"
                 "src/perfctr.c"
                 "src/pmc_table.c"
                 "src/queues.c"
                 "src/time.c"
                 "src/topology.c"
                 "src/rbtree.c"
                 "src/spm.c"
                 "src/version.c"
                 "src/svm.c"
                 "src/pc_sampling.c")

## Declare the library target name
add_library (${HSAKMT_TARGET} STATIC "")

## Add sources
target_sources ( ${HSAKMT_TARGET} PRIVATE ${HSAKMT_SRC} )

## Add headers.  The public headers need to point at their location in both build and install
## directory layouts.  This declaration allows publishing library use data to downstream clients.
target_include_directories( ${HSAKMT_TARGET}
  PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>
  PRIVATE
  ${CMAKE_CURRENT_SOURCE_DIR}/src )

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
find_package(NUMA)
if(NUMA_FOUND)
  set(NUMA "${NUMA_LIBRARIES}")
else()
  find_library(NUMA NAMES numa REQUIRED)
endif()
message(STATUS "LIBC: " ${LIBC})
message(STATUS "NUMA: " ${NUMA})

## If environment variable DRM_DIR is set, the script
## will pick up the corresponding libraries from that path.
if(DRM_DIR)
  list (PREPEND CMAKE_PREFIX_PATH "${DRM_DIR}")
endif()

# The module name passed to pkg_check_modules() is determined by the
# name of file *.pc
pkg_check_modules(DRM REQUIRED IMPORTED_TARGET libdrm)
pkg_check_modules(DRM_AMDGPU REQUIRED IMPORTED_TARGET libdrm_amdgpu)
include_directories(${DRM_AMDGPU_INCLUDE_DIRS})
include_directories(${DRM_INCLUDE_DIRS})

target_link_libraries ( ${HSAKMT_TARGET}
  PRIVATE ${DRM_LDFLAGS} ${DRM_AMDGPU_LDFLAGS} pthread rt ${LIBC} ${NUMA} ${CMAKE_DL_LIBS}
)

target_compile_options(${HSAKMT_TARGET} PRIVATE ${DRM_CFLAGS} ${HSAKMT_C_FLAGS})

include(CheckFunctionExists)
set(CMAKE_REQUIRED_DEFINITIONS -D__USE_GNU=1)
set(CMAKE_REQUIRED_INCLUDES sys/mman.h)
check_function_exists(memfd_create HAVE_MEMFD_CREATE)
if(HAVE_MEMFD_CREATE)
  target_compile_definitions(${HSAKMT_TARGET} PRIVATE -DHAVE_MEMFD_CREATE=1)
endif()

## Define default paths and packages.
if( CMAKE_INSTALL_PREFIX_INITIALIZED_TO_DEFAULT )
  set ( CMAKE_INSTALL_PREFIX "/opt/rocm" )
endif()
set ( CMAKE_INSTALL_PREFIX ${CMAKE_INSTALL_PREFIX} CACHE STRING "Default installation directory." FORCE )

# Installs binaries and exports the library usage data to ${HSAKMT_TARGET}Targets
install ( TARGETS ${HSAKMT_TARGET} EXPORT ${HSAKMT_TARGET}Targets
    ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR} COMPONENT asan
    LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR} COMPONENT asan )
install ( TARGETS ${HSAKMT_TARGET}
    ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR} COMPONENT binary
    LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR} COMPONENT binary )

# Install public headers
install ( DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/include/${HSAKMT_TARGET} DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
  COMPONENT dev PATTERN "linux" EXCLUDE PATTERN "*virtio*" EXCLUDE)

# Record our usage data for clients find_package calls.
install ( EXPORT ${HSAKMT_TARGET}Targets
  FILE ${HSAKMT_TARGET}Targets.cmake
  NAMESPACE ${HSAKMT_TARGET}::
  DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/${HSAKMT_TARGET}
  COMPONENT dev)

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
        COMPONENT dev)

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

install ( FILES ${CMAKE_CURRENT_BINARY_DIR}/libhsakmt.pc DESTINATION ${CMAKE_INSTALL_LIBDIR}/pkgconfig COMPONENT dev)

if ( NOT BUILD_SHARED_LIBS)
  ## Create separate target file for static builds
  ## In static builds, libdrm and libdrm_amdgpu need to be linked statically
  add_library (${HSAKMT_STATIC_DRM_TARGET}  STATIC "")
  target_sources (${HSAKMT_STATIC_DRM_TARGET} PRIVATE ${HSAKMT_SRC})

  target_include_directories( ${HSAKMT_STATIC_DRM_TARGET}
    PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>
    PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/src )

  ## Set the VERSION and SOVERSION values
  set_property(TARGET ${HSAKMT_STATIC_DRM_TARGET} PROPERTY LINK_FLAGS ${HSAKMT_LINK_FLAGS}
              PROPERTY VERSION "${LIB_VERSION_STRING}"
              PROPERTY SOVERSION "${LIB_VERSION_MAJOR}" )

  #Additional search path for static libraries
  if(${DISTRO_ID} MATCHES "ubuntu")
      set(AMDGPU_STATIC_LIB_PATHS "-L/opt/amdgpu/lib/x86_64-linux-gnu")
  else()
      set(AMDGPU_STATIC_LIB_PATHS "-L/opt/amdgpu/lib64" "-L/opt/amdgpu/lib")
  endif()
  # Link drm_amdgpu and drm library statically
  target_link_libraries ( ${HSAKMT_STATIC_DRM_TARGET}
    PRIVATE pthread rt c numa ${CMAKE_DL_LIBS}
    INTERFACE -Wl,-Bstatic ${AMDGPU_STATIC_LIB_PATHS} ${DRM_AMDGPU_LDFLAGS} ${DRM_LDFLAGS} -Wl,-Bdynamic
  )
  target_compile_options(${HSAKMT_STATIC_DRM_TARGET} PRIVATE ${DRM_CFLAGS} ${HSAKMT_C_FLAGS})

  install ( TARGETS ${HSAKMT_STATIC_DRM_TARGET} EXPORT ${HSAKMT_STATIC_DRM_TARGET}Targets
            ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR} COMPONENT binary
            LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR} COMPONENT binary)
  install ( EXPORT ${HSAKMT_STATIC_DRM_TARGET}Targets
    FILE ${HSAKMT_STATIC_DRM_TARGET}Targets.cmake
    NAMESPACE ${HSAKMT_STATIC_DRM_TARGET}::
    DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/${HSAKMT_TARGET}
    COMPONENT dev)

  add_library( ${HSAKMT_STATIC_DRM_TARGET}::${HSAKMT_STATIC_DRM_TARGET} ALIAS ${HSAKMT_STATIC_DRM_TARGET} )
endif()

###########################
# Packaging directives
###########################
# Use component packaging
set ( ENABLE_LDCONFIG ON CACHE BOOL "Set library links and caches using ldconfig.")
