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

cmake_minimum_required ( VERSION 3.15 )

set(CMAKE_VERBOSE_MAKEFILE ON)

set ( HSAKMT "hsakmt" )
set ( HSAKMT_PACKAGE "hsakmt-roct" )
set ( HSAKMT_COMPONENT "lib${HSAKMT}" )
set ( HSAKMT_TARGET "${HSAKMT}" )
set(HSAKMT_STATIC_DRM_TARGET "${HSAKMT_TARGET}-staticdrm")

project ( ${HSAKMT_TARGET} VERSION 1.9.0)

# Build thunk as static by default
if ( NOT DEFINED BUILD_SHARED_LIBS )
  set ( BUILD_SHARED_LIBS "off" )
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

list( PREPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake_modules" )

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
set (HSAKMT_CXX_FLAGS -fPIC)

if ( CMAKE_COMPILER_IS_GNUCC )
    set ( HSAKMT_CXX_FLAGS "${HSAKMT_CXX_FLAGS}" -Wlogical-op)
endif ()
if ( ${HSAKMT_WERROR} )
    set ( HSAKMT_CXX_FLAGS "${HSAKMT_CXX_FLAGS}" -Werror )
endif ()
if ( "${CMAKE_BUILD_TYPE}" STREQUAL Release )
    set ( HSAKMT_CXX_FLAGS "${HSAKMT_CXX_FLAGS}" -O2 )
else ()
    set ( HSAKMT_CXX_FLAGS "${HSAKMT_CXX_FLAGS}" -g )
endif ()

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
else ()
    if ( CMAKE_COMPILER_IS_GNUCC )
        set ( HSAKMT_LINK_FLAGS "${HSAKMT_LINK_FLAGS} -Wl,-no-undefined" )
    else ()
        set ( HSAKMT_LINK_FLAGS "${HSAKMT_LINK_FLAGS} -Wl,-undefined,error" )
    endif ()
endif ()

## Source files
set ( HSAKMT_SRC "src/debug.cpp"
                 "src/events.cpp"
                 "src/globals.cpp"
                 "src/memory.cpp"
                 "src/libdrm.cpp"
                 "src/hsa.cpp"
                 "src/openclose.cpp"
                 "src/perfctr.cpp"
                 "src/queues.cpp"
                 "src/time.cpp"
                 "src/topology.cpp"
                 "src/spm.cpp"
                 "src/version.cpp"
                 "src/svm.cpp"
                 "src/pc_sampling.cpp"
                 "src/hsakmtmodel.cpp"
                 "src/wddm/device.cpp"
                 "src/wddm/gpu_memory.cpp"
                 "src/wddm/va_mgr.cpp"
                 "src/wddm/queue.cpp"
                 "src/wddm/cmd_util.cpp" )

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
  ${WIN_SDK}
  ${CMAKE_CURRENT_SOURCE_DIR}/src
  ${CMAKE_CURRENT_SOURCE_DIR}/../runtime )

add_compile_definitions(LINUX __AMD64__ LITTLEENDIAN_CPU HSA_LARGE_MODEL)
target_link_directories(${HSAKMT_TARGET} PRIVATE ${DXCORE_LIB_PATH})
target_link_libraries(${HSAKMT_TARGET} PRIVATE dxcore)

target_link_directories(${HSAKMT_TARGET} PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/src/thunk_proxy)
target_link_libraries(${HSAKMT_TARGET} PRIVATE thunk_proxy)

if (FALSE)
target_link_directories(${HSAKMT_TARGET} PRIVATE /opt/rocm/lib)
target_link_libraries(${HSAKMT_TARGET} PRIVATE hsa-runtime64)
endif ()

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
message(STATUS "LIBC:" ${LIBC})

target_link_libraries ( ${HSAKMT_TARGET}
  PRIVATE pthread rt c ${CMAKE_DL_LIBS}
)

target_compile_options(${HSAKMT_TARGET} PRIVATE ${HSAKMT_CXX_FLAGS})

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
  COMPONENT dev PATTERN "linux" EXCLUDE )

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
  add_library (${HSAKMT_STATIC_DRM_TARGET}  STATIC "")
  target_sources (${HSAKMT_STATIC_DRM_TARGET} PRIVATE ${HSAKMT_SRC})

  target_include_directories( ${HSAKMT_STATIC_DRM_TARGET}
    PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>
    PRIVATE
    ${DK_HEADER}/ms_sdk/include/shared
    ${DK_HEADER}/inc/shared
    ${DK_HEADER}/inc/asic_reg
    ${CMAKE_CURRENT_SOURCE_DIR}/src
    ${CMAKE_CURRENT_SOURCE_DIR}/../runtime )

    ## Set the VERSION and SOVERSION values
  set_property(TARGET ${HSAKMT_STATIC_DRM_TARGET} PROPERTY LINK_FLAGS ${HSAKMT_LINK_FLAGS}
              PROPERTY VERSION "${LIB_VERSION_STRING}"
              PROPERTY SOVERSION "${LIB_VERSION_MAJOR}" )

  target_link_libraries ( ${HSAKMT_STATIC_DRM_TARGET}
    PRIVATE pthread rt c ${CMAKE_DL_LIBS}
    INTERFACE -Wl,-Bdynamic
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
