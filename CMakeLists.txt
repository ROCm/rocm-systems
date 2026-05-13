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

set ( ROCDXG "rocdxg" )
set ( ROCDXG_PACKAGE "rocdxg-roct" )
set ( ROCDXG_COMPONENT "lib${ROCDXG}" )
set ( ROCDXG_TARGET "${ROCDXG}" )

# Single source of truth for the package version. Callers may override by
# passing -DPACKAGE_VERSION=x.y.z; otherwise the value is read from the
# shared VERSION file at the repository root.
if ( NOT DEFINED PACKAGE_VERSION OR "${PACKAGE_VERSION}" STREQUAL "" )
    file ( STRINGS "${CMAKE_CURRENT_SOURCE_DIR}/VERSION" PACKAGE_VERSION
           LIMIT_COUNT 1 )
    string ( STRIP "${PACKAGE_VERSION}" PACKAGE_VERSION )
endif ()
set ( ROCDXG_VERSION "${PACKAGE_VERSION}" )

# Optional pre-release suffix (e.g. -DROCDXG_VERSION_PRERELEASE=rc1 produces 1.1.2~rc1).
# Leave empty for a normal release build (1.1.2).
set ( ROCDXG_VERSION_PRERELEASE "" CACHE STRING "Pre-release suffix appended to package version (e.g. rc1). Empty for release." )

project ( ${ROCDXG_TARGET} VERSION ${ROCDXG_VERSION} )
# Project/version initialized; expose version to code via target defs below

# Optionally, build ROCDXG with ccache.
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

## Define default install prefix before GNUInstallDirs computes paths.
if( CMAKE_INSTALL_PREFIX_INITIALIZED_TO_DEFAULT )
  set ( CMAKE_INSTALL_PREFIX "/opt/rocm" )
endif()
set ( CMAKE_INSTALL_PREFIX ${CMAKE_INSTALL_PREFIX} CACHE STRING "Default installation directory." FORCE )

## Include common cmake modules
include ( utils )
include ( GNUInstallDirs )

## Setup the package version.
get_version ( "${ROCDXG_VERSION}" )

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
set (ROCDXG_CXX_FLAGS -fPIC -include ${CMAKE_CURRENT_SOURCE_DIR}/src/librocdxg.h)

if ( CMAKE_COMPILER_IS_GNUCC )
    set ( ROCDXG_CXX_FLAGS "${ROCDXG_CXX_FLAGS}" -Wlogical-op)
endif ()
if ( ${ROCDXG_WERROR} )
    set ( ROCDXG_CXX_FLAGS "${ROCDXG_CXX_FLAGS}" -Werror )
endif ()
if ( "${CMAKE_BUILD_TYPE}" STREQUAL Release )
    set ( ROCDXG_CXX_FLAGS "${ROCDXG_CXX_FLAGS}" -O2 )
else ()
    set ( ROCDXG_CXX_FLAGS "${ROCDXG_CXX_FLAGS}" -g )
endif ()

set ( ROCDXG_LINKER_SCRIPT "${CMAKE_CURRENT_SOURCE_DIR}/src/librocdxg.ver" )

## Linker Flags
## Add --enable-new-dtags to generate DT_RUNPATH
set (ROCDXG_LINK_FLAGS "${ROCDXG_LINK_FLAGS} -Wl,--enable-new-dtags -Wl,--version-script=${ROCDXG_LINKER_SCRIPT} -Wl,-soname=${ROCDXG_COMPONENT}.so.${LIB_VERSION_MAJOR} -Wl,-z,nodelete")

## Linker undefined symbol handling
if ( CMAKE_COMPILER_IS_GNUCC )
    set ( ROCDXG_LINK_FLAGS "${ROCDXG_LINK_FLAGS} -Wl,-no-undefined" )
else ()
    set ( ROCDXG_LINK_FLAGS "${ROCDXG_LINK_FLAGS} -Wl,-undefined,error" )
endif ()

## Source files
set ( ROCDXG_SRC "src/debug.cpp"
                 "src/events.cpp"
                 "src/heap_manager.cpp"
                 "src/memory.cpp"
                 "src/libdrm.cpp"
                 "src/hsa.cpp"
                 "src/openclose.cpp"
                 "src/perfctr.cpp"
                 "src/queues.cpp"
                 "src/runtime.cpp"
                 "src/time.cpp"
                 "src/topology.cpp"
                 "src/spm.cpp"
                 "src/version.cpp"
                 "src/svm.cpp"
                 "src/pc_sampling.cpp"
                 "src/hsakmtmodel.cpp"
                 "src/ais.cpp"
                 "src/wddm/device.cpp"
                 "src/wddm/gpu_memory.cpp"
                 "src/wddm/va_mgr.cpp"
                 "src/wddm/queue.cpp"
                 "src/wddm/cmd_util.cpp" )

## Declare the library target name
add_library (${ROCDXG_TARGET} SHARED "")

## Add sources
target_sources ( ${ROCDXG_TARGET} PRIVATE ${ROCDXG_SRC} )

## Add headers.  The public headers need to point at their location in both build and install
## directory layouts.  This declaration allows publishing library use data to downstream clients.
target_include_directories( ${ROCDXG_TARGET}
  PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>
  PRIVATE
  ${CMAKE_CURRENT_SOURCE_DIR}/src )

target_include_directories( ${ROCDXG_TARGET} SYSTEM PRIVATE ${WIN_SDK} )

add_compile_definitions(LINUX __AMD64__ LITTLEENDIAN_CPU HSA_LARGE_MODEL)

# Ensure version macro is defined for this target
target_compile_definitions(${ROCDXG_TARGET} PRIVATE ROCDXG_VERSION="${ROCDXG_VERSION}")

add_subdirectory(shared)
target_link_libraries(${ROCDXG_TARGET} PRIVATE thunk_proxy dxg_shared)

set_property(TARGET ${ROCDXG_TARGET} PROPERTY LINK_FLAGS ${ROCDXG_LINK_FLAGS})

## Ensure runtime can locate ROCm libs from /opt/rocm/lib via RPATH
set_target_properties(${ROCDXG_TARGET} PROPERTIES
  BUILD_RPATH "/opt/rocm/lib"
  INSTALL_RPATH "/opt/rocm/lib"
  INSTALL_RPATH_USE_LINK_PATH TRUE)

## Set the VERSION and SOVERSION values
set_property ( TARGET ${ROCDXG_TARGET} PROPERTY VERSION "${LIB_VERSION_STRING}" )
set_property ( TARGET ${ROCDXG_TARGET} PROPERTY SOVERSION "${LIB_VERSION_MAJOR}" )

find_package(PkgConfig)
# get OS-info for OS-specific build dependencies
get_os_info()
# Check for libraries required for building
find_library(LIBC NAMES c REQUIRED)
message(STATUS "LIBC:" ${LIBC})

target_link_libraries ( ${ROCDXG_TARGET}
  PRIVATE pthread rt c ${CMAKE_DL_LIBS}
)

target_compile_options(${ROCDXG_TARGET} PRIVATE ${ROCDXG_CXX_FLAGS})

## Define default paths and packages.
set ( CPACK_PACKAGING_INSTALL_PREFIX "${CMAKE_INSTALL_PREFIX}"  CACHE STRING "Default packaging prefix." )
set ( CPACK_GENERATOR "DEB"  CACHE STRING "Default packaging generators." )

# Installs binaries and exports the library usage data to ${ROCDXG_TARGET}Targets
install ( TARGETS ${ROCDXG_TARGET} EXPORT ${ROCDXG_TARGET}Targets
    ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR} COMPONENT binary
    LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR} COMPONENT binary )

# Install public headers
#install ( DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/include/${ROCDXG_TARGET} DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
#  COMPONENT dev PATTERN "*drm*" EXCLUDE )

# Record our usage data for clients find_package calls.
install ( EXPORT ${ROCDXG_TARGET}Targets
  FILE ${ROCDXG_TARGET}Targets.cmake
  NAMESPACE ${ROCDXG_TARGET}::
  DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/${ROCDXG_TARGET}
  COMPONENT dev )

# Adds the target alias rocdxg::rocdxg to the local cmake cache.
# This isn't necessary today.  It's harmless preparation for some
# hypothetical future in which the we might be included by add_subdirectory()
# in some other project's cmake file.  It allows uniform use of find_package
# and target_link_library() without regard to whether a target is external or
# a subdirectory of the current build.
add_library( ${ROCDXG_TARGET}::${ROCDXG_TARGET} ALIAS ${ROCDXG_TARGET} )

# Create cmake configuration files
include(CMakePackageConfigHelpers)

configure_package_config_file(${ROCDXG_TARGET}-config.cmake.in
                            ${ROCDXG_TARGET}-config.cmake
                            INSTALL_DESTINATION
                            ${CMAKE_INSTALL_LIBDIR}/cmake/${ROCDXG_TARGET} )

write_basic_package_version_file(${ROCDXG_TARGET}-config-version.cmake
                 VERSION ${BUILD_VERSION_STRING}
                 COMPATIBILITY
                 AnyNewerVersion)

install(FILES
        ${CMAKE_CURRENT_BINARY_DIR}/${ROCDXG_TARGET}-config.cmake
        ${CMAKE_CURRENT_BINARY_DIR}/${ROCDXG_TARGET}-config-version.cmake
        DESTINATION
        ${CMAKE_INSTALL_LIBDIR}/cmake/${ROCDXG_TARGET}
        COMPONENT dev )

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
  export(TARGETS ${ROCDXG_TARGET} NAMESPACE ${ROCDXG_TARGET}:: FILE ${ROCDXG_TARGET}Targets.cmake)
  # Record the package in the user's cache.
  export(PACKAGE ${ROCDXG_TARGET})
endif()

# Since librocdxg.pc and libhsakmt.pc are installed to the same pkgconfig directory,
# we can directly use libhsakmt's header file path in the includedir.
# This allows librocdxg to reference the same header files as libhsakmt without
# duplicating header installation.
configure_file ( librocdxg.pc.in librocdxg.pc @ONLY )

install ( FILES ${CMAKE_CURRENT_BINARY_DIR}/librocdxg.pc DESTINATION ${CMAKE_INSTALL_LIBDIR}/pkgconfig COMPONENT dev )

set ( ENABLE_LDCONFIG ON CACHE BOOL "Set library links and caches using ldconfig.")
if(ENABLE_LDCONFIG)
  # Match DEBIAN/postinst.in: system ld.so.conf.d, not $PREFIX/etc.
  set ( ROCDXG_LD_SO_CONF_DIR "/${CMAKE_INSTALL_SYSCONFDIR}/ld.so.conf.d" )
  install(CODE "
    file(WRITE \"\$ENV{DESTDIR}${ROCDXG_LD_SO_CONF_DIR}/x86_64-libhsakmt.conf\"
         \"${CMAKE_INSTALL_FULL_LIBDIR}\n\")
    execute_process(COMMAND ldconfig)
  " COMPONENT binary)
endif()

###########################
# Packaging directives
###########################
# Use component packaging
set(CPACK_COMPONENTS_GROUPING IGNORE)
set(CPACK_DEB_COMPONENT_INSTALL ON)
set(CPACK_PACKAGE_VENDOR "Advanced Micro Devices, Inc.")
set(CPACK_PACKAGE_VERSION_MAJOR ${VERSION_MAJOR})
set(CPACK_PACKAGE_VERSION_MINOR ${VERSION_MINOR})
set(CPACK_PACKAGE_VERSION_PATCH ${VERSION_PATCH})
set(CPACK_PACKAGE_CONTACT "AMD GFX mailing list <amd-gfx@lists.freedesktop.org>")
set(CPACK_RESOURCE_FILE_LICENSE "${CMAKE_CURRENT_SOURCE_DIR}/LICENSE.md")
set(CPACK_COMPONENT_DESCRIPTION "ROCDXG development package.\n This package includes the user-mode API interfaces\nused to interact with the ROCm driver.\n This package contains the libraries and cmake files for the ROCDXG package.")

# Install License file
install ( FILES ${CPACK_RESOURCE_FILE_LICENSE} DESTINATION ${CMAKE_INSTALL_DOCDIR} COMPONENT binary )

install(FILES ${CMAKE_CURRENT_SOURCE_DIR}/data/dids.conf
        DESTINATION ${CMAKE_INSTALL_DATADIR}/rocdxg
        COMPONENT binary)

file(WRITE "${CMAKE_CURRENT_BINARY_DIR}/DEBIAN/conffiles"
     "${CMAKE_INSTALL_FULL_DATADIR}/rocdxg/dids.conf\n")

# Prepare final version for the CPACK use
set(PACKAGE_VERSION_STR "${CPACK_PACKAGE_VERSION_MAJOR}.${CPACK_PACKAGE_VERSION_MINOR}.${CPACK_PACKAGE_VERSION_PATCH}")
if ( NOT "${ROCDXG_VERSION_PRERELEASE}" STREQUAL "" )
    string ( TOLOWER "${ROCDXG_VERSION_PRERELEASE}" _ROCDXG_VERSION_PRERELEASE_LC )
    # Use Debian's '~' separator so pre-releases sort BEFORE the final release.
    set(PACKAGE_VERSION_STR "${PACKAGE_VERSION_STR}~${_ROCDXG_VERSION_PRERELEASE_LC}")
    message(STATUS "Building pre-release version: ${PACKAGE_VERSION_STR}")
endif()
set(CPACK_PACKAGE_VERSION "${PACKAGE_VERSION_STR}")

# Debian package specific variables
set(CPACK_DEBIAN_PACKAGE_NAME "rocdxg-roct")

# Debian binary package specific variables (runtime package)
set(CPACK_DEBIAN_BINARY_PACKAGE_NAME "rocdxg-roct")
set(CPACK_DEBIAN_BINARY_PACKAGE_DESCRIPTION "ROCDXG runtime package containing libraries")

# Debian dev package specific variables
set(CPACK_DEBIAN_DEV_PACKAGE_NAME "rocdxg-roct-dev")
set(CPACK_DEBIAN_DEV_PACKAGE_DESCRIPTION "ROCDXG development package containing pkgconfig and cmake files")

## Process the Debian install/remove scripts to update the CPACK variables
configure_file ( ${CMAKE_CURRENT_SOURCE_DIR}/DEBIAN/postinst.in DEBIAN/postinst @ONLY )
configure_file ( ${CMAKE_CURRENT_SOURCE_DIR}/DEBIAN/prerm.in DEBIAN/prerm @ONLY )
set(CPACK_DEBIAN_BINARY_PACKAGE_CONTROL_EXTRA
    "${CMAKE_CURRENT_BINARY_DIR}/DEBIAN/postinst"
    "${CMAKE_CURRENT_BINARY_DIR}/DEBIAN/prerm"
    "${CMAKE_CURRENT_BINARY_DIR}/DEBIAN/conffiles")

# Setting package dependencies
set(CPACK_DEBIAN_PACKAGE_DEPENDS "rocm-core")
set(CPACK_DEBIAN_BINARY_PACKAGE_DEPENDS "rocm-core")
set(CPACK_DEBIAN_DEV_PACKAGE_DEPENDS "rocdxg-roct (= ${PACKAGE_VERSION_STR}), rocm-core")

# Set the names now using CPACK utility
set(CPACK_DEBIAN_FILE_NAME "DEB-DEFAULT")

# Remove dependency on rocm-core if -DROCM_DEP_ROCMCORE=ON not given to cmake
if(NOT ROCM_DEP_ROCMCORE)
    string(REGEX REPLACE ",? ?rocm-core" "" CPACK_DEBIAN_PACKAGE_DEPENDS ${CPACK_DEBIAN_PACKAGE_DEPENDS})
    string(REGEX REPLACE ",? ?rocm-core" "" CPACK_DEBIAN_BINARY_PACKAGE_DEPENDS ${CPACK_DEBIAN_BINARY_PACKAGE_DEPENDS})
    string(REGEX REPLACE ",? ?rocm-core" "" CPACK_DEBIAN_DEV_PACKAGE_DEPENDS ${CPACK_DEBIAN_DEV_PACKAGE_DEPENDS})
endif()

include(CPack)

# Add component descriptions
cpack_add_component(binary
  DISPLAY_NAME "Runtime"
  DESCRIPTION "ROCDXG runtime libraries")

cpack_add_component(dev
  DISPLAY_NAME "Development"
  DESCRIPTION "ROCDXG development files (pkgconfig and cmake)")
