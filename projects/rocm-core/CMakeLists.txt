################################################################################
##
## MIT License
##
## Copyright (c) 2017 - 2023 Advanced Micro Devices, Inc. All rights Reserved.
##
## Permission is hereby granted, free of charge, to any person obtaining a copy
## of this software and associated documentation files (the "Software"), to deal
## in the Software without restriction, including without limitation the rights
## to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
## copies of the Software, and to permit persons to whom the Software is
## furnished to do so, subject to the following conditions:
##
## The above copyright notice and this permission notice shall be included in all
## copies or substantial portions of the Software.
##
## THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
## IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
## FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
## AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
## LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
## OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
## SOFTWARE.
##
################################################################################

cmake_minimum_required( VERSION 3.16 )
set( CORE_TARGET "rocm-core" )

project( ${CORE_TARGET} CXX )

## Define Lib Target Name and Lib Target Install Dir to be available to c++ source
add_definitions( -DTARGET_LIBRARY_NAME=\"${CORE_TARGET}\" -DTARGET_LIBRARY_INSTALL_DIR=\"${CMAKE_INSTALL_LIBDIR}\" )

## Build Type Settings
set( CMAKE_BUILD_TYPE Release CACHE STRING "Choose the type of build, options are: None Debug Release RelWithDebInfo MinSizeRel." )

## Verbose output.
set( CMAKE_VERBOSE_MAKEFILE on )
include( GNUInstallDirs )

if(ENABLE_ASAN_PACKAGING)
  # Only libraries required for ASAN Package
  set( CPACK_COMPONENTS_ALL CORE_ASAN )
  set( COMP_TYPE "CORE_ASAN" )
  set( PKG_DESCRIPTION "ROCm AddressSanitizer Instrumented Runtime software stack" )
  set( CMAKE_INSTALL_DOCDIR "${CMAKE_INSTALL_DOCDIR}-asan" )
else()
  if(BUILD_SHARED_LIBS)
    set( CPACK_COMPONENTS_ALL CORE_RUNTIME )
    set( COMP_TYPE "CORE_RUNTIME" )
    set( PKG_DESCRIPTION "ROCm Runtime software stack" )
  else()
    set( CPACK_COMPONENTS_ALL CORE_STATIC )
    set( STATIC_COMP_TYPE "CORE_STATIC" )
    set( PKG_DESCRIPTION "ROCm Static software stack" )
  endif()
endif()

include( utils.cmake )

#Parse Version
message( STATUS "ROCM_VERSION = ${ROCM_VERSION}" )
parse_rocm_version( ${ROCM_VERSION} )

set( CPACK_PACKAGING_INSTALL_PREFIX "/opt/rocm" CACHE PATH "default cpack directory" )
set( BUILD_SHARED_LIBS ON CACHE BOOL "Build shared library (.so) or not." )
set( CPACK_GENERATOR "TGZ;DEB;RPM" CACHE STRING "package types to be produced " )
set( LICENSE_FILE "${CMAKE_CURRENT_SOURCE_DIR}/copyright"  )
set( BUILD_DIR ${CMAKE_CURRENT_BINARY_DIR} )
set( EXTENDED_PACKAGE_DESCRIPTION
     "ROCM-CORE is a package which can be used to get ROCm release version, get ROCm install path information etc.
     It is also important to note that ROCM-CORE takes the role as a base component on which all of ROCm can depend,
     to make it easy to remove all of ROCm with a package manager." CACHE STRING "Debian Package Extended Description")

## Set variables
set_variables()

set( SO_MAJOR "1")
set( SO_MINOR "0" )

if( NOT DEFINED BUILD_ID )
  set( BUILD_ID "9999")
endif()

#Add the libpatch version if available
if( DEFINED VERSION_POINT )
  set( PACKAGE_BUILD_INFO "${VERSION_MAJOR}.${VERSION_MINOR}.${VERSION_PATCH}.${VERSION_POINT}.${VERSION_COMMIT_COUNT}-${BUILD_ID}-${VERSION_HASH}" )
  message ( STATUS "PACKAGE_BUILD_INFO ${PACKAGE_BUILD_INFO}" )
else ()
  set( PACKAGE_BUILD_INFO "${VERSION_MAJOR}.${VERSION_MINOR}.${VERSION_PATCH}.${VERSION_COMMIT_COUNT}-${BUILD_ID}-${VERSION_HASH}" )
endif()

configure_file( ${CMAKE_CURRENT_SOURCE_DIR}/rocmmod.in ${BUILD_DIR}/rocmmod @ONLY )

# Setting Permission 0755 for pre/post install scripts
configure_file( ${CMAKE_CURRENT_SOURCE_DIR}/rocm-core.postinst ${BUILD_DIR}/postinst
	 FILE_PERMISSIONS OWNER_READ OWNER_WRITE OWNER_EXECUTE GROUP_READ GROUP_EXECUTE WORLD_READ WORLD_EXECUTE @ONLY )
configure_file( ${CMAKE_CURRENT_SOURCE_DIR}/rocm-core.prerm ${BUILD_DIR}/prerm
	 FILE_PERMISSIONS OWNER_READ OWNER_WRITE OWNER_EXECUTE GROUP_READ GROUP_EXECUTE WORLD_READ WORLD_EXECUTE @ONLY )

#Generate BUILD_INFO
configure_file( ${CMAKE_CURRENT_SOURCE_DIR}/rocm_version.h.in ${BUILD_DIR}/rocm_version.h @ONLY )
configure_file( ${CMAKE_CURRENT_SOURCE_DIR}/rocm_getpath.h.in ${BUILD_DIR}/rocm_getpath.h @ONLY )

#Make the rocmlib
set( SRCS rocm_version.cpp rocm_getpath.cpp)
add_library( ${CORE_TARGET} ${SRCS} )

set( CXX_FLAGS ${CXX_FLAGS} -g -fPIC -fvisibility=hidden -W -Wall -Wextra -Wno-unused-parameter -Wformat-security -Wundef -Wshadow -Wpointer-arith -Wcast-qual -Wmissing-declarations -Wredundant-decls -Wunreachable-code -std=c++11 )
set( CMAKE_SHARED_LINKER_FLAGS "${CMAKE_SHARED_LINKER_FLAGS}  -Wl,-z,nodelete -Wl,-no-undefined" )
target_include_directories( ${CORE_TARGET} PRIVATE ${BUILD_DIR} )
if(BUILD_SHARED_LIBS)
  target_link_libraries( ${CORE_TARGET} PRIVATE dl )
endif()

## Set the VERSION and SOVERSION values
set( SO_VERSION_STRING "${SO_MAJOR}.${SO_MINOR}.${ROCM_LIBPATCH_VERSION}" )
set_property( TARGET ${CORE_TARGET} PROPERTY VERSION "${SO_VERSION_STRING}" )
set_property( TARGET ${CORE_TARGET} PROPERTY SOVERSION "${SO_MAJOR}" )

#Add the libpatch version if available to the version file
if( DEFINED VERSION_POINT )
  file ( WRITE "${CMAKE_CURRENT_BINARY_DIR}/version" "${VERSION_MAJOR}.${VERSION_MINOR}.${VERSION_PATCH}.${VERSION_POINT}\n" )
else()
  file ( WRITE "${CMAKE_CURRENT_BINARY_DIR}/version" "${VERSION_MAJOR}.${VERSION_MINOR}.${VERSION_PATCH}\n" )
endif()

## Support for Static and Shared Target Packaging
if(BUILD_SHARED_LIBS)
  install ( FILES ${BUILD_DIR}/version DESTINATION .info COMPONENT CORE_RUNTIME )

  # Target Library needs to installed/packaged to same destination
  # in both ASAN and nonASAN Shared library builds
  install ( TARGETS ${CORE_TARGET} EXPORT rocmCoreTargets DESTINATION ${CMAKE_INSTALL_LIBDIR}
            COMPONENT ${COMP_TYPE} )
  install ( FILES ${BUILD_DIR}/rocm_version.h ${BUILD_DIR}/rocm_getpath.h
            DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/rocm-core
            COMPONENT CORE_RUNTIME )

  install ( FILES ${CMAKE_CURRENT_SOURCE_DIR}/runpath_to_rpath.py
            DESTINATION ${CMAKE_INSTALL_LIBEXECDIR}/rocm-core
            COMPONENT CORE_RUNTIME )
  install ( FILES ${BUILD_DIR}/rocmmod
            DESTINATION ${CMAKE_INSTALL_LIBDIR}
            COMPONENT CORE_RUNTIME )

  # License file
  install ( FILES ${LICENSE_FILE}
            DESTINATION ${CMAKE_INSTALL_DOCDIR} RENAME LICENSE.txt
            COMPONENT ${COMP_TYPE})
else()
  install ( FILES ${BUILD_DIR}/version
            DESTINATION .info
            COMPONENT ${STATIC_COMP_TYPE} )
  install ( TARGETS ${CORE_TARGET}
            EXPORT rocmCoreTargets
            DESTINATION ${CMAKE_INSTALL_LIBDIR}
            COMPONENT ${STATIC_COMP_TYPE} )

  install ( FILES ${BUILD_DIR}/rocm_version.h ${BUILD_DIR}/rocm_getpath.h
            DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/rocm-core
            COMPONENT ${STATIC_COMP_TYPE} )

  install ( FILES ${CMAKE_CURRENT_SOURCE_DIR}/runpath_to_rpath.py
            DESTINATION ${CMAKE_INSTALL_LIBEXECDIR}/rocm-core
            COMPONENT ${STATIC_COMP_TYPE} )
  install ( FILES ${BUILD_DIR}/rocmmod
            DESTINATION ${CMAKE_INSTALL_LIBDIR}
            COMPONENT ${STATIC_COMP_TYPE} )
  # License file
  install ( FILES ${LICENSE_FILE}
            DESTINATION ${CMAKE_INSTALL_DOCDIR} RENAME LICENSE.txt
            COMPONENT ${STATIC_COMP_TYPE})
endif()

## Cmake module config file configurations
set(CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake_modules/" CACHE INTERNAL "Default module path.")

## Export the package for use from the build-tree
## (this registers the build-tree with a global CMake-registry)
export(PACKAGE ${CORE_TARGET})
set(CONF_LIB_DIR "${CMAKE_INSTALL_PREFIX}/${CMAKE_INSTALL_LIBDIR}")

## Create the rocm-core-config.cmake and rocm-core-config-version files
include(CMakePackageConfigHelpers)
set(CONFIG_PACKAGE_INSTALL_DIR ${CONF_LIB_DIR}/cmake/${CORE_TARGET})
set(LIB_INSTALL_DIR  "${CMAKE_INSTALL_LIBDIR}")
set(INCLUDE_INSTALL_DIR  "${CMAKE_INSTALL_INCLUDEDIR}")
set(BIN_INSTALL_DIR  "${CMAKE_INSTALL_BINDIR}")
set(PKG_MAINTAINER_NM  "ROCm Dev Support")
set(PKG_MAINTAINER_EMAIL  "rocm-dev.support@amd.com")

## Generate Config File for rocm-core
configure_package_config_file(
    ${CMAKE_MODULE_PATH}/${CORE_TARGET}-config.cmake.in
    ${BUILD_DIR}/${CORE_TARGET}-config.cmake
    INSTALL_DESTINATION ${CONFIG_PACKAGE_INSTALL_DIR}
    PATH_VARS LIB_INSTALL_DIR INCLUDE_INSTALL_DIR BIN_INSTALL_DIR
)

## Generate config Version File for rocm-core
write_basic_package_version_file(
    ${BUILD_DIR}/${CORE_TARGET}-config-version.cmake
    VERSION "${VERSION_MAJOR}.${VERSION_MINOR}.${VERSION_PATCH}"
    COMPATIBILITY SameMajorVersion)

## Install the rocm-core-config.cmake and rocm-core-config-version.cmake
install(FILES
    "${PROJECT_BINARY_DIR}/${CORE_TARGET}-config.cmake"
    "${PROJECT_BINARY_DIR}/${CORE_TARGET}-config-version.cmake"
    DESTINATION "${CMAKE_INSTALL_LIBDIR}/cmake/${CORE_TARGET}"
    COMPONENT ${COMP_TYPE})

## Install the export set for use with the install-tree
install(EXPORT rocmCoreTargets DESTINATION
    "${CMAKE_INSTALL_LIBDIR}/cmake/${CORE_TARGET}" COMPONENT ${COMP_TYPE})

## Packaging directives
set ( CPACK_PACKAGE_VENDOR "Advanced Micro Devices, Inc." )
set ( CPACK_PACKAGE_VERSION_MAJOR ${VERSION_MAJOR} )
set ( CPACK_PACKAGE_VERSION_MINOR ${VERSION_MINOR} )
set ( CPACK_PACKAGE_VERSION_PATCH ${VERSION_PATCH} )
set ( CPACK_PACKAGE_VERSION "${CPACK_PACKAGE_VERSION_MAJOR}.${CPACK_PACKAGE_VERSION_MINOR}.${CPACK_PACKAGE_VERSION_PATCH}" )
set ( CPACK_PACKAGE_CONTACT "${PKG_MAINTAINER_NM} <${PKG_MAINTAINER_EMAIL}>" )
set ( CPACK_PACKAGE_DESCRIPTION_SUMMARY "${PKG_DESCRIPTION}" )
set ( CPACK_RESOURCE_FILE_LICENSE "${LICENSE_FILE}" )
## packaging variables
if ( DEFINED ROCM_LIBPATCH_VERSION )
  set ( CPACK_PACKAGE_VERSION "${CPACK_PACKAGE_VERSION}.${ROCM_LIBPATCH_VERSION}" )
  message ( STATUS "Using CPACK_PACKAGE_VERSION ${CPACK_PACKAGE_VERSION}" )
endif()

## Debian package specific variables
set ( CPACK_DEB_COMPONENT_INSTALL ON )
set ( CPACK_DEBIAN_CORE_RUNTIME_PACKAGE_NAME "${CORE_TARGET}" )
# ASAN package name
set ( CPACK_DEBIAN_CORE_ASAN_PACKAGE_NAME ${CORE_TARGET}-asan )
# STATIC package name - For Now, No change for static package Name from default
set ( CPACK_DEBIAN_CORE_STATIC_PACKAGE_NAME ${CORE_TARGET} )
message ( STATUS "Using CPACK_DEBIAN_PACKAGE_RELEASE ${CPACK_DEBIAN_PACKAGE_RELEASE}" )
set ( CPACK_DEBIAN_FILE_NAME "DEB-DEFAULT" )

# Debian specific packaging configuration
if(COMP_TYPE)
configure_debian_pkg( ${CORE_TARGET} ${COMP_TYPE} ${CPACK_PACKAGE_VERSION} ${PKG_MAINTAINER_NM} ${PKG_MAINTAINER_EMAIL} )
elseif(STATIC_COMP_TYPE)
configure_debian_pkg( ${CORE_TARGET} ${STATIC_COMP_TYPE} ${CPACK_PACKAGE_VERSION} ${PKG_MAINTAINER_NM} ${PKG_MAINTAINER_EMAIL} )
endif()

## RPM package specific variables
set ( CPACK_RPM_COMPONENT_INSTALL ON )
set ( CPACK_RPM_CORE_RUNTIME_PACKAGE_NAME "${CORE_TARGET}" )
# ASAN package name
set ( CPACK_RPM_CORE_ASAN_PACKAGE_NAME ${CORE_TARGET}-asan )
# STATIC package name - For Now, No change for static package Name from default
set ( CPACK_RPM_CORE_STATIC_PACKAGE_NAME ${CORE_TARGET} )

## 'dist' breaks manual builds on debian systems due to empty Provides
execute_process( COMMAND rpm --eval %{?dist}
                 RESULT_VARIABLE PROC_RESULT
                 OUTPUT_VARIABLE EVAL_RESULT
                 OUTPUT_STRIP_TRAILING_WHITESPACE )

if ( PROC_RESULT EQUAL "0" AND NOT EVAL_RESULT STREQUAL "" )
  string ( APPEND CPACK_RPM_PACKAGE_RELEASE "%{?dist}" )
endif()
message(STATUS "Using CPACK_RPM_PACKAGE_RELEASE: ${CPACK_RPM_PACKAGE_RELEASE}")
set ( CPACK_RPM_FILE_NAME "RPM-DEFAULT" )

# Debian package specific variables
set ( CPACK_DEBIAN_PACKAGE_DEPENDS ${DEB_DEPENDS_STRING} )
set ( CPACK_DEBIAN_PACKAGE_HOMEPAGE "https://github.com/ROCm/rocm-core" )
set ( CPACK_DEBIAN_PACKAGE_CONTROL_EXTRA "${BUILD_DIR}/postinst;${BUILD_DIR}/prerm" )
set ( CPACK_DEBIAN_CORE_ASAN_PACKAGE_DEPENDS ${CORE_TARGET} )
set ( CPACK_DEBIAN_CORE_RUNTIME_PACKAGE_CONTROL_EXTRA "${CPACK_DEBIAN_PACKAGE_CONTROL_EXTRA}" )
set ( CPACK_DEBIAN_CORE_STATIC_PACKAGE_CONTROL_EXTRA "${CPACK_DEBIAN_PACKAGE_CONTROL_EXTRA}" )
set ( CPACK_DEBIAN_PACKAGE_DESCRIPTION "${EXTENDED_PACKAGE_DESCRIPTION}" )

## RPM package specific variables
set ( CPACK_RPM_PACKAGE_REQUIRES ${RPM_DEPENDS_STRING} )
set ( CPACK_RPM_PACKAGE_LICENSE "MIT" )
set ( CPACK_RPM_CORE_ASAN_PACKAGE_REQUIRES ${CORE_TARGET} )
set ( CPACK_RPM_CORE_RUNTIME_POST_INSTALL_SCRIPT_FILE "${BUILD_DIR}/postinst" )
set ( CPACK_RPM_CORE_RUNTIME_POST_UNINSTALL_SCRIPT_FILE "${BUILD_DIR}/prerm" )
set ( CPACK_RPM_CORE_STATIC_POST_INSTALL_SCRIPT_FILE "${BUILD_DIR}/postinst" )
set ( CPACK_RPM_CORE_STATIC_POST_UNINSTALL_SCRIPT_FILE "${BUILD_DIR}/prerm" )
set ( CPACK_RPM_PACKAGE_DESCRIPTION "${EXTENDED_PACKAGE_DESCRIPTION}" )

if ( DEFINED CPACK_PACKAGING_INSTALL_PREFIX )
    set ( CPACK_RPM_EXCLUDE_FROM_AUTO_FILELIST_ADDITION 
          "${CPACK_PACKAGING_INSTALL_PREFIX} ${CPACK_PACKAGING_INSTALL_PREFIX}/.info" )
endif()

# Backward compatibility support for deprecated package rocm-clang-ocl
# TODO: To be removed after two releases
set(CPACK_DEBIAN_RUNTIME_PACKAGE_REPLACES "rocm-clang-ocl")
set(CPACK_DEBIAN_RUNTIME_PACKAGE_CONFLICTS "rocm-clang-ocl")
set(CPACK_RPM_RUNTIME_PACKAGE_OBSOLETES "rocm-clang-ocl")

## Include packaging
include ( CPack )
