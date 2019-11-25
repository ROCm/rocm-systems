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

cmake_minimum_required ( VERSION 2.8.12 )

## Verbose output.
set ( CMAKE_VERBOSE_MAKEFILE TRUE CACHE BOOL "Verbose Output" FORCE )

## Set module name and project name.
set ( ROCPROFILER_NAME "rocprofiler" )
set ( ROCPROFILER_TARGET "${ROCPROFILER_NAME}64" )
set ( ROCPROFILER_LIBRARY "lib${ROCPROFILER_TARGET}" )
project ( ${ROCPROFILER_TARGET} )

## Adding default path cmake modules
list ( APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake_modules" )
## Include common cmake modules
include ( utils )
## Set build environment
include ( env )

## Setup the package version.
get_version ( "1.0.0" )
message ( "-- LIB-VERSION: ${VERSION_MAJOR}.${VERSION_MINOR}.${VERSION_PATCH}" )

set ( BUILD_VERSION_MAJOR ${VERSION_MAJOR} )
set ( BUILD_VERSION_MINOR ${VERSION_MINOR} )
set ( BUILD_VERSION_PATCH ${VERSION_PATCH} )
set ( LIB_VERSION_STRING "${BUILD_VERSION_MAJOR}.${BUILD_VERSION_MINOR}.${BUILD_VERSION_PATCH}" )
if ( DEFINED VERSION_BUILD AND NOT ${VERSION_BUILD} STREQUAL "" )
  message ( "VERSION BUILD DEFINED ${VERSION_BUILD}" )
  set ( BUILD_VERSION_PATCH "${BUILD_VERSION_PATCH}-${VERSION_BUILD}" )
endif ()
set ( BUILD_VERSION_STRING "${BUILD_VERSION_MAJOR}.${BUILD_VERSION_MINOR}.${BUILD_VERSION_PATCH}" )

## Set target and root/lib/test directory
set ( TARGET_NAME "${ROCPROFILER_TARGET}" )
set ( ROOT_DIR "${CMAKE_CURRENT_SOURCE_DIR}" )
set ( LIB_DIR "${ROOT_DIR}/src" )
set ( TEST_DIR "${ROOT_DIR}/test" )

## Build library
include ( ${LIB_DIR}/CMakeLists.txt )

## Set the VERSION and SOVERSION values
set_property ( TARGET ${TARGET_NAME} PROPERTY VERSION "${LIB_VERSION_STRING}" )
set_property ( TARGET ${TARGET_NAME} PROPERTY SOVERSION "${BUILD_VERSION_MAJOR}" )

## If the library is a release, strip the target library
if ( "${CMAKE_BUILD_TYPE}" STREQUAL release )
  add_custom_command ( TARGET ${ROCPROFILER_TARGET} POST_BUILD COMMAND ${CMAKE_STRIP} *.so )
endif ()

## Build tests
add_subdirectory ( ${TEST_DIR} ${PROJECT_BINARY_DIR}/test )

## Create symlinks for packaging and install
add_custom_target ( rocprof-link ALL WORKING_DIRECTORY ${PROJECT_BINARY_DIR}
                    COMMAND ${CMAKE_COMMAND} -E create_symlink ../${ROCPROFILER_NAME}/bin/rpl_run.sh rocprof-link )
add_custom_target ( inc-link ALL WORKING_DIRECTORY ${PROJECT_BINARY_DIR}
                    COMMAND ${CMAKE_COMMAND} -E create_symlink ../${ROCPROFILER_NAME}/include inc-link )
add_custom_target ( so-link ALL WORKING_DIRECTORY ${PROJECT_BINARY_DIR}
                    COMMAND ${CMAKE_COMMAND} -E create_symlink ../${ROCPROFILER_NAME}/lib/${ROCPROFILER_LIBRARY}.so so-link )

set ( CMAKE_INSTALL_PREFIX "${CMAKE_INSTALL_PREFIX}/${ROCPROFILER_NAME}" )
message ( "---------Install-Dir: ${CMAKE_INSTALL_PREFIX}" )
install ( TARGETS ${ROCPROFILER_TARGET} LIBRARY DESTINATION lib )
install ( FILES ${CMAKE_CURRENT_SOURCE_DIR}/inc/rocprofiler.h DESTINATION include )
# rpl_run.sh tblextr.py txt2xml.sh
install ( FILES
          ${CMAKE_CURRENT_SOURCE_DIR}/bin/rpl_run.sh
          ${CMAKE_CURRENT_SOURCE_DIR}/bin/txt2xml.sh
          ${CMAKE_CURRENT_SOURCE_DIR}/bin/tblextr.py
          ${CMAKE_CURRENT_SOURCE_DIR}/bin/dform.py
          ${CMAKE_CURRENT_SOURCE_DIR}/bin/sqlitedb.py
          DESTINATION bin
          PERMISSIONS OWNER_READ OWNER_EXECUTE GROUP_READ GROUP_EXECUTE WORLD_READ WORLD_EXECUTE )
install ( FILES ${PROJECT_BINARY_DIR}/inc-link DESTINATION ../include RENAME ${ROCPROFILER_NAME} )
install ( FILES ${PROJECT_BINARY_DIR}/so-link DESTINATION ../lib RENAME ${ROCPROFILER_LIBRARY}.so )
install ( FILES ${PROJECT_BINARY_DIR}/rocprof-link DESTINATION ../bin
          PERMISSIONS OWNER_READ OWNER_EXECUTE GROUP_READ GROUP_EXECUTE WORLD_READ WORLD_EXECUTE
          RENAME rocprof )
# gfx_metrics.xml metrics.xml
install ( FILES
          ${CMAKE_CURRENT_SOURCE_DIR}/test/tool/metrics.xml
          ${CMAKE_CURRENT_SOURCE_DIR}/test/tool/gfx_metrics.xml
          DESTINATION lib )
# libtool.so
install ( FILES ${PROJECT_BINARY_DIR}/test/libtool.so DESTINATION tool )
install ( FILES ${PROJECT_BINARY_DIR}/test/ctrl DESTINATION tool
          PERMISSIONS OWNER_READ OWNER_EXECUTE GROUP_READ GROUP_EXECUTE WORLD_READ WORLD_EXECUTE )

## Packaging directives
set ( CPACK_GENERATOR "DEB" "RPM" "TGZ" )
set ( CPACK_PACKAGE_NAME "${ROCPROFILER_NAME}-dev" )
set ( CPACK_PACKAGE_VENDOR "AMD" )
set ( CPACK_PACKAGE_VERSION_MAJOR ${BUILD_VERSION_MAJOR} )
set ( CPACK_PACKAGE_VERSION_MINOR ${BUILD_VERSION_MINOR} )
set ( CPACK_PACKAGE_VERSION_PATCH ${BUILD_VERSION_PATCH} )
set ( CPACK_PACKAGE_CONTACT "Advanced Micro Devices Inc." )
set ( CPACK_PACKAGE_DESCRIPTION_SUMMARY "ROCPROFILER library for AMD HSA runtime API extension support" )
set ( CPACK_RESOURCE_FILE_LICENSE "${CMAKE_CURRENT_SOURCE_DIR}/LICENSE" )

## Debian package specific variables
set ( CPACK_DEBIAN_PACKAGE_DEPENDS "hsa-rocr-dev" )
set ( CPACK_DEBIAN_PACKAGE_CONTROL_EXTRA "${CMAKE_CURRENT_SOURCE_DIR}/DEBIAN/postinst;${CMAKE_CURRENT_SOURCE_DIR}/DEBIAN/prerm" )

## RPM package specific variables
set ( CPACK_RPM_PACKAGE_DEPENDS "hsa-rocr-dev" )
set ( CPACK_RPM_POST_INSTALL_SCRIPT_FILE "${CMAKE_CURRENT_SOURCE_DIR}/RPM/rpm_post" )
set ( CPACK_RPM_POST_UNINSTALL_SCRIPT_FILE "${CMAKE_CURRENT_SOURCE_DIR}/RPM/rpm_postun" )

include ( CPack )
