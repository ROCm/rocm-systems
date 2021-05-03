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

cmake_minimum_required ( VERSION 3.5.0 )

## Verbose output.
set ( CMAKE_VERBOSE_MAKEFILE TRUE CACHE BOOL "Verbose Output" FORCE )

# Install prefix
set(CMAKE_INSTALL_PREFIX "/opt/rocm" CACHE PATH "Install path prefix, prepended onto install directories")

## Set module name and project name.
set ( ROCTRACER_NAME "roctracer" )
set ( ROCTRACER_TARGET "${ROCTRACER_NAME}64" )
set ( ROCTRACER_LIBRARY "lib${ROCTRACER_TARGET}" )
project ( ${ROCTRACER_TARGET} )

## Adding default path cmake modules
list ( APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake_modules" )
## Include common cmake modules
include ( utils )
## Set build environment
include ( env )

## Setup the package version.
get_version ( "1.0.0" )

set ( BUILD_VERSION_MAJOR ${VERSION_MAJOR} )
set ( BUILD_VERSION_MINOR ${VERSION_MINOR} )
set ( BUILD_VERSION_PATCH ${VERSION_PATCH} )

if ( DEFINED VERSION_BUILD AND NOT ${VERSION_BUILD} STREQUAL "" )
  message ( "VERSION BUILD DEFINED ${VERSION_BUILD}" )
  set ( BUILD_VERSION_PATCH "${BUILD_VERSION_PATCH}-${VERSION_BUILD}" )
endif ()

set ( LIB_VERSION_MAJOR ${BUILD_VERSION_MAJOR} )
set ( LIB_VERSION_MINOR ${BUILD_VERSION_MINOR} )
if (DEFINED ENV{ROCM_LIBPATCH_VERSION})
  set (LIB_VERSION_PATCH $ENV{ROCM_LIBPATCH_VERSION} )
else ()
  set (LIB_VERSION_PATCH ${BUILD_VERSION_PATCH} )
endif()
set ( LIB_VERSION_STRING "${LIB_VERSION_MAJOR}.${LIB_VERSION_MINOR}.${LIB_VERSION_PATCH}" )
message ( "-- LIB-VERSION: ${LIB_VERSION_MAJOR}.${LIB_VERSION_MINOR}.${LIB_VERSION_PATCH}" )

## Set target and root/lib/test directory
set ( TARGET_NAME "${ROCTRACER_TARGET}" )
set ( ROOT_DIR "${CMAKE_CURRENT_SOURCE_DIR}" )
set ( LIB_DIR "${ROOT_DIR}/src" )
set ( TEST_DIR "${ROOT_DIR}/test" )

## Build library
include ( ${LIB_DIR}/CMakeLists.txt )

## Set the VERSION and SOVERSION values
set_property ( TARGET ${TARGET_NAME} PROPERTY VERSION "${LIB_VERSION_STRING}" )
set_property ( TARGET ${TARGET_NAME} PROPERTY SOVERSION "${LIB_VERSION_MAJOR}" )

# If the library is a release, strip the target library
#if ( "${CMAKE_BUILD_TYPE}" STREQUAL release )
#  add_custom_command ( TARGET ${ROCTRACER_TARGET} POST_BUILD COMMAND ${CMAKE_STRIP} *.so )
#endif ()

## Build tests
if ( ${LIBRARY_TYPE} STREQUAL SHARED )
add_subdirectory ( ${TEST_DIR} ${PROJECT_BINARY_DIR}/test )
endif()

## pbulic headers
set ( PUBLIC_HEADERS
  roctx.h
  roctracer.h
  roctracer_ext.h
  roctracer_hcc.h
  roctracer_hip.h
  roctracer_hsa.h
  roctracer_kfd.h
  roctracer_roctx.h
  roctracer_cb_table.h
  ext/prof_protocol.h
  ext/hsa_rt_utils.hpp
)
set ( GEN_HEADERS
  hip_ostream_ops.h
  hsa_prof_str.h
  hsa_ostream_ops.h
  kfd_prof_str.h
  kfd_ostream_ops.h
)

if ( ${LIBRARY_TYPE} STREQUAL SHARED )

## Installation and packaging
set ( DEST_NAME ${ROCTRACER_NAME} )
if ( DEFINED CPACK_PACKAGING_INSTALL_PREFIX )
  get_filename_component ( DEST_NAME ${CPACK_PACKAGING_INSTALL_PREFIX} NAME )
  get_filename_component ( DEST_DIR ${CPACK_PACKAGING_INSTALL_PREFIX} DIRECTORY )
  set ( CPACK_PACKAGING_INSTALL_PREFIX ${DEST_DIR} )
endif ()
message ( "-----------Dest-name: ${DEST_NAME}" )
message ( "------Install-prefix: ${CMAKE_INSTALL_PREFIX}" )
message ( "-----------CPACK-dir: ${CPACK_PACKAGING_INSTALL_PREFIX}" )

#add_custom_target ( inc-link ALL WORKING_DIRECTORY ${PROJECT_BINARY_DIR}
#                    COMMAND ${CMAKE_COMMAND} -E create_symlink ../${DEST_NAME}/include inc-link )
add_custom_target ( so-link ALL WORKING_DIRECTORY ${PROJECT_BINARY_DIR}
                    COMMAND ${CMAKE_COMMAND} -E create_symlink ../${DEST_NAME}/lib/${ROCTRACER_LIBRARY}.so so-link )
add_custom_target ( so-major-link ALL WORKING_DIRECTORY ${PROJECT_BINARY_DIR}
                    COMMAND ${CMAKE_COMMAND} -E create_symlink ../${DEST_NAME}/lib/${ROCTRACER_LIBRARY}.so.${LIB_VERSION_MAJOR} so-major-link )
add_custom_target ( so-patch-link ALL WORKING_DIRECTORY ${PROJECT_BINARY_DIR}
                    COMMAND ${CMAKE_COMMAND} -E create_symlink ../${DEST_NAME}/lib/${ROCTRACER_LIBRARY}.so.${LIB_VERSION_STRING} so-patch-link )

## Install information
install ( TARGETS ${ROCTRACER_TARGET} LIBRARY DESTINATION ${DEST_NAME}/lib )
foreach ( header ${PUBLIC_HEADERS} )
  get_filename_component ( header_subdir ${header} DIRECTORY )
  install ( FILES ${CMAKE_CURRENT_SOURCE_DIR}/inc/${header} DESTINATION ${DEST_NAME}/include/${header_subdir} )
  install ( FILES ${CMAKE_CURRENT_SOURCE_DIR}/inc/${header} DESTINATION include/${DEST_NAME}/${header_subdir} )
endforeach ()
foreach ( header ${GEN_HEADERS} )
  get_filename_component ( header_subdir ${header} DIRECTORY )
  install ( FILES ${PROJECT_BINARY_DIR}/inc/${header} DESTINATION ${DEST_NAME}/include/${header_subdir} )
  install ( FILES ${PROJECT_BINARY_DIR}/inc/${header} DESTINATION include/${DEST_NAME}/${header_subdir} )
endforeach ()
#install ( FILES ${PROJECT_BINARY_DIR}/inc-link DESTINATION include RENAME ${DEST_NAME} )
install ( FILES ${PROJECT_BINARY_DIR}/so-link DESTINATION lib RENAME ${ROCTRACER_LIBRARY}.so )
install ( FILES ${PROJECT_BINARY_DIR}/so-major-link DESTINATION lib RENAME ${ROCTRACER_LIBRARY}.so.${LIB_VERSION_MAJOR} )
install ( FILES ${PROJECT_BINARY_DIR}/so-patch-link DESTINATION lib RENAME ${ROCTRACER_LIBRARY}.so.${LIB_VERSION_STRING} )
install ( FILES ${PROJECT_BINARY_DIR}/test/libtracer_tool.so DESTINATION ${DEST_NAME}/tool )

## rocTX
set ( ROCTX_TARGET "roctx64" )
set ( ROCTX_LIBRARY "lib${ROCTX_TARGET}" )

## Set the VERSION and SOVERSION values
set_property ( TARGET ${ROCTX_TARGET} PROPERTY VERSION "${LIB_VERSION_STRING}" )
set_property ( TARGET ${ROCTX_TARGET} PROPERTY SOVERSION "${LIB_VERSION_MAJOR}" )

add_custom_target ( so-roctx-link ALL WORKING_DIRECTORY ${PROJECT_BINARY_DIR}
                    COMMAND ${CMAKE_COMMAND} -E create_symlink ../${DEST_NAME}/lib/${ROCTX_LIBRARY}.so so-roctx-link )
add_custom_target ( so-roctx-major-link ALL WORKING_DIRECTORY ${PROJECT_BINARY_DIR}
                    COMMAND ${CMAKE_COMMAND} -E create_symlink ../${DEST_NAME}/lib/${ROCTX_LIBRARY}.so.${LIB_VERSION_MAJOR} so-roctx-major-link )
add_custom_target ( so-roctx-patch-link ALL WORKING_DIRECTORY ${PROJECT_BINARY_DIR}
                    COMMAND ${CMAKE_COMMAND} -E create_symlink ../${DEST_NAME}/lib/${ROCTX_LIBRARY}.so.${LIB_VERSION_STRING} so-roctx-patch-link )

install ( TARGETS "roctx64" LIBRARY DESTINATION ${DEST_NAME}/lib )
install ( FILES ${PROJECT_BINARY_DIR}/so-roctx-link DESTINATION lib RENAME ${ROCTX_LIBRARY}.so )
install ( FILES ${PROJECT_BINARY_DIR}/so-roctx-major-link DESTINATION lib RENAME ${ROCTX_LIBRARY}.so.${LIB_VERSION_MAJOR} )
install ( FILES ${PROJECT_BINARY_DIR}/so-roctx-patch-link DESTINATION lib RENAME ${ROCTX_LIBRARY}.so.${LIB_VERSION_STRING} )

## KFD wrapper
install ( TARGETS "kfdwrapper64" LIBRARY DESTINATION ${DEST_NAME}/lib )

## Packaging directives
set ( CPACK_GENERATOR "DEB" "RPM" "TGZ" )
set ( ENABLE_LDCONFIG ON CACHE BOOL "Set library links and caches using ldconfig.")
set ( CPACK_PACKAGE_NAME "${ROCTRACER_NAME}-dev" )
set ( CPACK_PACKAGE_VENDOR "Advanced Micro Devices, Inc." )
set ( CPACK_PACKAGE_VERSION_MAJOR ${BUILD_VERSION_MAJOR} )
set ( CPACK_PACKAGE_VERSION_MINOR ${BUILD_VERSION_MINOR} )
set ( CPACK_PACKAGE_VERSION_PATCH ${BUILD_VERSION_PATCH} )
set ( CPACK_PACKAGE_VERSION "${CPACK_PACKAGE_VERSION_MAJOR}.${CPACK_PACKAGE_VERSION_MINOR}.${CPACK_PACKAGE_VERSION_PATCH}" )
if ( DEFINED ENV{ROCM_LIBPATCH_VERSION} )
  set ( CPACK_PACKAGE_VERSION "${CPACK_PACKAGE_VERSION}.$ENV{ROCM_LIBPATCH_VERSION}" )
endif()
message ( "-- CPACK_PACKAGE_VERSION: ${CPACK_PACKAGE_VERSION}"  )
set ( CPACK_PACKAGE_CONTACT "ROCm Profiler Support <dl.ROCm-Profiler.support@amd.com>" )
set ( CPACK_PACKAGE_DESCRIPTION_SUMMARY "AMD ROCTRACER library" )
set ( CPACK_RESOURCE_FILE_LICENSE "${CMAKE_CURRENT_SOURCE_DIR}/LICENSE" )

## Debian package specific variables
if ( DEFINED ENV{CPACK_DEBIAN_PACKAGE_RELEASE} )
  set ( CPACK_DEBIAN_PACKAGE_RELEASE $ENV{CPACK_DEBIAN_PACKAGE_RELEASE} )
else()
  set ( CPACK_DEBIAN_PACKAGE_RELEASE "local" )
endif()
message ( "Using CPACK_DEBIAN_PACKAGE_RELEASE ${CPACK_DEBIAN_PACKAGE_RELEASE}" )
set ( CPACK_DEBIAN_FILE_NAME "DEB-DEFAULT" )
## Process the Debian install/remove scripts to update the CPACK variables
configure_file ( ${CMAKE_CURRENT_SOURCE_DIR}/DEBIAN/postinst.in DEBIAN/postinst @ONLY )
configure_file ( ${CMAKE_CURRENT_SOURCE_DIR}/DEBIAN/prerm.in DEBIAN/prerm @ONLY )
set ( CPACK_DEBIAN_PACKAGE_CONTROL_EXTRA "DEBIAN/postinst;DEBIAN/prerm" )

## RPM package specific variables
if ( DEFINED ENV{CPACK_RPM_PACKAGE_RELEASE} )
  set ( CPACK_RPM_PACKAGE_RELEASE $ENV{CPACK_RPM_PACKAGE_RELEASE} )
else()
  set ( CPACK_RPM_PACKAGE_RELEASE "local" )
endif()
message ( "Using CPACK_RPM_PACKAGE_RELEASE ${CPACK_RPM_PACKAGE_RELEASE}" )

## 'dist' breaks manual builds on debian systems due to empty Provides
execute_process( COMMAND rpm --eval %{?dist}
                 RESULT_VARIABLE PROC_RESULT
                 OUTPUT_VARIABLE EVAL_RESULT
                 OUTPUT_STRIP_TRAILING_WHITESPACE )
message("RESULT_VARIABLE ${PROC_RESULT} OUTPUT_VARIABLE: ${EVAL_RESULT}")

if ( PROC_RESULT EQUAL "0" AND NOT EVAL_RESULT STREQUAL "" )
  string ( APPEND CPACK_RPM_PACKAGE_RELEASE "%{?dist}" )
endif()
set ( CPACK_RPM_FILE_NAME "RPM-DEFAULT" )
message("CPACK_RPM_PACKAGE_RELEASE: ${CPACK_RPM_PACKAGE_RELEASE}")
## Process the Rpm install/remove scripts to update the CPACK variables
configure_file ( "${CMAKE_CURRENT_SOURCE_DIR}/RPM/post.in" RPM/post @ONLY )
configure_file ( "${CMAKE_CURRENT_SOURCE_DIR}/RPM/postun.in" RPM/postun @ONLY )

set ( CPACK_RPM_POST_INSTALL_SCRIPT_FILE "${CMAKE_CURRENT_BINARY_DIR}/RPM/post" )
set ( CPACK_RPM_POST_UNINSTALL_SCRIPT_FILE "${CMAKE_CURRENT_BINARY_DIR}/RPM/postun" )

include ( CPack )

endif()
