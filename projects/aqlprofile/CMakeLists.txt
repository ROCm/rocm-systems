

cmake_minimum_required(VERSION 3.16.0)

## Set module name and project name.
set ( AQLPROFILE_NAME "hsa-amd-aqlprofile" )
set ( AQLPROFILE_TARGET "${AQLPROFILE_NAME}64" )
set ( AQLPROFILE_LIBRARY "lib${AQLPROFILE_TARGET}" )
project ( ${AQLPROFILE_NAME} )

include(GNUInstallDirs)
## Adding default path cmake modules
list ( APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake_modules" )
## Include common cmake modules
include ( utils )
## Set build environment
include ( env )

set(CMAKE_INSTALL_LIBDIR "lib" CACHE STRING "Library install directory")

## Setup the package version.
get_version ( "1.0.0" )
message ( "-- BUILD-VERSION: ${VERSION_MAJOR}.${VERSION_MINOR}.${VERSION_PATCH}" )

set ( BUILD_VERSION_MAJOR ${VERSION_MAJOR} )
set ( BUILD_VERSION_MINOR ${VERSION_MINOR} )
set ( BUILD_VERSION_PATCH ${VERSION_PATCH} )
set ( BUILD_VERSION_STRING "${BUILD_VERSION_MAJOR}.${BUILD_VERSION_MINOR}.${BUILD_VERSION_PATCH}" )

set ( LIB_VERSION_MAJOR ${VERSION_MAJOR} )
set ( LIB_VERSION_MINOR ${VERSION_MINOR} )
if ( ${ROCM_PATCH_VERSION} )
   set ( LIB_VERSION_PATCH ${ROCM_PATCH_VERSION} )
else()
   set ( LIB_VERSION_PATCH ${VERSION_PATCH} )
endif()
set ( LIB_VERSION_STRING "${LIB_VERSION_MAJOR}.${LIB_VERSION_MINOR}.${LIB_VERSION_PATCH}" )

## Set target and root/lib/test directory
set ( TARGET_NAME "${AQLPROFILE_TARGET}" )
set ( ROOT_DIR "${CMAKE_CURRENT_SOURCE_DIR}" )
set ( LIB_DIR "${ROOT_DIR}/src" )
set ( TEST_DIR "${ROOT_DIR}/test" )

## Build library
include ( ${LIB_DIR}/CMakeLists.txt )

## Set the VERSION and SOVERSION values
set_property ( TARGET ${TARGET_NAME} PROPERTY VERSION "${LIB_VERSION_STRING}" )
set_property ( TARGET ${TARGET_NAME} PROPERTY SOVERSION "${LIB_VERSION_MAJOR}" )

## If the library is a release, strip the target library
if ( "${CMAKE_BUILD_TYPE}" STREQUAL release )
  add_custom_command ( TARGET ${AQLPROFILE_TARGET} POST_BUILD COMMAND ${CMAKE_STRIP} *aqlprofile*.so )
endif ()

## Build tests
enable_testing()
include(CTest)
set ( TEST_BINARY_DIR ${PROJECT_BINARY_DIR}/test )
add_subdirectory ( ${TEST_DIR} ${TEST_BINARY_DIR} )

## Add the install directives for the runtime library.
set ( DEST_NAME ${AQLPROFILE_NAME} )
install ( TARGETS ${AQLPROFILE_TARGET} LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR} COMPONENT runtime )
install ( TARGETS ${AQLPROFILE_TARGET} LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR} COMPONENT asan )

# Enable/disable test
option(AQLPROFILE_BUILD_TESTS "Build tests for AQLProfile" OFF)

## Add the packaging directives for the runtime library.
if ( ENABLE_ASAN_PACKAGING )
  set ( CPACK_PACKAGE_NAME ${AQLPROFILE_NAME}-asan )
  # ASAN Package requires only asan component with libraries and license file
  set ( CPACK_COMPONENTS_ALL asan )
else()
  set ( CPACK_PACKAGE_NAME ${AQLPROFILE_NAME} )
  set ( CPACK_COMPONENTS_ALL runtime tests )
endif()
set ( CPACK_PACKAGE_VENDOR "Advanced Micro Devices, Inc." )
set ( CPACK_PACKAGE_VERSION_MAJOR ${BUILD_VERSION_MAJOR} )
set ( CPACK_PACKAGE_VERSION_MINOR ${BUILD_VERSION_MINOR} )
set ( CPACK_PACKAGE_VERSION_PATCH ${BUILD_VERSION_PATCH} )
set ( CPACK_PACKAGE_CONTACT "ROCm Profiler Support <dl.rocm-profiler.support@amd.com>" )
set ( CPACK_PACKAGE_VERSION "${CPACK_PACKAGE_VERSION_MAJOR}.${CPACK_PACKAGE_VERSION_MINOR}.${CPACK_PACKAGE_VERSION_PATCH}" )
set ( CPACK_PACKAGE_DESCRIPTION_SUMMARY "AQLPROFILE library for AMD HSA runtime API extension support" )
set ( ENABLE_LDCONFIG ON CACHE BOOL "Set library links and caches using ldconfig.")


set(CPACK_STGZ_HEADER_FILE ${PROJECT_SOURCE_DIR}/cmake_modules/CPack.STGZ_Header.sh.in)
set(CPACK_STGZ_INCLUDE_SUBDIR OFF)

if ( DEFINED ENV{ROCM_LIBPATCH_VERSION} )
  set ( CPACK_PACKAGE_VERSION "${CPACK_PACKAGE_VERSION}.$ENV{ROCM_LIBPATCH_VERSION}" )
endif()

## Debian package specific variables
set ( CPACK_DEBIAN_PACKAGE_HOMEPAGE "https://github.com/RadeonOpenCompute/HSA-AqlProfile-AMD-extension" )
## Process the Debian install/remove scripts to update the CPACK variables
configure_file ( ${CMAKE_CURRENT_SOURCE_DIR}/DEBIAN/postinst.in DEBIAN/postinst @ONLY )
configure_file ( ${CMAKE_CURRENT_SOURCE_DIR}/DEBIAN/prerm.in DEBIAN/prerm @ONLY )
set ( CPACK_DEBIAN_PACKAGE_CONTROL_EXTRA "DEBIAN/postinst;DEBIAN/prerm" )

## Process the Rpm install/remove scripts to update the CPACK variables
configure_file ( "${CMAKE_CURRENT_SOURCE_DIR}/RPM/post.in" RPM/post @ONLY )
configure_file ( "${CMAKE_CURRENT_SOURCE_DIR}/RPM/postun.in" RPM/postun @ONLY )
## RPM package specific variables
set ( CPACK_RPM_PRE_INSTALL_SCRIPT_FILE "${CMAKE_CURRENT_BINARY_DIR}/RPM/post" )
set ( CPACK_RPM_POST_UNINSTALL_SCRIPT_FILE "${CMAKE_CURRENT_BINARY_DIR}/RPM/postun" )

if ( DEFINED ENV{CPACK_DEBIAN_PACKAGE_RELEASE} )
  set ( CPACK_DEBIAN_PACKAGE_RELEASE $ENV{CPACK_DEBIAN_PACKAGE_RELEASE} )
else()
  set ( CPACK_DEBIAN_PACKAGE_RELEASE "local" )
endif()
set ( CPACK_DEBIAN_FILE_NAME "DEB-DEFAULT" )

## RPM package specific variables
if ( DEFINED ENV{CPACK_RPM_PACKAGE_RELEASE} )
  set ( CPACK_RPM_PACKAGE_RELEASE $ENV{CPACK_RPM_PACKAGE_RELEASE} )
else()
  set ( CPACK_RPM_PACKAGE_RELEASE "local" )
endif()
set( CPACK_RPM_PACKAGE_LICENSE "AMD Proprietary" )

#Disable build id for rocprofiler as its creating transaction error
set ( CPACK_RPM_SPEC_MORE_DEFINE "%define _build_id_links none
                                    %global __strip ${CPACK_STRIP_EXECUTABLE}
                                    %global __objdump ${CPACK_OBJDUMP_EXECUTABLE}
                                    %global __objcopy ${CPACK_OBJCOPY_EXECUTABLE}
                                    %global __readelf ${CPACK_READELF_EXECUTABLE}")

## 'dist' breaks manual builds on debian systems due to empty Provides
execute_process( COMMAND rpm --eval %{?dist}
                 RESULT_VARIABLE PROC_RESULT
                 OUTPUT_VARIABLE EVAL_RESULT
                 OUTPUT_STRIP_TRAILING_WHITESPACE )

if ( PROC_RESULT EQUAL "0" AND NOT EVAL_RESULT STREQUAL "" )
  string ( APPEND CPACK_RPM_PACKAGE_RELEASE "%{?dist}" )
endif()
set ( CPACK_RPM_FILE_NAME "RPM-DEFAULT" )
if ( DEFINED CPACK_PACKAGING_INSTALL_PREFIX )
    set ( CPACK_RPM_EXCLUDE_FROM_AUTO_FILELIST_ADDITION "${CPACK_PACKAGING_INSTALL_PREFIX}" )
endif ( )

# Enable Component Mode & install settings.
set(CPACK_DEB_COMPONENT_INSTALL ON)
set(CPACK_DEBIAN_RUNTIME_PACKAGE_NAME "${AQLPROFILE_NAME}")
set(CPACK_DEBIAN_ASAN_PACKAGE_NAME "${AQLPROFILE_NAME}-asan")
set(CPACK_DEBIAN_TESTS_PACKAGE_NAME "${AQLPROFILE_NAME}-tests")
set(CPACK_RPM_COMPONENT_INSTALL ON)
set(CPACK_RPM_RUNTIME_PACKAGE_NAME "${AQLPROFILE_NAME}")
set(CPACK_RPM_ASAN_PACKAGE_NAME "${AQLPROFILE_NAME}-asan")
set(CPACK_RPM_TESTS_PACKAGE_NAME "${AQLPROFILE_NAME}-tests")
# Add dependency on rocm-core if -DROCM_DEP_ROCMCORE=ON
if(ROCM_DEP_ROCMCORE)
    set(CPACK_DEBIAN_PACKAGE_DEPENDS "rocm-core")
    set(CPACK_RPM_PACKAGE_REQUIRES "rocm-core")
    set(CPACK_DEBIAN_RUNTIME_PACKAGE_DEPENDS "rocm-core")
    set(CPACK_RPM_RUNTIME_PACKAGE_REQUIRES "rocm-core")
    set(CPACK_DEBIAN_ASAN_PACKAGE_DEPENDS "rocm-core-asan")
    set(CPACK_RPM_ASAN_PACKAGE_REQUIRES "rocm-core-asan")
    set(CPACK_DEBIAN_TESTS_PACKAGE_DEPENDS "rocm-core")
    set(CPACK_RPM_TESTS_PACKAGE_REQUIRES "rocm-core")
endif()

include ( CPack )

cpack_add_component(
    runtime
    DISPLAY_NAME "Runtime"
    DESCRIPTION "Dynamic libraries for the AQLProfile")

cpack_add_component(
    asan
    DISPLAY_NAME "ASAN"
    DESCRIPTION "ASAN libraries for the AQLProfile"
    DEPENDS asan)

cpack_add_component(
    tests
    DISPLAY_NAME "Tests"
    DESCRIPTION "Tests for the AQLProfile"
    DEPENDS runtime)
