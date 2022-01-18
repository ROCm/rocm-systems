# Copyright (c) 2019 - present Advanced Micro Devices, Inc. All rights reserved.
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

#
# Minimum version of cmake required
#
cmake_minimum_required(VERSION 3.15)

# ROCM_DIR should be passed in via command line; these will be used
# in sub-projects
set(RSMI_INC_DIR ${ROCM_DIR}/rocm_smi/include)

# When cmake -DBUILD_STANDALONE=off, it will not build rdcd and rdci
# which requires the gRPC
option(BUILD_STANDALONE "Build targets for rdci and rdcd" ON)

# When cmake -DBUILD_RASLIB=off, it will not build the raslib
# which requires the ROCT-Thunk-Interface.
option(BUILD_RASLIB "Build targets for raslib" OFF)

# When cmake -DBUILD_ROCRTEST=off, it will not build the librdc_rocr.so
# which requires the Rocm run time.
option(BUILD_ROCRTEST "Build targets for librdc_rocr.so" ON)

if( NOT EXISTS "${CMAKE_SOURCE_DIR}/raslib/.git" AND BUILD_RASLIB)
    message( FATAL_ERROR "The git submodule raslib is not available. Please run
    git submodule update --init --recursive
    If you do not want to build raslib, use cmake -DBUILD_RASLIB=off"
    )
endif()

if (BUILD_RASLIB AND NOT DEFINED HSA_DIR)
    message( FATAL_ERROR "Please specify libhsakmt directory which is required by raslib
    cmake -DHSA_DIR=<libhsakmt directory>
    If you do not want to build raslib, use cmake -DBUILD_RASLIB=off"
    )
endif()


set(RSMI_LIB_DIR ${ROCM_DIR}/rocm_smi/lib)

if (NOT DEFINED GRPC_ROOT)
    set(GRPC_ROOT "/usr")
endif()
if (NOT DEFINED CPACK_PACKAGING_INSTALL_PREFIX)
    set(CPACK_PACKAGING_INSTALL_PREFIX "/"
                              CACHE STRING "Default packaging prefix.")
endif()
if (NOT DEFINED CMAKE_INSTALL_PREFIX)
set(CMAKE_INSTALL_PREFIX "/"
                         CACHE STRING "Default installation directory.")
endif()

if (NOT DEFINED RDC_SERVER_ROOT_PATH)
    set(RDC_SERVER_ROOT_PATH "/")
endif()
if (NOT DEFINED RDC_CLIENT_INSTALL_PREFIX)
    set(RDC_CLIENT_INSTALL_PREFIX "opt/rocm")
endif()

## Set default module path
if(NOT DEFINED CMAKE_MODULE_PATH)
    set(CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake_modules/")
endif()
## Include common cmake modules
include(utils)

set(RDC "rdc")

################# Determine the library version #########################
## Setup the package version based on git tags.
set(PKG_VERSION_GIT_TAG_PREFIX "rdc_pkg_ver")

# provide git to utilities
find_program (GIT NAMES git)

get_version_from_tag("1.0.0" ${PKG_VERSION_GIT_TAG_PREFIX} GIT)
# VERSION_* variables should be set by get_version_from_tag
message("Package version: ${PKG_VERSION_STR}")

set(${RDC}_VERSION_MAJOR "${VERSION_MAJOR}")
set(${RDC}_VERSION_MINOR "${VERSION_MINOR}")
set(${RDC}_VERSION_PATCH "0")
set(${RDC}_VERSION_BUILD "0")

## Define default variable and variables for the optional build target
##  RDC_lib-dev
set(SOURCE_DIR ${CMAKE_CURRENT_SOURCE_DIR}
                              CACHE STRING "Location of RDC source code.")
set(CPACK_GENERATOR "DEB;RPM"  CACHE STRING "Default packaging generators.")

project(${RDC})

set(RDC_SRC_ROOT "${PROJECT_SOURCE_DIR}")

message("Build Configuration:")

if(BUILD_STANDALONE)
  message("-----------GRPC ROOT: " ${GRPC_ROOT})
endif()
message("-----------ROCM_DIR : " ${ROCM_DIR})

## Add address sanitizer
if (${ADDRESS_SANITIZER})
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fsanitize=address -g")
    set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -fsanitize=address")
    set(CMAKE_SHARED_LINKER_FLAGS "${CMAKE_SHARED_LINKER_FLAGS} -shared-libasan")
    message(STATUS "ADDRESS_SANITIZE: CMAKE_CXX_FLAGS=: ${CMAKE_CXX_FLAGS}")
    message(STATUS "ADDRESS_SANITIZE: CMAKE_EXE_LINKER_FLAGS=: ${CMAKE_EXE_LINKER_FLAGS}")
endif ()

# Create a configure file to get version info from within library
configure_file(
  "${PROJECT_SOURCE_DIR}/src/${RDC}64Config.in"
  "${PROJECT_SOURCE_DIR}/include/rdc/${RDC}64Config.h")

if (NOT DEFINED CPACK_PACKAGE_VENDOR)
    set(CPACK_PACKAGE_VENDOR "Advanced Micro Devices, Inc.")
endif()

if (NOT DEFINED CPACK_PACKAGE_CONTACT)
    set(CPACK_PACKAGE_CONTACT "RDC Support <rdc.support@amd.com>")
endif()

if (NOT DEFINED CPACK_PACKAGE_DESCRIPTION_SUMMARY)
    set(CPACK_PACKAGE_DESCRIPTION_SUMMARY "AMD Radeon Data Center")
endif()

if (NOT DEFINED CPACK_PACKAGE_DESCRIPTION)
    set(CPACK_PACKAGE_DESCRIPTION "This package contains the AMD Radeon Data Center tools.")
endif()

if (NOT DEFINED CPACK_RESOURCE_FILE_LICENSE)
    set(CPACK_RESOURCE_FILE_LICENSE "${CMAKE_CURRENT_SOURCE_DIR}/LICENSE")
endif()

if (NOT RDC_PACKAGE)
   set(RDC_PACKAGE ${RDC})
endif()

## Verbose output.
set(CMAKE_VERBOSE_MAKEFILE on)
if(BUILD_STANDALONE)
   # Compile .proto files
   file(GLOB PROTOB_DEF_SRC_FILES "protos/*.proto")
   set(PROTOB_SRC_DIR "${PROJECT_SOURCE_DIR}/protos")
   set(PROTOB_OUT_DIR "${CMAKE_CURRENT_BINARY_DIR}")
   set(PROTOB_CMD "${GRPC_ROOT}/bin/protoc")
   set(GRPC_PLUGIN "${GRPC_ROOT}/bin/grpc_cpp_plugin")
   set(GRPC_LIB_DIR "${GRPC_ROOT}/lib")

   set(ENV{LD_LIBRARY_PATH} ${GRPC_LIB_DIR}:${GRPC_LIB_DIR}64:$ENV{LD_LIBRARY_PATH})
   message("LD_LIBRARY_PATH = $ENV{LD_LIBRARY_PATH}")
   foreach(file ${PROTOB_DEF_SRC_FILES})
       execute_process(COMMAND
               ${PROTOB_CMD} --proto_path=${PROTOB_SRC_DIR}
                                              --cpp_out=${PROTOB_OUT_DIR} ${file}
               WORKING_DIRECTORY ${PROJECT_SOURCE_DIR}
               RESULT_VARIABLE PROTOB_RESULT
               OUTPUT_VARIABLE PROTOB_OUT_VAR)
       message("protoc command returned: ${PROTOB_RESULT}")

   #    find_program (GRPC_PLUGIN NAMES grpc_cpp_plugin)
       message("GRPC_PLUGIN=${GRPC_PLUGIN})")
       message("protoc cmd:")
       message("  $ ${PROTOB_CMD} --proto_path=${PROTOB_SRC_DIR}")
       message("    --grpc_out=${PROTOB_OUT_DIR}")
       message("....--plugin=protoc-gen-grpc=\"${GRPC_PLUGIN}\" ${file}")

       execute_process(COMMAND
          ${PROTOB_CMD} --proto_path=${PROTOB_SRC_DIR}
                                                     --grpc_out=${PROTOB_OUT_DIR}
               --plugin=protoc-gen-grpc=${GRPC_PLUGIN} ${file}
               WORKING_DIRECTORY ${PROJECT_SOURCE_DIR}
               RESULT_VARIABLE PROTOB_RESULT
               OUTPUT_VARIABLE PROTOB_OUT_VAR)
       message("protoc command returned: ${PROTOB_RESULT}")

   endforeach()
endif()

set(CMAKE_INSTALL_PREFIX ${RDC_SERVER_ROOT_PATH}
                                CACHE STRING "Default installation directory.")

set(CPACK_PACKAGING_INSTALL_PREFIX ${RDC_SERVER_ROOT_PATH}
                                     CACHE STRING "Default packaging prefix.")

# define all the install component labels to install
set(SERVER_COMPONENT "server")
set(CLIENT_COMPONENT "client")

# Standalone only folders
if(BUILD_STANDALONE)
  install(DIRECTORY ${GRPC_ROOT}/lib
                USE_SOURCE_PERMISSIONS
                DESTINATION ${RDC_CLIENT_INSTALL_PREFIX}/${RDC}/grpc
                COMPONENT ${SERVER_COMPONENT})
# Also include dev setup for people do not want to build grpc
  install(DIRECTORY ${GRPC_ROOT}/bin
                USE_SOURCE_PERMISSIONS
                DESTINATION ${RDC_CLIENT_INSTALL_PREFIX}/${RDC}/grpc
                COMPONENT ${SERVER_COMPONENT})

  install(DIRECTORY ${GRPC_ROOT}/include
                USE_SOURCE_PERMISSIONS
                DESTINATION ${RDC_CLIENT_INSTALL_PREFIX}/${RDC}/grpc
                COMPONENT ${SERVER_COMPONENT})

  add_subdirectory("server")
  add_subdirectory("client")
  add_subdirectory("rdci")
endif()

# Raslib
if(BUILD_RASLIB)
   message("&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&")
   message("                    Build raslib")
   message("&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&")

   set(RDC_BUILD_DIR ${PROJECT_BINARY_DIR}/rdc_libs)
   add_subdirectory("raslib")
endif()

# Folders for both standalone and embedded
add_subdirectory("rdc_libs")
add_subdirectory("example")

# Create cmake target
# Add all targets to the build-tree export set
export(TARGETS ${BOOTSTRAP_LIB} ${RDC_LIB} ${RDCCLIENT_LIB}
  FILE "${PROJECT_BINARY_DIR}/rdc_libs.cmake")

# Export the package for use from the build-tree
# (this registers the build-tree with a global CMake-registry)
export(PACKAGE rdc)
# Create the rdc-config.cmake and rdc-config-version files
# ... for the build tree
set(CONF_INCLUDE_DIRS "${PROJECT_SOURCE_DIR}" "${PROJECT_BINARY_DIR}")
set(CONF_LIB_DIR  "${PROJECT_BINARY_DIR}")
set(CONF_LIBS "librdc_bootstrap.so")
configure_file(rdc-config.cmake.in
  "${PROJECT_BINARY_DIR}/rdc-config.cmake" @ONLY)
# ... for the install tree
set(CONF_INCLUDE_DIRS "${CMAKE_INSTALL_PREFIX}${RDC_CLIENT_INSTALL_PREFIX}/rdc/include")
set(CONF_LIB_DIR  "${CMAKE_INSTALL_PREFIX}${RDC_CLIENT_INSTALL_PREFIX}/rdc/lib")
set(CONF_LIBS "librdc_bootstrap.so")

configure_file(rdc-config.cmake.in
  "${PROJECT_BINARY_DIR}${CMAKE_FILES_DIRECTORY}/rdc-config.cmake" @ONLY)
# ... for both
configure_file(rdc-config-version.cmake.in
  "${PROJECT_BINARY_DIR}/rdc-config-version.cmake" @ONLY)
# Install the rdc-config.cmake and rdc-config-version.cmake
install(FILES
  "${PROJECT_BINARY_DIR}${CMAKE_FILES_DIRECTORY}/rdc-config.cmake"
  "${PROJECT_BINARY_DIR}/rdc-config-version.cmake"
  DESTINATION "${RDC_CLIENT_INSTALL_PREFIX}/${RDC}/lib/cmake" COMPONENT dev)
# Install the export set for use with the install-tree
install(EXPORT rdcTargets DESTINATION
  "${RDC_CLIENT_INSTALL_PREFIX}/${RDC}/lib/cmake" COMPONENT dev)


# License file
install( FILES ${CPACK_RESOURCE_FILE_LICENSE} DESTINATION share/doc/rdc RENAME LICENSE.txt)

# Python binding and example
install(DIRECTORY ${PROJECT_SOURCE_DIR}/python_binding
             DESTINATION  ${RDC_CLIENT_INSTALL_PREFIX}/${RDC}
                                         COMPONENT ${CLIENT_COMPONENT})
install(DIRECTORY ${PROJECT_SOURCE_DIR}/example
                 DESTINATION  ${RDC_CLIENT_INSTALL_PREFIX}/${RDC}
                                         COMPONENT ${CLIENT_COMPONENT})

# Prebuild packages to install
install(FILES  ${PROJECT_SOURCE_DIR}/ras_prebuild/librdc_ras.so
    DESTINATION  ${RDC_CLIENT_INSTALL_PREFIX}/${RDC}/lib
    COMPONENT ${CLIENT_COMPONENT})
install(DIRECTORY ${PROJECT_SOURCE_DIR}/ras_prebuild/config
     DESTINATION  ${RDC_CLIENT_INSTALL_PREFIX}/${RDC}/lib
     COMPONENT ${CLIENT_COMPONENT})
install(DIRECTORY ${PROJECT_SOURCE_DIR}/ras_prebuild/sp3
     DESTINATION  ${RDC_CLIENT_INSTALL_PREFIX}/${RDC}/lib
     COMPONENT ${CLIENT_COMPONENT})


set(CPACK_PACKAGE_NAME ${RDC_PACKAGE})
set(CPACK_PACKAGE_VERSION ${VERSION_STRING})

# This is used by configure_file for *_postinit.in
set(RDC_CLIENT_ROOT_PATH "${RDC_SERVER_ROOT_PATH}/${RDC_CLIENT_INSTALL_PREFIX}")

#Identify between SLES and Centos for setting symlink for rdc.service
#SLES need serice file in /usr/lib/systemd/system/rdc.service
#CENTOS/RHEL Require file in /lib/systemd/system/rdc.service
file(STRINGS /etc/os-release LINUX_DISTRO LIMIT_COUNT 1 REGEX "NAME=")
message ( "Using Linux Distro: ${LINUX_DISTRO}" )
if(LINUX_DISTRO MATCHES "SLES")
   set(DISTRO_ROOT "/usr/lib/systemd/system")
else()
   set(DISTRO_ROOT "/lib/systemd/system")
endif()

# 755 permissions
set(INST_SCR_PERM OWNER_READ OWNER_WRITE OWNER_EXECUTE GROUP_READ GROUP_EXECUTE WORLD_READ WORLD_EXECUTE)
configure_file(
  "${PROJECT_SOURCE_DIR}/src/DEBIAN_postinst.in"
  "${PROJECT_SOURCE_DIR}/DEBIAN/postinst" FILE_PERMISSIONS ${INST_SCR_PERM})
configure_file(
  "${PROJECT_SOURCE_DIR}/src/RPM_rpm_post.in"
  "${PROJECT_SOURCE_DIR}/RPM/rpm_post" FILE_PERMISSIONS ${INST_SCR_PERM})
configure_file(
  "${PROJECT_SOURCE_DIR}/src/RPM_postun.in"
  "${PROJECT_SOURCE_DIR}/RPM/rpm_postun" FILE_PERMISSIONS ${INST_SCR_PERM})

if ( DEFINED ENV{ROCM_LIBPATCH_VERSION} )
  set ( CPACK_PACKAGE_VERSION "${CPACK_PACKAGE_VERSION}.$ENV{ROCM_LIBPATCH_VERSION}" )
  message ( "Using CPACK_PACKAGE_VERSION ${CPACK_PACKAGE_VERSION}" )
else()
  set ( ROCM_LIBPATCH_VERSION )
endif()

# TODO set(CPACK_DEBIAN_PACKAGE_HOMEPAGE <GITHUB URL> )
# Debian package specific variables
if ( DEFINED ENV{CPACK_DEBIAN_PACKAGE_RELEASE} )
  set ( CPACK_DEBIAN_PACKAGE_RELEASE $ENV{CPACK_DEBIAN_PACKAGE_RELEASE} )
else()
  set ( CPACK_DEBIAN_PACKAGE_RELEASE "local" )
endif()
message ( "Using CPACK_DEBIAN_PACKAGE_RELEASE ${CPACK_DEBIAN_PACKAGE_RELEASE}" )
set ( CPACK_DEBIAN_FILE_NAME "DEB-DEFAULT" )

set(CPACK_DEBIAN_PACKAGE_CONTROL_EXTRA
    "${CMAKE_CURRENT_SOURCE_DIR}/DEBIAN/postinst;
                    ${CMAKE_CURRENT_SOURCE_DIR}/DEBIAN/prerm")
set(CPACK_DEBIAN_PACKAGE_DEPENDS "rocm-smi-lib, libc6, rocm-core")

# RPM package specific variables
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
if ( PROC_RESULT EQUAL "0" AND NOT EVAL_RESULT STREQUAL "" )
  string ( APPEND CPACK_RPM_PACKAGE_RELEASE "%{?dist}" )
endif()
set ( CPACK_RPM_FILE_NAME "RPM-DEFAULT" )

set(CPACK_RPM_PACKAGE_AUTOREQ 0)
set(CPACK_RPM_PACKAGE_AUTOPROV 0)
set(CPACK_RPM_PACKAGE_REQUIRES "rocm-smi-lib")
list(APPEND CPACK_RPM_EXCLUDE_FROM_AUTO_FILELIST_ADDITION "/lib")
list(APPEND CPACK_RPM_EXCLUDE_FROM_AUTO_FILELIST_ADDITION "/usr/sbin")
list(APPEND CPACK_RPM_EXCLUDE_FROM_AUTO_FILELIST_ADDITION "/lib/systemd")
list(APPEND CPACK_RPM_EXCLUDE_FROM_AUTO_FILELIST_ADDITION "/lib/systemd/system")
list(APPEND CPACK_RPM_EXCLUDE_FROM_AUTO_FILELIST_ADDITION "/usr")
list(APPEND CPACK_RPM_EXCLUDE_FROM_AUTO_FILELIST_ADDITION "/opt")

set(CPACK_RPM_PRE_INSTALL_SCRIPT_FILE
                   "${CMAKE_CURRENT_SOURCE_DIR}/RPM/rpm_post")
set(CPACK_RPM_POST_UNINSTALL_SCRIPT_FILE
                 "${CMAKE_CURRENT_SOURCE_DIR}/RPM/rpm_postun")
set(CPACK_RPM_PACKAGE_VERSION ${PKG_VERSION_STR})

## Add the packaging directives
# Remove dependency on rocm-core if -DROCM_DEP_ROCMCORE=ON not given to cmake
if(NOT ROCM_DEP_ROCMCORE)
    string(REGEX REPLACE ",? ?rocm-core" "" CPACK_RPM_PACKAGE_REQUIRES ${CPACK_RPM_PACKAGE_REQUIRES})
    string(REGEX REPLACE ",? ?rocm-core" "" CPACK_DEBIAN_PACKAGE_DEPENDS ${CPACK_DEBIAN_PACKAGE_DEPENDS})
endif()
include (CPack)
