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
cmake_minimum_required(VERSION 3.5.0)

# ROCM_DIR should be passed in via command line; these will be used
# in sub-projects
set(RSMI_INC_DIR ${ROCM_DIR}/rocm_smi/include)
set(RSMI_LIB_DIR ${ROCM_DIR}/rocm_smi/lib)

if (NOT DEFINED GRPC_ROOT)
    set(GRPC_ROOT "/usr/local")
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

get_package_version_number("1.0.0" ${PKG_VERSION_GIT_TAG_PREFIX} GIT)
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
message("-----------GRPC ROOT: " ${GRPC_ROOT})
message("-----------ROCM_DIR : " ${ROCM_DIR})

# Create a configure file to get version info from within library
configure_file(
  "${PROJECT_SOURCE_DIR}/src/${RDC}64Config.in"
  "${PROJECT_SOURCE_DIR}/include/rdc/${RDC}64Config.h")

if (NOT DEFINED CPACK_PACKAGE_VENDOR)
    set(CPACK_PACKAGE_VENDOR "AMD")
endif()

if (NOT DEFINED CPACK_PACKAGE_CONTACT)
    set(CPACK_PACKAGE_CONTACT "Advanced Micro Devices Inc.")
endif()

if (NOT DEFINED CPACK_PACKAGE_DESCRIPTION_SUMMARY)
set(CPACK_PACKAGE_DESCRIPTION_SUMMARY
                 "Radeon Data Center Tools")
endif()

if (NOT RDC_PACKAGE)
   set(RDC_PACKAGE ${RDC})
endif()

## Verbose output.
set(CMAKE_VERBOSE_MAKEFILE on)

# Compile .proto files
file(GLOB PROTOB_DEF_SRC_FILES "protos/*.proto")
set(PROTOB_SRC_DIR "${PROJECT_SOURCE_DIR}/protos")
set(PROTOB_OUT_DIR "${CMAKE_CURRENT_BINARY_DIR}")
set(PROTOB_CMD "${GRPC_ROOT}/bin/protoc")
set(GRPC_PLUGIN "${GRPC_ROOT}/bin/grpc_cpp_plugin")
set(GRPC_LIB_DIR "${GRPC_ROOT}/lib")

set(ENV{LD_LIBRARY_PATH} ${GRPC_LIB_DIR}:${GRPC_LIB_DIR}64)
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
    message("....--plugin=\"${GRPC_PLUGIN}\" ${file}")

    execute_process(COMMAND
       ${PROTOB_CMD} --proto_path=${PROTOB_SRC_DIR}
                                                  --grpc_out=${PROTOB_OUT_DIR}
            --plugin=protoc-gen-grpc=${GRPC_PLUGIN} ${file}
            WORKING_DIRECTORY ${PROJECT_SOURCE_DIR}
            RESULT_VARIABLE PROTOB_RESULT
            OUTPUT_VARIABLE PROTOB_OUT_VAR)
    message("protoc command returned: ${PROTOB_RESULT}")

endforeach()

set(CMAKE_INSTALL_PREFIX ${RDC_SERVER_ROOT_PATH}
                                CACHE STRING "Default installation directory.")

set(CPACK_PACKAGING_INSTALL_PREFIX ${RDC_SERVER_ROOT_PATH}
                                     CACHE STRING "Default packaging prefix.")

# define all the install component labels to install
set(SERVER_COMPONENT "server")
set(CLIENT_COMPONENT "client")

add_subdirectory("server")
add_subdirectory("client")
add_subdirectory("rdc_libs")
add_subdirectory("example")
add_subdirectory("rdci")

set(CPACK_PACKAGE_NAME ${RDC_PACKAGE})
set(CPACK_PACKAGE_VERSION ${PKG_VERSION_STR})
set(CPACK_PROJECT_CONFIG_FILE ${CMAKE_SOURCE_DIR}/package.txt)

# This is used by configure_file for *_postinit.in
set(RDC_CLIENT_ROOT_PATH "${RDC_SERVER_ROOT_PATH}/${RDC_CLIENT_INSTALL_PREFIX}")
configure_file(
  "${PROJECT_SOURCE_DIR}/src/DEBIAN_postinst.in"
  "${PROJECT_SOURCE_DIR}/DEBIAN/postinst")
configure_file(
  "${PROJECT_SOURCE_DIR}/src/RPM_rpm_post.in"
  "${PROJECT_SOURCE_DIR}/RPM/rpm_post")

# TODO set(CPACK_DEBIAN_PACKAGE_HOMEPAGE <GITHUB URL> )
set(CPACK_DEBIAN_PACKAGE_CONTROL_EXTRA
    "${CMAKE_CURRENT_SOURCE_DIR}/DEBIAN/postinst;
                    ${CMAKE_CURRENT_SOURCE_DIR}/DEBIAN/prerm")
set(CPACK_DEBIAN_PACKAGE_VERSION ${PKG_VERSION_STR})

# RPM package specific variables
set(CPACK_RPM_PRE_INSTALL_SCRIPT_FILE
                   "${CMAKE_CURRENT_SOURCE_DIR}/RPM/rpm_post")
set(CPACK_RPM_POST_UNINSTALL_SCRIPT_FILE
                 "${CMAKE_CURRENT_SOURCE_DIR}/RPM/rpm_postun")
set(CPACK_RPM_PACKAGE_VERSION ${PKG_VERSION_STR})

## Add the packaging directives
include (CPack)

