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
set(RSMI_INC_DIR ${ROCM_DIR}/include)
set(RSMI_LIB_DIR ${ROCM_DIR}/lib)

## Set default module path
if(NOT DEFINED CMAKE_MODULE_PATH)
    set(CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake_modules/")
endif()
## Include common cmake modules
include(utils)

set(RDC "rdc")
set(RDC_TARGET "${RDC}64")

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
set(CMAKE_INSTALL_PREFIX "/"
                               CACHE STRING "Default installation directory.")
set(CPACK_PACKAGING_INSTALL_PREFIX "/"
                                     CACHE STRING "Default packaging prefix.")
set(CPACK_GENERATOR "DEB;RPM"  CACHE STRING "Default packaging generators.")

project(${RDC_TARGET})

# Create a configure file to get version info from within library
configure_file(
  "${PROJECT_SOURCE_DIR}/src/${RDC_TARGET}Config.in"
  "${PROJECT_SOURCE_DIR}/include/rdc/${RDC_TARGET}Config.h")

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

set(CPACK_PACKAGE_FILE_NAME "${RDC_PACKAGE}-${PKG_VERSION_STR}")
## Verbose output.
set(CMAKE_VERBOSE_MAKEFILE on)

# Compile .proto files
file(GLOB PROTOB_DEF_SRC_FILES "protos/*.proto")
set(PROTOB_SRC_DIR "${PROJECT_SOURCE_DIR}/protos")
set(PROTOB_OUT_DIR "${CMAKE_CURRENT_BINARY_DIR}")
set(PROTOB_CMD "protoc")

foreach(file ${PROTOB_DEF_SRC_FILES})
    execute_process(COMMAND
     protoc --proto_path=${PROTOB_SRC_DIR} --cpp_out=${PROTOB_OUT_DIR} ${file} 
            WORKING_DIRECTORY ${PROJECT_SOURCE_DIR}
            RESULT_VARIABLE PROTOB_RESULT
            OUTPUT_VARIABLE PROTOB_OUT_VAR)
    message("protoc command returned: ${PROTOB_RESULT}")

    find_program (GRPC_PLUGIN NAMES grpc_cpp_plugin)
    message("GRPC_PLUGIN=${GRPC_PLUGIN})")
    message("protoc cmd:")
    message("  $ protoc --proto_path=${PROTOB_SRC_DIR}")
    message("    --grpc_out=${PROTOB_OUT_DIR}")
    message("....--plugin=\"${GRPC_PLUGIN}\" ${file}")

    execute_process(COMMAND
            protoc --proto_path=${PROTOB_SRC_DIR} --grpc_out=${PROTOB_OUT_DIR}
            --plugin=protoc-gen-grpc=${GRPC_PLUGIN} ${file} 
            WORKING_DIRECTORY ${PROJECT_SOURCE_DIR}
            RESULT_VARIABLE PROTOB_RESULT
            OUTPUT_VARIABLE PROTOB_OUT_VAR)
    message("protoc command returned: ${PROTOB_RESULT}")

endforeach()

add_subdirectory("server")
add_subdirectory("client")
add_subdirectory("tests/example")

# TODO set(CPACK_DEBIAN_PACKAGE_HOMEPAGE <GITHUB URL> )
set(CPACK_DEBIAN_PACKAGE_CONTROL_EXTRA
    "${CMAKE_CURRENT_SOURCE_DIR}/DEBIAN/postinst;
                                    ${CMAKE_CURRENT_SOURCE_DIR}/DEBIAN/prerm")
set (CPACK_DEBIAN_PACKAGE_NAME  ${RDC_PACKAGE})
set (CPACK_DEBIAN_PACKAGE_VERSION ${PKG_VERSION_STR})

# RPM package specific variables
set(CPACK_RPM_PRE_INSTALL_SCRIPT_FILE
                                   "${CMAKE_CURRENT_SOURCE_DIR}/RPM/rpm_post")
set(CPACK_RPM_POST_UNINSTALL_SCRIPT_FILE
                                 "${CMAKE_CURRENT_SOURCE_DIR}/RPM/rpm_postun")
set (CPACK_RPM_PACKAGE_NAME  ${RDC_PACKAGE})
set (CPACK_RPM_PACKAGE_VERSION ${PKG_VERSION_STR})

# Generate Doxygen documentation
find_package(Doxygen)
if (DOXYGEN_FOUND)
  configure_file(${CMAKE_CURRENT_SOURCE_DIR}/docs/rdc_doxygen.cfg
                                   ${CMAKE_CURRENT_BINARY_DIR}/Doxyfile @ONLY)
  add_custom_target(doc
         ${DOXYGEN_EXECUTABLE} ${CMAKE_CURRENT_BINARY_DIR}/Doxyfile
         WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
                  COMMENT "Generating API documentation with Doxygen" VERBATIM)
else()
  message("Doxygen is not found. Will not generate documents.")
endif(DOXYGEN_FOUND)

## Add the packaging directives
include (CPack)
