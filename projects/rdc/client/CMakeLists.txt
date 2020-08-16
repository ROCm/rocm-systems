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

message("&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&")
message("                       Cmake Client Lib                            ")
message("&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&")

## Compiler flags
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Wextra -m64")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -msse -msse2 -std=c++11 ")
# Use this instead of above for 32 bit
# set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -m32")

if ("${CMAKE_BUILD_TYPE}" STREQUAL Release)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O2")
else ()
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -ggdb -O0 -DDEBUG")
endif ()


# Required Defines first:

message("")
message("Build Configuration:")
message("-------------BuildType: " ${CMAKE_BUILD_TYPE})
message("--------------Compiler: " ${CMAKE_CXX_COMPILER})
message("---------------Version: " ${CMAKE_CXX_COMPILER_VERSION})
message("----------Proj Src Dir: " ${PROJECT_SOURCE_DIR})
message("----------Proj Bld Dir: " ${PROJECT_BINARY_DIR})
message("----------Proj Lib Dir: " ${PROJECT_BINARY_DIR}/lib)
message("----------Proj Exe Dir: " ${PROJECT_BINARY_DIR}/bin)
message("----------RSMI Lib Dir: " ${RSMI_LIB_DIR})
message("----------RSMI Inc Dir: " ${RSMI_INC_DIR})
message("---------GRPC Root Dir: " ${GRPC_ROOT})
message("-Client Install Prefix: " ${RDC_CLIENT_INSTALL_PREFIX})
message("")

## Set default module path if not already set
if(NOT DEFINED CMAKE_MODULE_PATH)
    set(CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake_modules/")
endif()
## Include common cmake modules
# TODO see if we really need this in the end
include(utils)

set(CLIENT_LIB "rdc_client_smi")
set(RDC "rdc")
set(SRC_DIR "${PROJECT_SOURCE_DIR}/client/src")
set(RDC_CLIENT_INC_DIR "${PROJECT_SOURCE_DIR}/client/include/rdc")

################# Determine the library version #########################
## Setup the SO version based on git tags.
set(SO_VERSION_GIT_TAG_PREFIX "rdc_so_ver")

# provide git to utilities
find_program (GIT NAMES git)

# Debian package specific variables
# Set a default value for the package version
get_version_from_tag("1.0.0.0" ${SO_VERSION_GIT_TAG_PREFIX} GIT)

# VERSION_* variables should be set by get_version_from_tag
set(SO_VERSION_STRING "${VERSION_MAJOR}.${VERSION_MINOR}")
message("SOVERSION: ${SO_VERSION_STRING}")

## Define default variable and variables for the optional build target
##  RDC_lib-dev
set(SOURCE_DIR ${CMAKE_CURRENT_SOURCE_DIR}
                   CACHE STRING "Location of RDC client library source code.")
set(CPACK_GENERATOR "DEB;RPM"  CACHE STRING "Default packaging generators.")

if (NOT DEFINED CPACK_PACKAGE_VENDOR)
    set(CPACK_PACKAGE_VENDOR "Advanced Micro Devices, Inc.")
endif()

if (NOT DEFINED CPACK_PACKAGE_CONTACT)
    set(CPACK_PACKAGE_CONTACT "Advanced Micro Devices, Inc.")
endif()

if (NOT DEFINED CPACK_PACKAGE_DESCRIPTION_SUMMARY)
set(CPACK_PACKAGE_DESCRIPTION_SUMMARY
                 "Radeon Data Center Tools")
endif()

if (NOT RDC_PACKAGE)
   set(RDC_PACKAGE RDC_lib64)
endif()

set(CPACK_PACKAGE_FILE_NAME "${RDC_PACKAGE}-${PKG_VERSION_STR}")
## Verbose output.
set(CMAKE_VERBOSE_MAKEFILE on)

# TODO delete these if not used
file(GLOB PROTOBUF_GENERATED_INCLUDES "${PROTOB_OUT_DIR}/*.h")
file(GLOB PROTOBUF_GENERATED_SRCS "${PROTOB_OUT_DIR}/*.cc")

set(CLIENT_LIB_SRC_LIST "${SRC_DIR}/rdc_client.cc")
set(CLIENT_LIB_SRC_LIST ${CLIENT_LIB_SRC_LIST}  "${SRC_DIR}/rdc_client_main.cc")
set(CLIENT_LIB_SRC_LIST ${CLIENT_LIB_SRC_LIST}  "${SRC_DIR}/rdc_client_utils.cc")
set(CLIENT_LIB_SRC_LIST ${CLIENT_LIB_SRC_LIST} "${PROTOBUF_GENERATED_SRCS}")
set(CLIENT_LIB_SRC_LIST ${CLIENT_LIB_SRC_LIST}
                           "${PROJECT_SOURCE_DIR}/common/rdc_utils.cc")
message("CLIENT_LIB_SRC_LIST=${CLIENT_LIB_SRC_LIST}")

set(CLIENT_LIB_INC_LIST "${RDC_CLIENT_INC_DIR}/rdc_client.h")
set(CLIENT_LIB_INC_LIST ${CLIENT_LIB_INC_LIST}
                                      "${RDC_CLIENT_INC_DIR}/rdc_exception.h")
set(CLIENT_LIB_INC_LIST ${CLIENT_LIB_INC_LIST}
                                    "${RDC_CLIENT_INC_DIR}/rdc_client_main.h")
set(CLIENT_LIB_INC_LIST ${CLIENT_LIB_INC_LIST}
                                   "${PROJECT_SOURCE_DIR}/common/rdc_utils.h")

link_directories("${GRPC_ROOT}/lib" "${GRPC_ROOT}/lib64")
add_library(${CLIENT_LIB} SHARED ${CLIENT_LIB_SRC_LIST} ${CLIENT_LIB_INC_LIST})
target_link_libraries(${CLIENT_LIB} pthread rt grpc grpc++ grpc++_reflection
                                                                  dl protobuf)
target_include_directories(${CLIENT_LIB} PRIVATE
                                         "${PROJECT_SOURCE_DIR}"
                                         "${PROJECT_SOURCE_DIR}/include"
                                         "${CMAKE_CURRENT_SOURCE_DIR}/include"
                                         "${PROTOB_OUT_DIR}"
                                         "${GRPC_ROOT}/include"
                                         "${RSMI_INC_DIR}")
# TODO: set the properties for the library once we have one
## Set the VERSION and SOVERSION values
set_property(TARGET ${CLIENT_LIB} PROPERTY
             SOVERSION "${VERSION_MAJOR}")
set_property(TARGET ${CLIENT_LIB} PROPERTY
             VERSION "${SO_VERSION_STRING}")

## If the library is a release, strip the target library
if ("${CMAKE_BUILD_TYPE}" STREQUAL Release)
    add_custom_command(
          TARGET ${CLIENT_LIB}
          POST_BUILD COMMAND ${CMAKE_STRIP} lib${CLIENT_LIB}.so)
endif ()

## Add the install directives for the runtime library.
install(TARGETS ${CLIENT_LIB}
        LIBRARY DESTINATION  ${RDC_CLIENT_INSTALL_PREFIX}/${RDC}/lib
               COMPONENT ${CLIENT_COMPONENT})
install(DIRECTORY ${SOURCE_DIR}/authentication
             DESTINATION  ${RDC_CLIENT_INSTALL_PREFIX}/${RDC}
                                         COMPONENT ${CLIENT_COMPONENT})
install(DIRECTORY ${SOURCE_DIR}/python_binding
             DESTINATION  ${RDC_CLIENT_INSTALL_PREFIX}/${RDC}
                                         COMPONENT ${CLIENT_COMPONENT})

# Generate Doxygen documentation for client api manual
find_package(Doxygen)
find_package(LATEX COMPONENTS PDFLATEX)
if (DOXYGEN_FOUND AND LATEX_FOUND)
  set (RDC_MANUAL_NAME "RDC_API_Manual")
  message("Building ${RDC_MANUAL_NAME}.pdf")
  configure_file(${PROJECT_SOURCE_DIR}/docs/rdc_doxygen.cfg
                                   ${CMAKE_CURRENT_BINARY_DIR}/Doxyfile @ONLY)

  add_custom_command(OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/latex/refman.tex
        COMMAND ${DOXYGEN_EXECUTABLE} ${CMAKE_CURRENT_BINARY_DIR}/Doxyfile
        DEPENDS ${PROJECT_SOURCE_DIR}/docs/rdc_doxygen.cfg
                "${PROJECT_SOURCE_DIR}/include/rdc/rdc.h"
         WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR})
  add_custom_command(OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/latex/refman.pdf
         COMMAND make  > /dev/null
         COMMAND cp  ${CMAKE_CURRENT_BINARY_DIR}/latex/refman.pdf
                      ${PROJECT_SOURCE_DIR}/docs/${RDC_MANUAL_NAME}_new.pdf
         DEPENDS ${CMAKE_CURRENT_BINARY_DIR}/latex/refman.tex
         WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/latex)

  add_custom_target(docs DEPENDS ${CMAKE_CURRENT_BINARY_DIR}/latex/refman.pdf)

  add_dependencies(${CLIENT_LIB} docs)
  install(FILES ${CMAKE_CURRENT_BINARY_DIR}/latex/refman.pdf
   DESTINATION ${RDC_CLIENT_INSTALL_PREFIX}/${RDC}/docs/${RDC_MANUAL_NAME}.pdf
                COMPONENT ${CLIENT_COMPONENT})
  install(FILES ${PROJECT_SOURCE_DIR}/README.md
                       DESTINATION  ${RDC_CLIENT_INSTALL_PREFIX}/${RDC}/docs/
                          COMPONENT ${CLIENT_COMPONENT})
else()
  message("Doxygen or Latex is not found. Will not generate documents.")
endif(DOXYGEN_FOUND AND LATEX_FOUND)


message("&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&")
message("                    Finished Cmake Client Lib                      ")
message("&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&")
