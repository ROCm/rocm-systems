# Copyright (c) 2020 - present Advanced Micro Devices, Inc. All rights reserved.
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
message("                       Cmake RDC Lib                            ")
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
message("-----------BuildType: " ${CMAKE_BUILD_TYPE})
message("------------Compiler: " ${CMAKE_CXX_COMPILER})
message("-------------Version: " ${CMAKE_CXX_COMPILER_VERSION})
message("--------Proj Src Dir: " ${PROJECT_SOURCE_DIR})
message("--------Proj Bld Dir: " ${PROJECT_BINARY_DIR})
message("--------Proj Lib Dir: " ${PROJECT_BINARY_DIR}/lib)
message("--------Proj Exe Dir: " ${PROJECT_BINARY_DIR}/bin)
message("--------RSMI Lib Dir: " ${RSMI_LIB_DIR})
message("--------RSMI Inc Dir: " ${RSMI_INC_DIR})
message("")

## Set default module path if not already set
if(NOT DEFINED CMAKE_MODULE_PATH)
    set(CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake_modules/")
endif()


set(SRC_DIR "${PROJECT_SOURCE_DIR}/rdc_libs")
set(RDC_LIB_INC_DIR "${PROJECT_SOURCE_DIR}/include")
set(COMMON_DIR "${PROJECT_SOURCE_DIR}/common")

## Set RUNPATH if ROCM_RPATH is defined and passed by the environment
## RUNPATH should help to run "rdcd" without the usage of ldconfig
if (DEFINED ENV{ROCM_RPATH})
 set (LIB_RUNPATH "\$ORIGIN/../lib:\$ORIGIN/../lib64:\$ORIGIN/../grpc/lib:\$ORIGIN/../grpc/lib64")
 set (CMAKE_SHARED_LINKER_FLAGS "-Wl,--enable-new-dtags -Wl,--no-as-needed -Wl,-z,origin")
 set (CMAKE_INSTALL_RPATH "${CMAKE_INSTALL_RPATH}:${LIB_RUNPATH}")
 set (CMAKE_BUILD_WITH_INSTALL_RPATH TRUE)
 set (CMAKE_SKIP_BUILD_RPATH TRUE)
endif()

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
set(SOURCE_DIR ${CMAKE_CURRENT_SOURCE_DIR}
                   CACHE STRING "Location of RDC library source code.")
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
set ( CPACK_DEBIAN_PACKAGE_DEPENDS "rocm-core")
set ( CPACK_RPM_PACKAGE_REQUIRES "rocm-core")
## Verbose output.
set(CMAKE_VERBOSE_MAKEFILE on)

# librdc_bootstrap.so set up
set(BOOTSTRAP_LIB "rdc_bootstrap")
set(BOOTSTRAP_LIB_COMPONENT "lib${BOOTSTRAP_LIB}")
set(BOOTSTRAP_LIB_SRC_LIST "${SRC_DIR}/bootstrap/src/RdcBootStrap.cc")
set(BOOTSTRAP_LIB_SRC_LIST ${BOOTSTRAP_LIB_SRC_LIST} "${SRC_DIR}/bootstrap/src/RdcLogger.cc")
set(BOOTSTRAP_LIB_SRC_LIST ${BOOTSTRAP_LIB_SRC_LIST} "${SRC_DIR}/bootstrap/src/RdcLibraryLoader.cc")
set(BOOTSTRAP_LIB_SRC_LIST ${BOOTSTRAP_LIB_SRC_LIST} "${COMMON_DIR}/rdc_fields_supported.cc")
set(BOOTSTRAP_LIB_INC_LIST "${RDC_LIB_INC_DIR}/rdc/rdc.h")
set(BOOTSTRAP_LIB_INC_LIST ${BOOTSTRAP_LIB_INC_LIST} "${RDC_LIB_INC_DIR}/rdc_lib/rdc_common.h")
set(BOOTSTRAP_LIB_INC_LIST ${BOOTSTRAP_LIB_INC_LIST} "${RDC_LIB_INC_DIR}/rdc_lib/RdcLogger.h")
set(BOOTSTRAP_LIB_INC_LIST ${BOOTSTRAP_LIB_INC_LIST} "${RDC_LIB_INC_DIR}/rdc_lib/RdcHandler.h")
set(BOOTSTRAP_LIB_INC_LIST ${BOOTSTRAP_LIB_INC_LIST} "${RDC_LIB_INC_DIR}/rdc_lib/RdcLibraryLoader.h")
set(BOOTSTRAP_LIB_INC_LIST ${BOOTSTRAP_LIB_INC_LIST} "${COMMON_DIR}/rdc_fields_supported.h")
message("BOOTSTRAP_LIB_INC_LIST=${BOOTSTRAP_LIB_INC_LIST}")

add_library(${BOOTSTRAP_LIB} SHARED ${BOOTSTRAP_LIB_SRC_LIST} ${BOOTSTRAP_LIB_INC_LIST})
target_link_libraries(${BOOTSTRAP_LIB} pthread dl)
target_include_directories(${BOOTSTRAP_LIB} PRIVATE
                                         "${RSMI_INC_DIR}"
                                         "${ROCM_DIR}/include"
                                         "${PROJECT_SOURCE_DIR}"
                                         "${PROJECT_SOURCE_DIR}/include"
                                         "${COMMON_DIR}"
                                         "${CMAKE_CURRENT_SOURCE_DIR}/include")

# TODO: set the properties for the library once we have one
## Set the VERSION and SOVERSION values
set_property(TARGET ${BOOTSTRAP_LIB} PROPERTY
             SOVERSION "${VERSION_MAJOR}")
set_property(TARGET ${BOOTSTRAP_LIB} PROPERTY
             VERSION "${SO_VERSION_STRING}")

# librdc.so set up
set(RDC_LIB "rdc")
#set(RDC_LIB_COMPONENT "lib${RDC_LIB}")
set(RDC_LIB_SRC_LIST ${RDC_LIB_SRC_LIST} "${SRC_DIR}/rdc/src/RdcEmbeddedHandler.cc")
set(RDC_LIB_SRC_LIST ${RDC_LIB_SRC_LIST} "${SRC_DIR}/rdc/src/RdcMetricFetcherImpl.cc")
set(RDC_LIB_SRC_LIST ${RDC_LIB_SRC_LIST} "${SRC_DIR}/rdc/src/RdcGroupSettingsImpl.cc")
set(RDC_LIB_SRC_LIST ${RDC_LIB_SRC_LIST} "${SRC_DIR}/rdc/src/RdcCacheManagerImpl.cc")
set(RDC_LIB_SRC_LIST ${RDC_LIB_SRC_LIST} "${SRC_DIR}/rdc/src/RdcMetricsUpdaterImpl.cc")
set(RDC_LIB_SRC_LIST ${RDC_LIB_SRC_LIST} "${SRC_DIR}/rdc/src/RdcWatchTableImpl.cc")
set(RDC_LIB_SRC_LIST ${RDC_LIB_SRC_LIST} "${SRC_DIR}/rdc/src/RdcRasLib.cc")
set(RDC_LIB_SRC_LIST ${RDC_LIB_SRC_LIST} "${SRC_DIR}/rdc/src/RdcSmiLib.cc")
set(RDC_LIB_SRC_LIST ${RDC_LIB_SRC_LIST} "${SRC_DIR}/rdc/src/RdcTelemetryModule.cc")
set(RDC_LIB_SRC_LIST ${RDC_LIB_SRC_LIST} "${SRC_DIR}/rdc/src/RdcDiagnosticModule.cc")
set(RDC_LIB_SRC_LIST ${RDC_LIB_SRC_LIST} "${SRC_DIR}/rdc/src/RdcSmiDiagnosticImpl.cc")
set(RDC_LIB_SRC_LIST ${RDC_LIB_SRC_LIST} "${SRC_DIR}/rdc/src/RdcModuleMgrImpl.cc")
set(RDC_LIB_SRC_LIST ${RDC_LIB_SRC_LIST} "${SRC_DIR}/rdc/src/RdcNotificationImpl.cc")
set(RDC_LIB_SRC_LIST ${RDC_LIB_SRC_LIST} "${SRC_DIR}/rdc/src/RsmiUtils.cc")
set(RDC_LIB_SRC_LIST ${RDC_LIB_SRC_LIST} "${COMMON_DIR}/rdc_fields_supported.cc")
set(RDC_LIB_SRC_LIST ${RDC_LIB_SRC_LIST} "${COMMON_DIR}/rdc_capabilities.cc")

set(RDC_LIB_INC_LIST ${RDC_LIB_INC_LIST} "${RDC_LIB_INC_DIR}/rdc_lib/impl/RdcEmbeddedHandler.h")
set(RDC_LIB_INC_LIST ${RDC_LIB_INC_LIST} "${RDC_LIB_INC_DIR}/rdc_lib/RdcMetricFetcher.h")
set(RDC_LIB_INC_LIST ${RDC_LIB_INC_LIST} "${RDC_LIB_INC_DIR}/rdc_lib/impl/RdcMetricFetcherImpl.h")
set(RDC_LIB_INC_LIST ${RDC_LIB_INC_LIST} "${RDC_LIB_INC_DIR}/rdc_lib/RdcGroupSettings.h")
set(RDC_LIB_INC_LIST ${RDC_LIB_INC_LIST} "${RDC_LIB_INC_DIR}/rdc_lib/impl/RdcGroupSettingsImpl.h")
set(RDC_LIB_INC_LIST ${RDC_LIB_INC_LIST} "${RDC_LIB_INC_DIR}/rdc_lib/RdcCacheManager.h")
set(RDC_LIB_INC_LIST ${RDC_LIB_INC_LIST} "${RDC_LIB_INC_DIR}/rdc_lib/impl/RdcCacheManagerImpl.h")
set(RDC_LIB_INC_LIST ${RDC_LIB_INC_LIST} "${RDC_LIB_INC_DIR}/rdc_lib/RdcMetricsUpdater.h")
set(RDC_LIB_INC_LIST ${RDC_LIB_INC_LIST} "${RDC_LIB_INC_DIR}/rdc_lib/impl/RdcMetricsUpdaterImpl.h")
set(RDC_LIB_INC_LIST ${RDC_LIB_INC_LIST} "${RDC_LIB_INC_DIR}/rdc_lib/RdcWatchTable.h")
set(RDC_LIB_INC_LIST ${RDC_LIB_INC_LIST} "${RDC_LIB_INC_DIR}/rdc_lib/impl/RdcWatchTableImpl.h")
set(RDC_LIB_INC_LIST ${RDC_LIB_INC_LIST} "${RDC_LIB_INC_DIR}/rdc_lib/impl/RdcRasLib.h")
set(RDC_LIB_INC_LIST ${RDC_LIB_INC_LIST} "${RDC_LIB_INC_DIR}/rdc_lib/impl/RdcSmiLib.h")
set(RDC_LIB_INC_LIST ${RDC_LIB_INC_LIST} "${RDC_LIB_INC_DIR}/rdc_lib/impl/RdcModuleMgrImpl.h")
set(RDC_LIB_INC_LIST ${RDC_LIB_INC_LIST} "${RDC_LIB_INC_DIR}/rdc_lib/RdcModuleMgr.h")
set(RDC_LIB_INC_LIST ${RDC_LIB_INC_LIST} "${RDC_LIB_INC_DIR}/rdc_lib/RdcTelemetry.h")
set(RDC_LIB_INC_LIST ${RDC_LIB_INC_LIST} "${RDC_LIB_INC_DIR}/rdc_lib/impl/RdcTelemetryModule.h")
set(RDC_LIB_INC_LIST ${RDC_LIB_INC_LIST} "${RDC_LIB_INC_DIR}/rdc_lib/RdcDiagnostic.h")
set(RDC_LIB_INC_LIST ${RDC_LIB_INC_LIST} "${RDC_LIB_INC_DIR}/rdc_lib/RdcDiagnosticLibInterface.h")
set(RDC_LIB_INC_LIST ${RDC_LIB_INC_LIST} "${RDC_LIB_INC_DIR}/rdc_lib/impl/RdcDiagnosticModule.h")
set(RDC_LIB_INC_LIST ${RDC_LIB_INC_LIST} "${RDC_LIB_INC_DIR}/rdc_lib/impl/RdcSmiDiagnosticImpl.h")
set(RDC_LIB_INC_LIST ${RDC_LIB_INC_LIST} "${RDC_LIB_INC_DIR}/rdc_lib/RdcNotification.h")
set(RDC_LIB_INC_LIST ${RDC_LIB_INC_LIST} "${RDC_LIB_INC_DIR}/rdc_lib/impl/RdcNotificationImpl.h")
set(RDC_LIB_INC_LIST ${RDC_LIB_INC_LIST} "${RDC_LIB_INC_DIR}/rdc_lib/impl/RsmiUtils.h")
set(RDC_LIB_INC_LIST ${RDC_LIB_INC_LIST} "${COMMON_DIR}/rdc_fields_supported.h")
set(RDC_LIB_INC_LIST ${RDC_LIB_INC_LIST} "${COMMON_DIR}/rdc_capabilities.h")

message("RDC_LIB_INC_LIST=${RDC_LIB_INC_LIST}")

link_directories(${RSMI_LIB_DIR} "${GRPC_ROOT}/lib" "${GRPC_ROOT}/lib64" "${ROCM_DIR}/lib")
add_library(${RDC_LIB} SHARED ${RDC_LIB_SRC_LIST} ${RDC_LIB_INC_LIST})
target_link_libraries(${RDC_LIB} ${BOOTSTRAP_LIB} pthread rocm_smi64 cap)
target_include_directories(${RDC_LIB} PRIVATE
                                         "${PROJECT_SOURCE_DIR}"
                                         "${PROJECT_SOURCE_DIR}/include"
                                         "${CMAKE_CURRENT_SOURCE_DIR}/include"
                                         "${RSMI_INC_DIR}")

# TODO: set the properties for the library once we have one
## Set the VERSION and SOVERSION values
set_property(TARGET ${RDC_LIB} PROPERTY
             SOVERSION "${VERSION_MAJOR}")
set_property(TARGET ${RDC_LIB} PROPERTY
             VERSION "${SO_VERSION_STRING}")

if(BUILD_STANDALONE)
   # librdc_client.so set up
   file(GLOB PROTOBUF_GENERATED_INCLUDES "${PROTOB_OUT_DIR}/*.h")
   file(GLOB PROTOBUF_GENERATED_SRCS "${PROTOB_OUT_DIR}/*.cc")

   set(RDCCLIENT_LIB "rdc_client")
   set(RDCCLIENT_LIB_COMPONENT "lib${RDCCLIENT_LIB}")
   set(RDCCLIENT_LIB_SRC_LIST "${SRC_DIR}/rdc_client/src/RdcStandaloneHandler.cc")
   set(RDCCLIENT_LIB_SRC_LIST ${RDCCLIENT_LIB_SRC_LIST} "${PROTOBUF_GENERATED_SRCS}")

   set(RDCCLIENT_LIB_INC_LIST "${RDC_LIB_INC_DIR}/rdc/rdc.h")
   set(BRDCCLIENT_LIB_INC_LIST ${RDCCLIENT_LIB_INC_LIST} "${RDC_LIB_INC_DIR}/rdc_lib/rdc_common.h")
   set(RDCCLIENT_LIB_INC_LIST ${RDCCLIENT_LIB_INC_LIST} "${RDC_LIB_INC_DIR}/rdc_lib/RdcHandler.h")
   set(RDCCLIENT_LIB_INC_LIST ${RDCCLIENT_LIB_INC_LIST} "${RDC_LIB_INC_DIR}/rdc_lib/impl/RdcStandaloneHandler.h")

   message("RDCCLIENT_LIB_INC_LIST=${RDCCLIENT_LIB_INC_LIST}")

   add_library(${RDCCLIENT_LIB} SHARED ${RDCCLIENT_LIB_SRC_LIST} ${RDCCLIENT_LIB_INC_LIST})
   target_link_libraries(${RDCCLIENT_LIB}  ${BOOTSTRAP_LIB} pthread rt grpc grpc++ grpc++_reflection
                                                                  dl protobuf)
   target_include_directories(${RDCCLIENT_LIB} PRIVATE
                                            "${GRPC_ROOT}/include"
                                            "${PROJECT_SOURCE_DIR}"
                                            "${PROJECT_SOURCE_DIR}/include"
                                            "${PROTOB_OUT_DIR}"
                                            "${CMAKE_CURRENT_SOURCE_DIR}/include")

   # TODO: set the properties for the library once we have one
   ## Set the VERSION and SOVERSION values
   set_property(TARGET ${RDCCLIENT_LIB} PROPERTY
                SOVERSION "${VERSION_MAJOR}")
   set_property(TARGET ${RDCCLIENT_LIB} PROPERTY
                VERSION "${SO_VERSION_STRING}")
endif()

## If the library is a release, strip the target library
if ("${CMAKE_BUILD_TYPE}" STREQUAL Release)
    add_custom_command(
          TARGET ${BOOTSTRAP_LIB}
          POST_BUILD COMMAND ${CMAKE_STRIP} lib${BOOTSTRAP_LIB}.so)
    add_custom_command(
          TARGET ${RDC_LIB}
          POST_BUILD COMMAND ${CMAKE_STRIP} lib${RDC_LIB}.so)
    if(BUILD_STANDALONE)
       add_custom_command(
             TARGET ${RDCCLIENT_LIB}
             POST_BUILD COMMAND ${CMAKE_STRIP} lib${RDCCLIENT_LIB}.so)
    endif()
endif ()

# use the target_include_directories() command to specify the include directories for the target
target_include_directories(${BOOTSTRAP_LIB}
                           PUBLIC
                           "$<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>"
                           "$<INSTALL_INTERFACE:${RDC_CLIENT_INSTALL_PREFIX}/rdc/include>"
)

## Add the install directives for the runtime library.
if(BUILD_STANDALONE)
    install(TARGETS ${BOOTSTRAP_LIB} ${RDC_LIB} ${RDCCLIENT_LIB}
            EXPORT rdcTargets
            LIBRARY DESTINATION  ${RDC_CLIENT_INSTALL_PREFIX}/${RDC}/lib
                                            COMPONENT ${CLIENT_COMPONENT})
else()
    install(TARGETS ${BOOTSTRAP_LIB} ${RDC_LIB}
            EXPORT rdcTargets
            LIBRARY DESTINATION  ${RDC_CLIENT_INSTALL_PREFIX}/${RDC}/lib
                                            COMPONENT ${CLIENT_COMPONENT})
endif()

install(FILES ${SOURCE_DIR}/include/rdc/rdc.h
             DESTINATION  ${RDC_CLIENT_INSTALL_PREFIX}/${RDC}/include/rdc
                                         COMPONENT ${CLIENT_COMPONENT})

message("&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&")
message("                    Finished Cmake RDC Lib                      ")
message("&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&")


