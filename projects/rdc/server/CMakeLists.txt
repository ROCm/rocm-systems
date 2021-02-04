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
message("                             Cmake Server                          ")
message("&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&")

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
message("---Server Install Path: " ${RDC_SERVER_ROOT_PATH})
message("")

set(SOURCE_DIR ${CMAKE_CURRENT_SOURCE_DIR}
                   CACHE STRING "Location of RDC client library source code.")

## Compiler flags
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Wextra -m64")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -msse -msse2 -std=c++11 ")
# Use this instead of above for 32 bit
# set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -m32")

if ("${CMAKE_BUILD_TYPE}" STREQUAL Release)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O2")
else()
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -ggdb -O0 -DDEBUG")
endif()

set(SRC_DIR "src")
set(INC_DIR "include/${RDC}")

# TODO delete these if not used
file(GLOB PROTOBUF_GENERATED_INCLUDES "${PROTOB_OUT_DIR}/*.h")
file(GLOB PROTOBUF_GENERATED_SRCS "${PROTOB_OUT_DIR}/*.cc")

include_directories(${CMAKE_CURRENT_SOURCE_DIR}/include
                    "${PROJECT_SOURCE_DIR}/include"
                    "${GRPC_ROOT}/include"
                    "${PROTOB_OUT_DIR}"
                    "${RSMI_INC_DIR}"
                    "${RDC_SRC_ROOT}")

set(SERVER_SRC_LIST "${SRC_DIR}/rdc_rsmi_service.cc")
set(SERVER_SRC_LIST ${SERVER_SRC_LIST} "${SRC_DIR}/rdc_admin_service.cc")
set(SERVER_SRC_LIST ${SERVER_SRC_LIST} "${SRC_DIR}/rdc_api_service.cc")
set(SERVER_SRC_LIST ${SERVER_SRC_LIST} "${SRC_DIR}/rdc_server_main.cc")
set(SERVER_SRC_LIST ${SERVER_SRC_LIST} "${PROTOBUF_GENERATED_SRCS}")
set(SERVER_SRC_LIST ${SERVER_SRC_LIST} "${RDC_SRC_ROOT}/common/rdc_utils.cc")
set(SERVER_SRC_LIST ${SERVER_SRC_LIST} "${RDC_SRC_ROOT}/common/rdc_capabilities.cc")
message("SERVER_SRC_LIST=${SERVER_SRC_LIST}")

set(SERVER_DAEMON_EXE "rdcd")
configure_file("rdc.service.in" "${PROJECT_BINARY_DIR}/rdc.service" @ONLY)
set(SERVICE_FILE_NAME "rdc.service")

link_directories(${RSMI_LIB_DIR} "${GRPC_ROOT}/lib" "${GRPC_ROOT}/lib64")

## Set RUNPATH if ROCM_RPATH is defined and passed by the environment
## RUNPATH should help to run "rdcd" without the usage of ldconfig
if (DEFINED ENV{ROCM_RPATH})
 set (GRPC_RUNPATH "\$ORIGIN/../lib:\$ORIGIN/../grpc/lib:\$ORIGIN/../grpc/lib64")
 set (CMAKE_EXE_LINKER_FLAGS "-Wl,--enable-new-dtags -Wl,--no-as-needed -Wl,-z,origin")
 set (CMAKE_INSTALL_RPATH "${CMAKE_INSTALL_RPATH}:${GRPC_RUNPATH}")
 set (CMAKE_BUILD_WITH_INSTALL_RPATH TRUE)
 set (CMAKE_SKIP_BUILD_RPATH TRUE)

 # Run path can only include direct refer library, here is the indirect libs
 # required by gRPC
 set (RDCD_EXTRA_LIB absl_str_format_internal absl_strings absl_throw_delegate
   absl_bad_optional_access absl_strings absl_throw_delegate absl_int128
        absl_strings_internal absl_raw_logging_internal address_sorting
        gpr upb ssl crypto)
endif()

add_executable(${SERVER_DAEMON_EXE} "${SERVER_SRC_LIST}")
# target_include_directories(${SERVER_DAEMON_EXE} PUBLIC ${RSMI_INC_DIR})

target_link_libraries(${SERVER_DAEMON_EXE} ${RDCD_EXTRA_LIB} pthread rt grpc grpc++
                   cap grpc++_reflection dl protobuf rocm_smi64 rdc_bootstrap)

install(FILES ${CMAKE_CURRENT_BINARY_DIR}/${SERVER_DAEMON_EXE}
        PERMISSIONS OWNER_EXECUTE OWNER_READ OWNER_WRITE GROUP_READ
        GROUP_EXECUTE WORLD_READ WORLD_EXECUTE
        DESTINATION ${RDC_CLIENT_INSTALL_PREFIX}/rdc/bin
        COMPONENT ${SERVER_COMPONENT})
install(FILES ${PROJECT_BINARY_DIR}/${SERVICE_FILE_NAME}
               DESTINATION ${RDC_CLIENT_INSTALL_PREFIX}/rdc/lib
               COMPONENT ${SERVER_COMPONENT})

message("&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&")
message("                    Finished Cmake Server                          ")
message("&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&")
