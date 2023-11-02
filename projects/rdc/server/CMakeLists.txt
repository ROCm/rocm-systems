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
message("")

## Compiler flags
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Wextra -m64")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -msse -msse2")
# Use this instead of above for 32 bit
# set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -m32")

if("${CMAKE_BUILD_TYPE}" STREQUAL Release)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O2 -s")
else()
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -ggdb -O0 -DDEBUG")
endif()

set(SRC_DIR "src")
set(INC_DIR "include/rdc")

# TODO delete these if not used
file(GLOB PROTOBUF_GENERATED_INCLUDES "${PROTOB_OUT_DIR}/*.h")
file(GLOB PROTOBUF_GENERATED_SRCS "${PROTOB_OUT_DIR}/*.cc")

include_directories(${CMAKE_CURRENT_SOURCE_DIR}/include
    "${PROJECT_SOURCE_DIR}/include"
    "${GRPC_ROOT}/include"
    "${PROTOB_OUT_DIR}"
    "${RSMI_INC_DIR}"
    "${PROJECT_SOURCE_DIR}")

set(SERVER_SRC_LIST
    "${COMMON_DIR}/rdc_capabilities.cc"
    "${COMMON_DIR}/rdc_utils.cc"
    "${PROTOBUF_GENERATED_SRCS}"
    "${SRC_DIR}/rdc_admin_service.cc"
    "${SRC_DIR}/rdc_api_service.cc"
    "${SRC_DIR}/rdc_rsmi_service.cc"
    "${SRC_DIR}/rdc_server_main.cc")
message("SERVER_SRC_LIST=${SERVER_SRC_LIST}")

set(SERVER_DAEMON_EXE "rdcd")
configure_file("rdc.service.in" "${PROJECT_BINARY_DIR}/rdc.service" @ONLY)
set(SERVICE_FILE_NAME "rdc.service")

link_directories(${RSMI_LIB_DIR})

add_executable(${SERVER_DAEMON_EXE} "${SERVER_SRC_LIST}")

target_link_libraries(${SERVER_DAEMON_EXE} pthread rt gRPC::grpc++
    cap dl rocm_smi64 rdc_bootstrap)

install(FILES ${CMAKE_CURRENT_BINARY_DIR}/${SERVER_DAEMON_EXE}
    PERMISSIONS OWNER_EXECUTE OWNER_READ OWNER_WRITE GROUP_READ
    GROUP_EXECUTE WORLD_READ WORLD_EXECUTE
    DESTINATION ${CMAKE_INSTALL_BINDIR}
    COMPONENT ${SERVER_COMPONENT})
install(FILES ${PROJECT_BINARY_DIR}/${SERVICE_FILE_NAME}
    DESTINATION ${CMAKE_INSTALL_LIBEXECDIR}/${RDC}
    COMPONENT ${SERVER_COMPONENT})

message("&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&")
message("                    Finished Cmake Server                          ")
message("&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&")
