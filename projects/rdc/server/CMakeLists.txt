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
message("----------SMI Lib Dir: " ${AMD_SMI_LIB_DIR})
message("----------SMI Inc Dir: " ${AMD_SMI_INCLUDE_DIR})
message("---------GRPC Root Dir: " ${GRPC_ROOT})
message("")

# Issue with ABI compatibility and gRPC.
# See:
# - https://github.com/llvm/llvm-project/issues/102443
# - https://github.com/abseil/abseil-cpp/issues/1747
if(
    CMAKE_CXX_COMPILER_ID STREQUAL "Clang"
    AND CMAKE_CXX_COMPILER_VERSION VERSION_GREATER_EQUAL 18.0.0
)
    message("Compiler is " ${CMAKE_CXX_COMPILER_ID} ":" ${CMAKE_CXX_COMPILER_VERSION})
    message("Applying abi patch...")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fclang-abi-compat=17")
endif()

set(SRC_DIR "src")
set(INC_DIR "include/rdc")

# TODO delete these if not used
file(GLOB PROTOBUF_GENERATED_INCLUDES "${PROTOB_OUT_DIR}/*.h")
file(GLOB PROTOBUF_GENERATED_SRCS "${PROTOB_OUT_DIR}/*.cc")

include_directories(
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    "${PROJECT_SOURCE_DIR}/include"
    "${GRPC_ROOT}/include"
    "${PROTOB_OUT_DIR}"
    "${AMD_SMI_INCLUDE_DIR}"
    "${PROJECT_SOURCE_DIR}"
)

set(SERVER_SRC_LIST
    "${COMMON_DIR}/rdc_capabilities.cc"
    "${COMMON_DIR}/rdc_utils.cc"
    "${PROTOBUF_GENERATED_SRCS}"
    "${SRC_DIR}/rdc_admin_service.cc"
    "${SRC_DIR}/rdc_api_service.cc"
    "${SRC_DIR}/rdc_server_main.cc"
)
message("SERVER_SRC_LIST=${SERVER_SRC_LIST}")

set(SERVER_DAEMON_EXE "rdcd")
set(SERVICE_FILE_NAME "rdc.service")
configure_file("rdc.service.in" "${PROJECT_BINARY_DIR}/${SERVICE_FILE_NAME}" @ONLY)
set(SERVER_CONFIG_FILE "rdc_options.conf")
configure_file("${SERVER_CONFIG_FILE}" "${PROJECT_BINARY_DIR}/${SERVER_CONFIG_FILE}" COPYONLY)

link_directories(${AMD_SMI_LIB_DIR})

add_executable(${SERVER_DAEMON_EXE} "${SERVER_SRC_LIST}")
target_compile_definitions(${SERVER_DAEMON_EXE} PRIVATE CURRENT_GIT_HASH=${GIT_HASH})

# help find librdc_bootstrap.so at runtime
set_target_properties(
    ${SERVER_DAEMON_EXE}
    PROPERTIES INSTALL_RPATH "\$ORIGIN/../lib:\$ORIGIN/../lib/rdc/grpc/lib"
)

target_link_libraries(
    ${SERVER_DAEMON_EXE}
    pthread
    rt
    gRPC::grpc++
    cap
    dl
    amd_smi
    rdc_bootstrap
)

install(
    TARGETS ${SERVER_DAEMON_EXE}
    PERMISSIONS
        OWNER_EXECUTE
        OWNER_READ
        OWNER_WRITE
        GROUP_READ
        GROUP_EXECUTE
        WORLD_READ
        WORLD_EXECUTE
    DESTINATION ${CMAKE_INSTALL_BINDIR}
    COMPONENT ${SERVER_COMPONENT}
)
install(
    FILES ${PROJECT_BINARY_DIR}/${SERVICE_FILE_NAME}
    DESTINATION ${CMAKE_INSTALL_LIBEXECDIR}/${RDC}
    COMPONENT ${SERVER_COMPONENT}
)
install(
    FILES ${PROJECT_BINARY_DIR}/${SERVER_CONFIG_FILE}
    DESTINATION ${CMAKE_INSTALL_DATAROOTDIR}/${RDC}/conf
    COMPONENT ${SERVER_COMPONENT}
)

message("&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&")
message("                    Finished Cmake Server                          ")
message("&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&")
