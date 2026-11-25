# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
#
# SPDX-License-Identifier: MIT

set(RPC_SHARED_INCLUDE_DIR "" CACHE PATH
    "Directory containing the LLVM libc shared RPC headers")
set(_rpc_shared_headers_cmake "${CMAKE_CURRENT_LIST_DIR}/../../../../cmake/RPCSharedHeaders.cmake")
if(NOT EXISTS "${_rpc_shared_headers_cmake}")
  set(_rpc_shared_headers_cmake "${CMAKE_CURRENT_LIST_DIR}/../../cmake/RPCSharedHeaders.cmake")
endif()
include("${_rpc_shared_headers_cmake}")
unset(_rpc_shared_headers_cmake)
set(_rpc_compiler "${CMAKE_HIP_COMPILER}")
if(NOT _rpc_compiler)
  set(_rpc_compiler "${CMAKE_CXX_COMPILER}")
endif()
find_rpc_shared_headers(_rpc_include "${_rpc_compiler}")

if(NOT _rpc_include)
  message(STATUS "RPC: required shared headers not found for this compiler; skipping RPC tests")
  return()
endif()
message(STATUS "RPC: using shared headers from ${_rpc_include}")

set(TEST_SRC
    rpc_ping.cc)

hip_add_exe_to_target(NAME RpcTest
                      TEST_SRC ${TEST_SRC}
                      TEST_TARGET_NAME build_tests
                      PROPERTY CXX_STANDARD 17)
target_include_directories(RpcTest PRIVATE "${_rpc_include}")

hip_add_exe_to_target(NAME RpcHostcallTest
                      TEST_SRC rpc_hostcall_coexistence.cc
                      TEST_TARGET_NAME build_tests
                      PROPERTY CXX_STANDARD 17)
target_include_directories(RpcHostcallTest PRIVATE "${_rpc_include}")
