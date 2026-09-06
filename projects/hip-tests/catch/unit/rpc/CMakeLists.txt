# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
#
# SPDX-License-Identifier: MIT

# The RPC headers come from the installed 'clang' compiler.
set(_rpc_candidates "")

# Explicit override wins.
if(RPC_SHARED_INCLUDE_DIR)
  list(APPEND _rpc_candidates "${RPC_SHARED_INCLUDE_DIR}")
endif()

# Preferred: <prefix>/include derived from the compiler's resource dir.
execute_process(
  COMMAND "${CMAKE_CXX_COMPILER}" -print-resource-dir
  OUTPUT_VARIABLE _rpc_resource_dir
  OUTPUT_STRIP_TRAILING_WHITESPACE
  ERROR_QUIET)
if(_rpc_resource_dir)
  get_filename_component(_rpc_prefix "${_rpc_resource_dir}/../../.." REALPATH)
  list(APPEND _rpc_candidates "${_rpc_prefix}/include")
endif()

# Fallback: <compiler_bin>/../include.
get_filename_component(_rpc_bin_dir "${CMAKE_CXX_COMPILER}" DIRECTORY)
get_filename_component(_rpc_bin_inc "${_rpc_bin_dir}/../include" REALPATH)
list(APPEND _rpc_candidates "${_rpc_bin_inc}")

set(_rpc_include "")
foreach(_cand IN LISTS _rpc_candidates)
  if(_cand AND EXISTS "${_cand}/shared/rpc.h")
    set(_rpc_include "${_cand}")
    break()
  endif()
endforeach()

if(NOT _rpc_include)
  message(STATUS "RPC: <shared/rpc.h> not found for this compiler; skipping RPC tests")
  return()
endif()
message(STATUS "RPC: using shared headers from ${_rpc_include}")

set(TEST_SRC
    rpc_ping.cc)

set_source_files_properties(rpc_ping.cc PROPERTIES
    COMPILE_OPTIONS "-I${_rpc_include}")

hip_add_exe_to_target(NAME RpcTest
                      TEST_SRC ${TEST_SRC}
                      TEST_TARGET_NAME build_tests
                      PROPERTY CXX_STANDARD 17)
