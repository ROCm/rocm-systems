# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
#
# SPDX-License-Identifier: MIT

function(find_rpc_shared_headers output_var compiler)
  set(_rpc_candidates)
  if(RPC_SHARED_INCLUDE_DIR)
    list(APPEND _rpc_candidates "${RPC_SHARED_INCLUDE_DIR}")
  endif()
  if(ROCM_PATH)
    list(APPEND _rpc_candidates "${ROCM_PATH}/include")
  endif()

  execute_process(
    COMMAND "${compiler}" -print-resource-dir
    RESULT_VARIABLE _rpc_resource_result
    OUTPUT_VARIABLE _rpc_resource_dir
    OUTPUT_STRIP_TRAILING_WHITESPACE
    ERROR_QUIET)
  if(_rpc_resource_result EQUAL 0 AND _rpc_resource_dir)
    get_filename_component(_rpc_prefix "${_rpc_resource_dir}/../../.." ABSOLUTE)
    list(APPEND _rpc_candidates "${_rpc_prefix}/include")
  endif()

  get_filename_component(_rpc_compiler_bin "${compiler}" DIRECTORY)
  get_filename_component(_rpc_compiler_include "${_rpc_compiler_bin}/../include" ABSOLUTE)
  list(APPEND _rpc_candidates "${_rpc_compiler_include}")
  list(REMOVE_DUPLICATES _rpc_candidates)

  foreach(_rpc_candidate IN LISTS _rpc_candidates)
    if(EXISTS "${_rpc_candidate}/shared/rpc.h"
       AND EXISTS "${_rpc_candidate}/shared/rpc_opcodes.h"
       AND EXISTS "${_rpc_candidate}/shared/rpc_server.h"
       AND EXISTS "${_rpc_candidate}/shared/rpc_util.h")
      set(${output_var} "${_rpc_candidate}" PARENT_SCOPE)
      return()
    endif()
  endforeach()

  set(${output_var} "" PARENT_SCOPE)
endfunction()
