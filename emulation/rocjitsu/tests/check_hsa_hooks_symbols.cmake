# Copyright (c) 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

if(NOT NM OR NOT HOOK_LIBRARY OR NOT TEST_HOOK_LIBRARY)
    message(FATAL_ERROR "NM, HOOK_LIBRARY and TEST_HOOK_LIBRARY are required")
endif()

# Check the actual dynamic ABI, including accidental RTTI or template exports
# from the VM objects. Those objects may also be linked into the caller.
function(check_exports library)
    set(_expected ${ARGN})
    execute_process(
        COMMAND "${NM}" --dynamic --defined-only --format=posix "${library}"
        RESULT_VARIABLE _nm_result
        OUTPUT_VARIABLE _symbols
        ERROR_VARIABLE _nm_error
    )
    if(NOT _nm_result EQUAL 0)
        message(FATAL_ERROR "nm failed for ${library}: ${_nm_error}")
    endif()
    string(REGEX MATCHALL "[^\r\n]+" _lines "${_symbols}")
    foreach(_line IN LISTS _lines)
        string(REGEX REPLACE " .*" "" _symbol "${_line}")
        list(FIND _expected "${_symbol}" _index)
        if(_index EQUAL -1)
            message(FATAL_ERROR "${library} unexpectedly exports ${_symbol}")
        endif()
        list(REMOVE_ITEM _expected "${_symbol}")
    endforeach()
    if(_expected)
        message(FATAL_ERROR "${library} is missing exports: ${_expected}")
    endif()
endfunction()

check_exports("${HOOK_LIBRARY}" OnLoad OnUnload)
check_exports(
    "${TEST_HOOK_LIBRARY}"
    OnLoad
    OnUnload
    rj_hsa_dbt_set_topology_nodes_root_for_test
)
