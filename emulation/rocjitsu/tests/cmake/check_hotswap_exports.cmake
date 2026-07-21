# Copyright (c) 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

cmake_minimum_required(VERSION 3.22)

foreach(_required NM_TOOL HOTSWAP_LIBRARY SIMULATOR_LIBRARY)
    if(NOT DEFINED ${_required} OR "${${_required}}" STREQUAL "")
        message(FATAL_ERROR "${_required} is required")
    endif()
endforeach()

function(read_dynamic_definitions library output_var)
    execute_process(
        COMMAND "${NM_TOOL}" -D --defined-only --format=posix "${library}"
        RESULT_VARIABLE _result
        OUTPUT_VARIABLE _output
        ERROR_VARIABLE _error
    )
    if(NOT _result EQUAL 0)
        message(
            FATAL_ERROR
            "Could not inspect ${library} with ${NM_TOOL}: ${_error}"
        )
    endif()

    string(REGEX MATCHALL "[^\r\n]+" _lines "${_output}")
    set(_symbols)
    foreach(_line IN LISTS _lines)
        string(REGEX MATCH "^[^ \t]+" _symbol "${_line}")
        string(REGEX REPLACE "@.*$" "" _symbol "${_symbol}")
        if(NOT _symbol STREQUAL "")
            list(APPEND _symbols "${_symbol}")
        endif()
    endforeach()
    list(REMOVE_DUPLICATES _symbols)
    list(SORT _symbols)
    set(${output_var} "${_symbols}" PARENT_SCOPE)
endfunction()

set(_expected_hotswap_symbols
    amd_comgr_create_data
    amd_comgr_get_data
    amd_comgr_get_data_isa_name
    amd_comgr_hotswap_rewrite
    amd_comgr_release_data
    amd_comgr_set_data
)
list(SORT _expected_hotswap_symbols)

read_dynamic_definitions("${HOTSWAP_LIBRARY}" _hotswap_symbols)
if(NOT _hotswap_symbols STREQUAL _expected_hotswap_symbols)
    message(
        FATAL_ERROR
        "Unexpected HotSwap dynamic exports.\n"
        "Expected: ${_expected_hotswap_symbols}\n"
        "Actual:   ${_hotswap_symbols}"
    )
endif()

read_dynamic_definitions("${SIMULATOR_LIBRARY}" _simulator_symbols)
set(_overlap)
foreach(_symbol IN LISTS _hotswap_symbols)
    if(_symbol IN_LIST _simulator_symbols)
        list(APPEND _overlap "${_symbol}")
    endif()
endforeach()
if(_overlap)
    message(FATAL_ERROR "HotSwap and simulator DSOs both define: ${_overlap}")
endif()

message(
    STATUS
    "HotSwap exports are limited to its six COMGR entry points and do not overlap the simulator DSO"
)
