# Copyright (c) 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

if(NOT NM OR NOT NARROW_LIBRARY OR NOT DOWNSTREAM_LIBRARY)
    message(
        FATAL_ERROR
        "NM, NARROW_LIBRARY and DOWNSTREAM_LIBRARY are required"
    )
endif()

foreach(_kind IN ITEMS narrow downstream)
    string(TOUPPER "${_kind}" _upper_kind)
    execute_process(
        COMMAND "${NM}" -D --defined-only "${${_upper_kind}_LIBRARY}"
        RESULT_VARIABLE _nm_result
        OUTPUT_VARIABLE _symbols
        ERROR_VARIABLE _nm_error
    )
    if(NOT _nm_result EQUAL 0)
        message(FATAL_ERROR "nm failed: ${_nm_error}")
    endif()

    string(REGEX MATCHALL "[^\n]+" _lines "${_symbols}")
    set(_exports)
    foreach(_line IN LISTS _lines)
        string(REGEX REPLACE "^.*[ \t]" "" _symbol "${_line}")
        list(APPEND _exports "${_symbol}")
    endforeach()
    list(SORT _exports)
    set(_expected "rj_test_${_kind}_has_target;rj_test_${_kind}_target_count")
    if(NOT "${_exports}" STREQUAL "${_expected}")
        message(
            FATAL_ERROR
            "${_kind} fixture exports unexpected symbols: ${_exports}"
        )
    endif()
endforeach()
