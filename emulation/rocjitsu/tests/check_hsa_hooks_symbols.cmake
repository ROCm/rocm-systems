# Copyright (c) 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

# HOOK_EXPORTS and TEST_HOOK_EXPORTS are comma-separated, because a semicolon
# list would be split into separate arguments on the test command line.
if(
    NOT NM
    OR NOT HOOK_LIBRARY
    OR NOT TEST_HOOK_LIBRARY
    OR NOT HOOK_EXPORTS
    OR NOT TEST_HOOK_EXPORTS
)
    message(
        FATAL_ERROR
        "NM, HOOK_LIBRARY, TEST_HOOK_LIBRARY, HOOK_EXPORTS and TEST_HOOK_EXPORTS are required"
    )
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

string(REPLACE "," ";" _hook_exports "${HOOK_EXPORTS}")
string(REPLACE "," ";" _test_hook_exports "${TEST_HOOK_EXPORTS}")
check_exports("${HOOK_LIBRARY}" ${_hook_exports})
check_exports("${TEST_HOOK_LIBRARY}" ${_test_hook_exports})
