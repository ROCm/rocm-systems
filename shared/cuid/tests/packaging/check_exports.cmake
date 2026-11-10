# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

# cmake -DNM=<nm> -DLIBRARY=<shared library> -P check_exports.cmake
# Fails when the library exports anything but its own probe function and the
# standard library's template instantiations, which libstdc++ declares with
# default visibility whatever the compiler is told.

execute_process(COMMAND ${NM} -D --defined-only ${LIBRARY} OUTPUT_VARIABLE _symbols RESULT_VARIABLE _result)
if(NOT _result EQUAL 0)
    message(FATAL_ERROR "${NM} failed on ${LIBRARY}")
endif()
string(REPLACE "\n" ";" _symbols "${_symbols}")
set(_unexpected "")
foreach(_line IN LISTS _symbols)
    if(_line MATCHES " [A-Za-z] (.+)$")
        set(_name "${CMAKE_MATCH_1}")
        if(
            NOT _name STREQUAL "cuid_exports_probe_version"
            AND NOT _name MATCHES "^_Z(ZN|N|T[ISV])?K?(St|S[absiod]|9__gnu_cxx)"
        )
            list(APPEND _unexpected "${_name}")
        endif()
    endif()
endforeach()
if(_unexpected)
    list(JOIN _unexpected "\n  " _text)
    message(FATAL_ERROR "${LIBRARY} exports symbols of libamdcuid_static.a:\n  ${_text}")
endif()
