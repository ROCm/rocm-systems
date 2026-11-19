# Copyright (c) 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

cmake_minimum_required(VERSION 3.22)

# Inspect full local symbols as well as dynamic exports. Hidden
# compiler or diagnosis code is still an unwanted dependency.
foreach(
    _input
    "${RUNTIME_CORE}"
    "${IMAGE_ARCHIVE}"
    "${PLUGIN}"
    "${TEST_PLUGIN}"
)
    execute_process(
        COMMAND "${NM}" -C "${_input}"
        RESULT_VARIABLE _status
        OUTPUT_VARIABLE _symbols
        ERROR_VARIABLE _error
    )
    if(NOT _status EQUAL 0)
        message(FATAL_ERROR "nm failed on ${_input}: ${_error}")
    endif()
    foreach(
        _forbidden
        "rocjitsu::Decoder::"
        "rocjitsu::BasicBlock::"
        "rocjitsu::AmdGpuCodeObject::"
        "rocjitsu::Instrumentor::"
        "rocjitsu::BinaryTranslator::"
        "rocjitsu::VirtualMachine::"
        "rocjitsu::serialize_kernarg_extension_metadata("
        "rocjitsu::parse_kernarg_extension_metadata("
        "rocjitsu::consan::prepare("
        "rocjitsu::consan::analyze_original("
        "rocjitsu::consan::analyze_capture("
    )
        string(FIND "${_symbols}" "${_forbidden}" _match)
        if(NOT _match EQUAL -1)
            message(FATAL_ERROR "${_input} acquired ${_forbidden}")
        endif()
    endforeach()
endforeach()
foreach(_variant PLUGIN TEST_PLUGIN)
    set(_plugin "${${_variant}}")
    execute_process(
        COMMAND "${NM}" -D --defined-only --format=posix "${_plugin}"
        RESULT_VARIABLE _status
        OUTPUT_VARIABLE _exports
        ERROR_VARIABLE _error
    )
    if(NOT _status EQUAL 0)
        message(FATAL_ERROR "dynamic nm failed: ${_error}")
    endif()
    set(_expected OnLoad OnUnload rj_consan_drain rj_consan_finish)
    if(_variant STREQUAL "TEST_PLUGIN")
        list(APPEND _expected rj_consan_get_statistics)
    endif()
    string(REGEX MATCHALL "[^\n]+" _lines "${_exports}")
    set(_actual)
    foreach(_line IN LISTS _lines)
        string(REGEX REPLACE " .*" "" _symbol "${_line}")
        list(APPEND _actual "${_symbol}")
    endforeach()
    list(SORT _expected)
    list(SORT _actual)
    if(NOT _actual STREQUAL _expected)
        message(
            FATAL_ERROR
            "${_plugin} exports ${_actual}; expected ${_expected}"
        )
    endif()
endforeach()
