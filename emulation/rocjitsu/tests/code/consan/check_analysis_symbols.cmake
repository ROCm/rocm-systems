# Copyright (c) 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

execute_process(
    COMMAND "${NM}" -C "${ANALYZER}"
    RESULT_VARIABLE _status
    OUTPUT_VARIABLE _symbols
    ERROR_VARIABLE _error
)
if(NOT _status EQUAL 0)
    message(FATAL_ERROR "nm failed: ${_error}")
endif()
# GCC annotates std::string-returning functions with [abi:cxx11] in nm -C.
# Normalize ABI tags before requiring the complete function name and '('.
string(REGEX REPLACE "\\[abi:[a-zA-Z0-9_]+\\]" "" _symbols "${_symbols}")
foreach(
    _required
    "rocjitsu::consan::analyze_capture("
    "rocjitsu::consan::has_findings("
    "rocjitsu::consan::render_analysis("
    "rocjitsu::consan::render_lds_race_analysis("
    "rocjitsu::consan::decode_capture("
)
    string(FIND "${_symbols}" "${_required}" _match)
    if(_match EQUAL -1)
        message(FATAL_ERROR "analyzer lacks ${_required}")
    endif()
endforeach()
foreach(
    _forbidden
    "rocjitsu::Decoder::"
    "rocjitsu::BasicBlock::"
    "rocjitsu::VirtualMachine::"
    "rocjitsu::AmdGpuCodeObject::"
    "rocjitsu::Instrumentor::"
    "rocjitsu::consan::prepare("
    "rocjitsu::consan::build_input_program("
    " OnLoad"
    " hsa_"
)
    string(FIND "${_symbols}" "${_forbidden}" _match)
    if(NOT _match EQUAL -1)
        message(FATAL_ERROR "analyzer acquired ${_forbidden}")
    endif()
endforeach()
