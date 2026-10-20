# Copyright (c) 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

foreach(_input "${CONTRACT_ARCHIVE}" "${CONTRACT_BINARY}")
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
        _required
        "rocjitsu::consan::encode_manifest("
        "rocjitsu::consan::decode_manifest("
        "rocjitsu::consan::match_images("
        "rocjitsu::consan::plan_launch("
    )
        string(FIND "${_symbols}" "${_required}" _match)
        if(_match EQUAL -1)
            message(FATAL_ERROR "${_input} lacks ${_required}")
        endif()
    endforeach()
    foreach(
        _forbidden
        "rocjitsu::Decoder::"
        "rocjitsu::BasicBlock::"
        "rocjitsu::AmdGpuCodeObject::"
        "rocjitsu::Instrumentor::"
        "rocjitsu::BinaryTranslator::"
        "rocjitsu::VirtualMachine::"
        "rocjitsu::consan::prepare"
        "rocjitsu::consan::analyze"
        "hsa_"
        " OnLoad"
    )
        string(FIND "${_symbols}" "${_forbidden}" _match)
        if(NOT _match EQUAL -1)
            message(FATAL_ERROR "${_input} acquired ${_forbidden}")
        endif()
    endforeach()
endforeach()
