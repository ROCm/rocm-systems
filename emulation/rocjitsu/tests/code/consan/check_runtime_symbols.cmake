# Copyright (c) 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

# Inspect full local symbols, not merely the five dynamic exports. Hidden
# compiler or diagnosis code is still an unwanted dependency.
foreach(_input "${RUNTIME_CORE}" "${IMAGE_ARCHIVE}" "${PLUGIN}")
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
execute_process(
    COMMAND "${NM}" -D --defined-only "${PLUGIN}"
    RESULT_VARIABLE _status
    OUTPUT_VARIABLE _exports
    ERROR_VARIABLE _error
)
if(NOT _status EQUAL 0)
    message(FATAL_ERROR "dynamic nm failed: ${_error}")
endif()
foreach(
    _required
    OnLoad
    OnUnload
    rj_consan_drain
    rj_consan_finish
    rj_consan_get_statistics
)
    string(FIND "${_exports}" " T ${_required}\n" _match)
    if(_match EQUAL -1)
        message(FATAL_ERROR "plugin lacks ${_required}")
    endif()
endforeach()
