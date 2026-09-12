# Copyright (c) 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

include(${CMAKE_CURRENT_LIST_DIR}/check_model_only_symbols.cmake)

# Static linking may omit analysis members not used by this executable. Audit
# the archive as well so a new dependency in any member cannot evade the check.
if(ANALYSIS_ARCHIVE)
    execute_process(
        COMMAND "${NM}" -C "${ANALYSIS_ARCHIVE}"
        RESULT_VARIABLE _archive_result
        OUTPUT_VARIABLE _archive_symbols
        ERROR_VARIABLE _archive_error
    )
    if(NOT _archive_result EQUAL 0)
        message(FATAL_ERROR "nm failed on analysis archive: ${_archive_error}")
    endif()
    foreach(
        _required
        "rocjitsu::LivenessAnalysis::"
        "rocjitsu::ExecMaskAnalysis::"
    )
        string(FIND "${_archive_symbols}" "${_required}" _match)
        if(_match EQUAL -1)
            message(
                FATAL_ERROR
                "analysis archive is missing symbol: ${_required}"
            )
        endif()
    endforeach()
    string(APPEND _symbols "\n${_archive_symbols}")
endif()

# An offline analysis executable must not acquire transformation or runtime
# state as new consumers are added to the shared object groups.
list(
    APPEND _forbidden_symbols
    "rocjitsu::BinaryTranslator::"
    "rocjitsu::Instrumentor::"
    "rocjitsu::Executable::"
    "rocjitsu::SimulatedKfd::"
    "rocjitsu::VirtualMachine::"
    "hsa_init"
    " OnLoad"
)
foreach(_forbidden IN LISTS _forbidden_symbols)
    string(FIND "${_symbols}" "${_forbidden}" _match)
    if(NOT _match EQUAL -1)
        message(
            FATAL_ERROR
            "static analysis contains runtime symbol: ${_forbidden}"
        )
    endif()
endforeach()

# Require actual code-object and CFG consumers so the boundary is not vacuous.
foreach(_required "rocjitsu::AmdGpuCodeObject::" "rocjitsu::BasicBlock::build(")
    string(FIND "${_symbols}" "${_required}" _match)
    if(_match EQUAL -1)
        message(FATAL_ERROR "static analysis is missing symbol: ${_required}")
    endif()
endforeach()
