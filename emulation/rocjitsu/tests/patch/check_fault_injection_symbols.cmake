# Copyright (c) 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

include(${CMAKE_CURRENT_LIST_DIR}/../check_model_only_symbols.cmake)
foreach(
    _forbidden
    "rocjitsu::consan::"
    "rocjitsu::validate_drop_barrier_candidate("
    "rocjitsu::validate_checked_text_candidate("
    "util::Sha256"
    "rocjitsu::VirtualMachine::"
    "rocjitsu::SimulatedKfd::"
    "hsa_init"
    "hsa_queue_create"
    " OnLoad"
)
    string(FIND "${_symbols}" "${_forbidden}" _match)
    if(NOT _match EQUAL -1)
        message(
            FATAL_ERROR
            "fault injection acquired runtime symbol: ${_forbidden}"
        )
    endif()
endforeach()
foreach(
    _required
    "rocjitsu::apply_drop_barrier("
    "rocjitsu::apply_checked_text_edits("
)
    string(FIND "${_symbols}" "${_required}" _match)
    if(_match EQUAL -1)
        message(FATAL_ERROR "missing mutation consumer: ${_required}")
    endif()
endforeach()
