# Copyright (c) 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

if(NOT NM OR NOT MODEL_BINARY)
    message(FATAL_ERROR "NM and MODEL_BINARY are required")
endif()

execute_process(
    COMMAND "${NM}" -C "${MODEL_BINARY}"
    RESULT_VARIABLE _nm_result
    OUTPUT_VARIABLE _symbols
    ERROR_VARIABLE _nm_error
)
if(NOT _nm_result EQUAL 0)
    message(FATAL_ERROR "nm failed: ${_nm_error}")
endif()

set(_forbidden_symbols
    "::execute_impl("
    "::execution_backend()"
    "rocjitsu::amdgpu::ComputeUnitCore"
    "ds_calculate_addresses("
    "flat_calculate_addresses("
    "mubuf_calculate_addresses("
)
foreach(_forbidden IN LISTS _forbidden_symbols)
    string(FIND "${_symbols}" "${_forbidden}" _match)
    if(NOT _match EQUAL -1)
        message(
            FATAL_ERROR
            "model-only binary contains forbidden symbol: ${_forbidden}"
        )
    endif()
endforeach()

# Execution-backed Operand helpers consistently end in `_exec`; match that
# method suffix specifically so analysis helpers such as
# `instruction_defines_exec` do not trip the model-only boundary.
string(REGEX MATCH "Operand::[^\n]*_exec\\(" _operand_exec_match "${_symbols}")
if(_operand_exec_match)
    message(
        FATAL_ERROR
        "model-only binary contains execution-backed Operand symbol: ${_operand_exec_match}"
    )
endif()

# Keep the denylist from passing vacuously if the model objects disappear from
# the final link. Aggregate consumers provide a comma-separated architecture
# list; existing cdna5-only checks retain their decoder-symbol assertion.
if(REQUIRED_MODEL_ARCHES)
    string(REPLACE "," ";" _required_arches "${REQUIRED_MODEL_ARCHES}")
    foreach(_arch IN LISTS _required_arches)
        set(_required_symbol
            "rocjitsu::${_arch}::create_model_target_decoder()"
        )
        string(FIND "${_symbols}" "${_required_symbol}" _match)
        if(_match EQUAL -1)
            message(
                FATAL_ERROR
                "model-only binary is missing model symbol: ${_required_symbol}"
            )
        endif()
    endforeach()
else()
    set(_required_symbol
        "rocjitsu::cdna5::Decoder::decode(unsigned int const*, util::DiagnosticEmitter const&)"
    )
    string(FIND "${_symbols}" "${_required_symbol}" _match)
    if(_match EQUAL -1)
        message(
            FATAL_ERROR
            "model-only binary is missing model symbol: ${_required_symbol}"
        )
    endif()
endif()
