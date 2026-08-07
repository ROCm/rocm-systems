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

# Generated gfx1250 execution helpers use an `_exec` suffix. Qualify the
# namespace so unrelated analysis helpers such as writes_exec remain valid.
set(_forbidden_symbol_pattern "rocjitsu::gfx1250::[A-Za-z0-9_:]+_exec\\(")
string(REGEX MATCH "${_forbidden_symbol_pattern}" _match "${_symbols}")
if(_match)
    message(
        FATAL_ERROR
        "model-only binary contains forbidden symbol matching: ${_forbidden_symbol_pattern}"
    )
endif()

# Keep the denylist from passing vacuously if the model objects disappear from
# the final link.
set(_required_symbol "rocjitsu::gfx1250::Decoder::decode(unsigned int const*)")
string(FIND "${_symbols}" "${_required_symbol}" _match)
if(_match EQUAL -1)
    message(
        FATAL_ERROR
        "model-only binary is missing model symbol: ${_required_symbol}"
    )
endif()
