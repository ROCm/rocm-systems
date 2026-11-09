# Copyright (c) 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

if(NOT NM OR NOT MODEL_BINARY OR NOT REQUIRED_MODEL_ARCHES)
    message(
        FATAL_ERROR
        "NM, MODEL_BINARY and REQUIRED_MODEL_ARCHES are required"
    )
endif()

execute_process(
    COMMAND "${NM}" -C --format=bsd "${MODEL_BINARY}"
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
    "_exec("
    "rocjitsu::amdgpu::ComputeUnitCore"
    "ds_calculate_addresses("
    "flat_calculate_addresses("
    "mubuf_calculate_addresses("
    ${EXTRA_FORBIDDEN_SYMBOLS}
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

# The registry takes each model factory's address, even when LTO inlines
# Decoder::decode. Require a definition in this binary to keep the denylist
# from passing on an empty image or one that only imports the model.
execute_process(
    COMMAND "${NM}" -C --defined-only --format=posix "${MODEL_BINARY}"
    RESULT_VARIABLE _nm_result
    OUTPUT_VARIABLE _definitions
    ERROR_VARIABLE _nm_error
)
if(NOT _nm_result EQUAL 0)
    message(FATAL_ERROR "nm failed: ${_nm_error}")
endif()

# Give the first POSIX record the same name boundary as every later record.
set(_definitions "\n${_definitions}")
string(REPLACE "," ";" _required_arches "${REQUIRED_MODEL_ARCHES}")
set(_required_symbols ${EXTRA_REQUIRED_SYMBOLS})
foreach(_arch IN LISTS _required_arches)
    list(
        APPEND _required_symbols
        "rocjitsu::${_arch}::create_model_target_decoder()"
    )
endforeach()
foreach(_required_symbol IN LISTS _required_symbols)
    string(FIND "${_definitions}" "\n${_required_symbol} " _match)
    if(_match EQUAL -1)
        message(
            FATAL_ERROR
            "model-only binary is missing symbol: ${_required_symbol}"
        )
    endif()
endforeach()
