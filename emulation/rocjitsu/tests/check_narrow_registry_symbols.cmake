# Copyright (c) 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

if(NOT NM OR NOT NARROW_LIBRARY)
    message(FATAL_ERROR "NM and NARROW_LIBRARY are required")
endif()

execute_process(
    COMMAND "${NM}" -C "${NARROW_LIBRARY}"
    RESULT_VARIABLE _nm_result
    OUTPUT_VARIABLE _symbols
    ERROR_VARIABLE _nm_error
)
if(NOT _nm_result EQUAL 0)
    message(FATAL_ERROR "nm failed: ${_nm_error}")
endif()

set(_required_symbols
    "rj_test_narrow_target_count"
    "rocjitsu::gfx1250::Decoder::decode(unsigned int const*)"
)
foreach(_required IN LISTS _required_symbols)
    string(FIND "${_symbols}" "${_required}" _match)
    if(_match EQUAL -1)
        message(FATAL_ERROR "narrow library is missing symbol: ${_required}")
    endif()
endforeach()

set(_forbidden_symbols
    "rocjitsu::cdna1::"
    "rocjitsu::cdna2::"
    "rocjitsu::cdna3::"
    "rocjitsu::cdna4::"
    "rocjitsu::rdna1::"
    "rocjitsu::rdna2::"
    "rocjitsu::rdna3::"
    "rocjitsu::rdna3_5::"
    "rocjitsu::rdna4::"
    "rocjitsu::gfx1250::execution_backend()"
    "::execute_impl("
    "rocjitsu::amdgpu::ComputeUnitCore"
)
foreach(_forbidden IN LISTS _forbidden_symbols)
    string(FIND "${_symbols}" "${_forbidden}" _match)
    if(NOT _match EQUAL -1)
        message(
            FATAL_ERROR
            "narrow library contains forbidden symbol: ${_forbidden}"
        )
    endif()
endforeach()
