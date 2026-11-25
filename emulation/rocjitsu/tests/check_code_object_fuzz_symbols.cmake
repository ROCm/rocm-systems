# Copyright (c) 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

if(NOT NM OR NOT FUZZ_BINARY)
    message(FATAL_ERROR "NM and FUZZ_BINARY are required")
endif()

execute_process(
    COMMAND "${NM}" -C "${FUZZ_BINARY}"
    RESULT_VARIABLE _nm_result
    OUTPUT_VARIABLE _symbols
    ERROR_VARIABLE _nm_error
)
if(NOT _nm_result EQUAL 0)
    message(FATAL_ERROR "nm failed: ${_nm_error}")
endif()

# The code-object fuzzer owns parsing and executable loading. Keep translator,
# simulator, runtime-hook, and ISA execution backends out of this binary.
set(_forbidden_symbols
    "::execute_impl("
    "::execution_backend()"
    "_exec("
    "rocjitsu::amdgpu::ComputeUnitCore"
    "rocjitsu::BinaryTranslator::"
    "rocjitsu::Instrumentor::"
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
            "code-object fuzzer contains forbidden symbol: ${_forbidden}"
        )
    endif()
endforeach()

# Require the subject under test so the denylist cannot pass vacuously.
foreach(_required "rocjitsu::Executable::" "rocjitsu::AmdGpuCodeObject::")
    string(FIND "${_symbols}" "${_required}" _match)
    if(_match EQUAL -1)
        message(
            FATAL_ERROR
            "code-object fuzzer is missing loader symbol: ${_required}"
        )
    endif()
endforeach()
