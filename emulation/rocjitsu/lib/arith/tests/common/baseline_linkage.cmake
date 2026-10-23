# Copyright (c) 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

# Scalar repair definitions must never compete with target-specific COMDATs.
# Check the archive members directly: arithmetic tests on a recent CPU cannot
# expose a baseline call accidentally resolving to a higher-ISA definition.
execute_process(
    COMMAND "${NM}" -A -C --defined-only "${ARCHIVE}"
    OUTPUT_VARIABLE symbols
    COMMAND_ERROR_IS_FATAL ANY
)
string(REPLACE "\n" ";" lines "${symbols}")
foreach(line IN LISTS lines)
    if(line MATCHES "x86_64[^:]*[.]cpp[.]o:" AND
       line MATCHES "rocjitsu::arith::(gfx11_dot2|gfx12_dot_bits|half_fma_value<|fma_instruction_value|fma_value_f32|fma_finite_round_odd|repair_)")
        message(FATAL_ERROR "Scalar repair emitted in a target-specific object: ${line}")
    endif()
endforeach()
