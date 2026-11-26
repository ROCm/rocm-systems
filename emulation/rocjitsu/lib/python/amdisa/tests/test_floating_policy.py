# Copyright (c) 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

import pytest
from amdisa.codegen._generator import CodeGenerator
from amdisa.codegen.execute.floating_policy import (
    F16_TRANS_OPERATIONS,
    FLUSH_NEAREST_F32_OPS,
)
from amdisa.semantics import InstructionSemantics

_F16_FLUSH = 'amdgpu::input_denormal::flush_input<amdgpu::fp_format::F16>('


_TRANS_OUTPUT = 'amdgpu::transcendental_output_modifier_policy<amdgpu::fp_format::F16>'


_PLAIN_OUTPUT = 'amdgpu::output_modifier_policy<amdgpu::fp_format::F16>'


_EXPECTED_F16_TRANS_INSTRUCTIONS = {
    'V_LOG_F16',
    'V_EXP_F16',
    'V_RCP_F16',
    'V_RSQ_F16',
    'V_SQRT_F16',
    'V_SIN_F16',
    'V_COS_F16',
}


_EXPECTED_FLUSH_NEAREST_F32_OPS = {
    'V_LOG_LEGACY_F32',
    'V_EXP_LEGACY_F32',
    'V_LOG_F32',
    'V_EXP_F32',
    'V_RCP_F32',
    'V_RSQ_F32',
    'V_SQRT_F32',
    'V_COS_F32',
}


_NO_LEGACY_SIMD_PROBE_F32 = {
    'V_COS_F32',
    'V_EXP_LEGACY_F32',
    'V_LOG_LEGACY_F32',
}


def test_f16_trans_membership_is_explicit():
    assert {operation.instruction for operation in F16_TRANS_OPERATIONS} == (
        _EXPECTED_F16_TRANS_INSTRUCTIONS
    )


def test_flush_nearest_f32_membership_is_explicit():
    assert FLUSH_NEAREST_F32_OPS == _EXPECTED_FLUSH_NEAREST_F32_OPS


@pytest.mark.parametrize('instruction', sorted(FLUSH_NEAREST_F32_OPS))
def test_sdwa_flush_nearest_f32_uses_matching_output_policy(instruction: str):
    sem = InstructionSemantics(instruction, 'vector_unary')

    assert CodeGenerator._sdwa_output_policy(sem) == (
        ', amdgpu::sdwa::OutputPolicy::FLUSH_NEAREST'
    )


# These forms have no specialized legacy SIMD probe.


"""Shared floating policy metadata and live scalar/SIMD fallback wiring."""


"""F16 and F32 arithmetic policy wiring in semantic lowering."""
