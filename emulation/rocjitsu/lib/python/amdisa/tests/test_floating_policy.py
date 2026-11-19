# Copyright (c) 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

import pytest
from amdisa.codegen._generator import CodeGenerator
from amdisa.codegen.execute.floating_policy import (
    F16_TRANS_OPERATIONS,
    FLUSH_NEAREST_F32_OPS,
    INPUT_FLUSHED_ROUNDING,
)
from amdisa.codegen.execute.simd_codegen import simd_probe_line
from amdisa.codegen.execute.sema_lower import lower_sema_block
from amdisa.sema_ast import ExecModel, SemaBlock, SemaNode, SemaNodeKind, SemaType
from amdisa.semantics import InstructionSemantics

_F16_FLUSH = 'amdgpu::input_denormal::flush_input<amdgpu::fp_format::F16>('


_F16_POLICY = 'amdgpu::input_denormal::Policy::make(wf.fp_denorm_mode_f16_f64())'


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


_MIGRATED_ROOT_F32 = {'V_RCP_F32', 'V_RSQ_F32', 'V_SQRT_F32'}


_NO_LEGACY_SIMD_PROBE_F32 = {
    'V_COS_F32',
    'V_EXP_LEGACY_F32',
    'V_LOG_LEGACY_F32',
}


def _operand(name: str, idx: int) -> SemaNode:
    return SemaNode(
        SemaNodeKind.INSTOPERAND,
        ty=SemaType.B32,
        children=(
            SemaNode(SemaNodeKind.ID, id_name=name),
            SemaNode(SemaNodeKind.LIT, lit_value=str(idx)),
        ),
    )


def _fp(inner: SemaNode, ty: SemaType) -> SemaNode:
    return SemaNode(
        SemaNodeKind.CAST,
        ty=ty,
        cast_target=ty,
        children=(inner,),
    )


def _f16(inner: SemaNode) -> SemaNode:
    return _fp(inner, SemaType.F16)


def _lower_fp_unary(
    value: SemaNode, ty: SemaType, instruction_name: str = 'V_F16_UNARY'
) -> str:
    for wrapper in ('apply_omod', 'apply_clamp'):
        value = SemaNode(
            SemaNodeKind.CALL,
            call_name=wrapper,
            ty=ty,
            children=(SemaNode(SemaNodeKind.ID, id_name=wrapper), value),
        )
    body = SemaNode(SemaNodeKind.ASSIGN, children=(_fp(_operand('D', 0), ty), value))
    return lower_sema_block(SemaBlock(instruction_name, ExecModel.VECTOR, body))


def _lower_f16_unary(value: SemaNode) -> str:
    return _lower_fp_unary(value, SemaType.F16)


def _fp_call(callee: str, ty: SemaType) -> SemaNode:
    return SemaNode(
        SemaNodeKind.CALL,
        call_name=callee,
        ty=ty,
        children=(
            SemaNode(SemaNodeKind.ID, id_name=callee),
            _fp(_operand('S', 0), ty),
        ),
    )


def _f16_call(callee: str) -> SemaNode:
    return _fp_call(callee, SemaType.F16)


def test_f16_trans_membership_is_explicit():
    assert {operation.instruction for operation in F16_TRANS_OPERATIONS} == (
        _EXPECTED_F16_TRANS_INSTRUCTIONS
    )


@pytest.mark.parametrize('operation', sorted(INPUT_FLUSHED_ROUNDING))
def test_scalar_input_flushed_rounding_has_plain_output(operation: str):
    values = {
        'ceil': _f16_call('ceil'),
        'floor': SemaNode(
            SemaNodeKind.FLOOR,
            ty=SemaType.F16,
            children=(_f16(_operand('S', 0)),),
        ),
    }
    result = _lower_f16_unary(values[operation])

    assert result.count(_F16_FLUSH) == 1
    assert _PLAIN_OUTPUT in result
    assert _TRANS_OUTPUT not in result


def test_scalar_trunc_keeps_source_without_trans_output():
    trunc = SemaNode(
        SemaNodeKind.TRUNC, ty=SemaType.F16, children=(_f16(_operand('S', 0)),)
    )
    result = _lower_f16_unary(trunc)

    assert _F16_FLUSH not in result
    assert _TRANS_OUTPUT not in result


def test_flush_nearest_f32_membership_is_explicit():
    assert FLUSH_NEAREST_F32_OPS == _EXPECTED_FLUSH_NEAREST_F32_OPS


@pytest.mark.parametrize('instruction', sorted(FLUSH_NEAREST_F32_OPS))
def test_scalar_flush_nearest_f32_forces_omod_despite_output_keep(instruction: str):
    callee = instruction.removeprefix('V_').removesuffix('_F32').lower()
    result = _lower_fp_unary(_fp_call(callee, SemaType.F32), SemaType.F32, instruction)

    assert 'effective_omod(wf.cu().arch(), 0, wf.ieee_mode(), inst_.omod)' in result


@pytest.mark.parametrize(
    'instruction',
    sorted(
        FLUSH_NEAREST_F32_OPS
        - _NO_LEGACY_SIMD_PROBE_F32
        - {'V_EXP_F32', 'V_LOG_F32'}
        - _MIGRATED_ROOT_F32
    ),
)
def test_simd_flush_nearest_f32_forces_output_flush(instruction: str):
    line = simd_probe_line(f'{instruction.lower()}_vop3')

    assert 'true /* force_output_flush */' in line


@pytest.mark.parametrize('instruction', sorted(FLUSH_NEAREST_F32_OPS))
def test_sdwa_flush_nearest_f32_uses_matching_output_policy(instruction: str):
    sem = InstructionSemantics(instruction, 'vector_unary')

    assert CodeGenerator._sdwa_output_policy(sem) == (
        ', amdgpu::sdwa::OutputPolicy::FLUSH_NEAREST'
    )


# These forms have no specialized legacy SIMD probe.


@pytest.mark.parametrize('instruction', sorted(_NO_LEGACY_SIMD_PROBE_F32))
def test_math_without_legacy_probe_has_no_specialized_simd_probe(instruction: str):
    assert simd_probe_line(f'{instruction.lower()}_vop3') is None


@pytest.mark.parametrize(
    'instruction',
    sorted(
        _MIGRATED_ROOT_F32
        | {
            'V_RCP_IFLAG_F32',
            'V_EXP_F32',
            'V_LOG_F32',
            'V_SIN_F32',
            'V_COS_F32',
            'V_EXP_LEGACY_F32',
            'V_LOG_LEGACY_F32',
        }
    ),
)
def test_migrated_math_has_no_legacy_simd_probe(instruction: str):
    for form in ('vop1', 'vop3'):
        assert simd_probe_line(f'{instruction.lower()}_{form}') is None


"""Shared floating policy metadata and live scalar/SIMD fallback wiring."""
