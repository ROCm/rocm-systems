# Copyright (c) 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""F16 policy wiring in scalar semantics lowering and SIMD probe generation.

TRANS cases follow the shared policy table and available SIMD probes. SIN/COS
currently have no specialized F16 VOP3 unary probes; their scalar wiring is covered.
"""

import pytest

from amdisa.codegen._generator import CodeGenerator
from amdisa.codegen.execute.floating_policy import (
    F16_TRANS_OPERATIONS,
    F16_TRANSCENDENTAL_CALLS,
    FLUSH_NEAREST_F32_OPS,
)
from amdisa.codegen.execute.simd_codegen import SIMD_VOP3_UNARY_FP16, simd_probe_line
from amdisa.codegen.execute.sema_lower import lower_sema_block
from amdisa.sema_ast import ExecModel, SemaBlock, SemaNode, SemaNodeKind, SemaType
from amdisa.semantics import InstructionSemantics

_F16_FLUSH = 'amdgpu::input_denormal::flush_input<amdgpu::fp_format::F16>('
_F16_POLICY = 'amdgpu::input_denormal::Policy::make(wf.fp_denorm_mode_f16_f64())'
_TRANS_OUTPUT = 'amdgpu::transcendental_output_modifier_policy<amdgpu::fp_format::F16>'
_PLAIN_OUTPUT = 'amdgpu::output_modifier_policy<amdgpu::fp_format::F16>'

# Probe availability and TRANS classification jointly determine SIMD coverage.
_TRANS_SIMD_TEMPLATES = tuple(
    template
    for operation in F16_TRANS_OPERATIONS
    if (template := f'{operation.instruction.lower()}_vop3') in SIMD_VOP3_UNARY_FP16
)
_EXPECTED_F16_TRANS_INSTRUCTIONS = {
    'V_LOG_F16',
    'V_EXP_F16',
    'V_RCP_F16',
    'V_RSQ_F16',
    'V_SQRT_F16',
    'V_SIN_F16',
    'V_COS_F16',
}
_EXPECTED_TRANS_SIMD_TEMPLATES = {
    'v_log_f16_vop3',
    'v_exp_f16_vop3',
    'v_rcp_f16_vop3',
    'v_rsq_f16_vop3',
    'v_sqrt_f16_vop3',
}
_EXPECTED_FLUSH_NEAREST_F32_OPS = {
    'V_LOG_F32',
    'V_EXP_F32',
    'V_RCP_F32',
    'V_RSQ_F32',
    'V_SQRT_F32',
    'V_COS_F32',
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
    assert set(_TRANS_SIMD_TEMPLATES) == _EXPECTED_TRANS_SIMD_TEMPLATES
    assert {
        f'{operation.instruction.lower()}_vop3' for operation in F16_TRANS_OPERATIONS
    } - set(_TRANS_SIMD_TEMPLATES) == {
        'v_sin_f16_vop3',
        'v_cos_f16_vop3',
    }


@pytest.mark.parametrize('callee', sorted(F16_TRANSCENDENTAL_CALLS))
def test_scalar_trans_call_flushes_source(callee: str):
    result = _lower_f16_unary(_f16_call(callee))

    assert result.count(_F16_FLUSH) == 1


@pytest.mark.parametrize('callee', sorted(F16_TRANSCENDENTAL_CALLS))
def test_scalar_trans_call_uses_trans_output(callee: str):
    result = _lower_f16_unary(_f16_call(callee))

    assert _TRANS_OUTPUT in result
    assert _PLAIN_OUTPUT not in result


@pytest.mark.parametrize('operation', ['ceil', 'floor'])
def test_scalar_input_flushed_rounding_has_plain_output(operation: str):
    value = (
        _f16_call(operation)
        if operation == 'ceil'
        else SemaNode(
            SemaNodeKind.FLOOR,
            ty=SemaType.F16,
            children=(_f16(_operand('S', 0)),),
        )
    )
    result = _lower_f16_unary(value)

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


@pytest.mark.parametrize('template', _TRANS_SIMD_TEMPLATES)
def test_simd_trans_functor_gets_flushed_source(template: str):
    line = simd_probe_line(template)

    assert _F16_POLICY in line


@pytest.mark.parametrize('template', _TRANS_SIMD_TEMPLATES)
def test_simd_trans_functor_uses_trans_output(template: str):
    line = simd_probe_line(template)

    assert ', true,' in line


@pytest.mark.parametrize('operation', ['ceil', 'floor'])
def test_simd_input_flushed_rounding_has_plain_output(operation: str):
    line = simd_probe_line(f'v_{operation}_f16_vop3')

    assert line.endswith(f', false, {_F16_POLICY});')


@pytest.mark.parametrize('operation', ['trunc', 'rndne', 'fract'])
def test_simd_unflushed_rounding_has_neither_property(operation: str):
    line = simd_probe_line(f'v_{operation}_f16_vop3')

    assert 'input_denormal::Policy' not in line
    assert ', true' not in line


def test_flush_nearest_f32_membership_is_explicit():
    assert FLUSH_NEAREST_F32_OPS == _EXPECTED_FLUSH_NEAREST_F32_OPS


@pytest.mark.parametrize('instruction', sorted(FLUSH_NEAREST_F32_OPS))
def test_scalar_flush_nearest_f32_forces_omod_despite_output_keep(instruction: str):
    callee = instruction.removeprefix('V_').removesuffix('_F32').lower()
    result = _lower_fp_unary(_fp_call(callee, SemaType.F32), SemaType.F32, instruction)

    assert 'effective_omod(wf.cu().arch(), 0, wf.ieee_mode(), inst_.omod)' in result


@pytest.mark.parametrize(
    'instruction',
    sorted(FLUSH_NEAREST_F32_OPS - {'V_COS_F32'}),
)
def test_simd_flush_nearest_f32_forces_output_flush(instruction: str):
    line = simd_probe_line(f'{instruction.lower()}_vop3')

    assert 'true /* force_output_flush */' in line


def test_cos_f32_has_no_specialized_simd_probe():
    assert simd_probe_line('v_cos_f32_vop3') is None


@pytest.mark.parametrize('instruction', sorted(FLUSH_NEAREST_F32_OPS))
def test_sdwa_flush_nearest_f32_uses_matching_output_policy(instruction: str):
    sem = InstructionSemantics(instruction, 'vector_unary')

    assert CodeGenerator._sdwa_output_policy(sem) == (
        ', amdgpu::sdwa::OutputPolicy::FLUSH_NEAREST'
    )
