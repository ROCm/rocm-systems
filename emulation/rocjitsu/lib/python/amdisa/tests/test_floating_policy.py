# Copyright (c) 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Tests that the F16 policy table reaches the scalar, per-lane and SIMD generators.

The expectations are written out here rather than read from floating_policy.py,
so a wrong table entry fails these tests before anyone regenerates code.
"""

import pytest

from amdisa.codegen.execute.simd_codegen import simd_probe_line
from amdisa.codegen.execute.sema_lower import lower_sema_block
from amdisa.codegen.execute.vector_alu import gen_vector_unary
from amdisa.sema_ast import ExecModel, SemaBlock, SemaNode, SemaNodeKind, SemaType

_F16_FLUSH = 'amdgpu::input_denormal::flush_input<amdgpu::fp_format::F16>('
_F16_POLICY = 'amdgpu::input_denormal::Policy::make(wf.fp_denorm_mode_f16_f64())'
_TRANS_OUTPUT = 'amdgpu::transcendental_output_modifier_policy<amdgpu::fp_format::F16>'
_PLAIN_OUTPUT = 'amdgpu::output_modifier_policy<amdgpu::fp_format::F16>'

# Every F16 TRANS-unit operation, under each helper spelling the generators
# receive, has its source flushed and the TRANS output policy (gfx1201).
_TRANS_CALLS = ('log', 'log2', 'exp', 'exp2', 'rcp', 'rsq', 'sqrt', 'sin', 'cos')
_TRANS_PER_LANE_OPS = ('log2', 'exp2', 'rcp', 'rsq', 'sqrt', 'sin', 'cos')
_TRANS_SIMD = ('log', 'exp', 'rcp', 'rsq', 'sqrt')


def _operand(name: str, idx: int) -> SemaNode:
    return SemaNode(
        SemaNodeKind.INSTOPERAND,
        ty=SemaType.B32,
        children=(
            SemaNode(SemaNodeKind.ID, id_name=name),
            SemaNode(SemaNodeKind.LIT, lit_value=str(idx)),
        ),
    )


def _f16(inner: SemaNode) -> SemaNode:
    return SemaNode(
        SemaNodeKind.CAST,
        ty=SemaType.F16,
        cast_target=SemaType.F16,
        children=(inner,),
    )


def _lower_f16_unary(value: SemaNode) -> str:
    for wrapper in ('apply_omod', 'apply_clamp'):
        value = SemaNode(
            SemaNodeKind.CALL,
            call_name=wrapper,
            ty=SemaType.F16,
            children=(SemaNode(SemaNodeKind.ID, id_name=wrapper), value),
        )
    body = SemaNode(SemaNodeKind.ASSIGN, children=(_f16(_operand('D', 0)), value))
    return lower_sema_block(SemaBlock('V_F16_UNARY', ExecModel.VECTOR, body))


def _f16_call(callee: str) -> SemaNode:
    return SemaNode(
        SemaNodeKind.CALL,
        call_name=callee,
        ty=SemaType.F16,
        children=(SemaNode(SemaNodeKind.ID, id_name=callee), _f16(_operand('S', 0))),
    )


@pytest.mark.parametrize('callee', _TRANS_CALLS)
def test_scalar_trans_call_flushes_source_and_uses_trans_output(callee: str):
    result = _lower_f16_unary(_f16_call(callee))

    assert result.count(_F16_FLUSH) == 1
    assert _TRANS_OUTPUT in result
    assert _PLAIN_OUTPUT not in result


def test_scalar_ceil_flushes_source_without_trans_output():
    result = _lower_f16_unary(_f16_call('ceil'))

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


@pytest.mark.parametrize('op', _TRANS_PER_LANE_OPS)
def test_per_lane_trans_op_flushes_source_and_uses_trans_output(op: str):
    body = gen_vector_unary(['vdst'], ['src0'], op, 'f16', is_vop3=True)

    assert _F16_FLUSH in body
    assert 'amdgpu::fp_mode::apply_omod_f16(result, effective_omod' in body


def test_per_lane_trunc_keeps_source_without_trans_output():
    body = gen_vector_unary(['vdst'], ['src0'], 'trunc', 'f16', is_vop3=True)

    assert _F16_FLUSH not in body
    assert 'apply_omod_f16(result, effective_omod' not in body


@pytest.mark.parametrize('operation', _TRANS_SIMD)
def test_simd_trans_functor_gets_flushed_source_and_trans_output(operation: str):
    line = simd_probe_line(f'v_{operation}_f16_vop3')

    assert line.endswith(f', true, {_F16_POLICY});')


@pytest.mark.parametrize('operation', ['ceil', 'floor'])
def test_simd_input_flushed_rounding_has_plain_output(operation: str):
    line = simd_probe_line(f'v_{operation}_f16_vop3')

    assert line.endswith(f', false, {_F16_POLICY});')


@pytest.mark.parametrize('operation', ['trunc', 'rndne', 'fract'])
def test_simd_unflushed_rounding_has_neither_property(operation: str):
    line = simd_probe_line(f'v_{operation}_f16_vop3')

    assert 'input_denormal::Policy' not in line
    assert ', true' not in line
