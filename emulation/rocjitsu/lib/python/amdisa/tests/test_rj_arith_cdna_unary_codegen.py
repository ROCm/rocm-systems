# Copyright (c) 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Check CDNA unary policy selection and operand modifiers before code generation."""

from types import SimpleNamespace

import pytest

from amdisa.codegen.execute import ExecuteContext
from amdisa.codegen.execute.rj_arith_cdna_unary import (
    CDNA_INTEGRAL_UNARY,
    gen_cdna_integral_unary,
)


def context(arch, name, vop3=False):
    return ExecuteContext(
        inst=SimpleNamespace(name=name),
        sem=None,
        dst_ops=['dst'],
        src_ops=['src'],
        profile=None,
        enc_name='vop3' if vop3 else 'vop1',
        is_vop3=vop3,
        has_abs=vop3,
        arch_name=arch,
    )


@pytest.mark.parametrize('arch', ['cdna3', 'cdna4', 'cdna5'])
@pytest.mark.parametrize('name', sorted(CDNA_INTEGRAL_UNARY))
@pytest.mark.parametrize('vop3', [False, True])
def test_integral_unary_policy_and_operands(arch, name, vop3):
    body = gen_cdna_integral_unary(context(arch, name, vop3))
    suffix = 'gfx12' if arch == 'cdna5' else 'gfx950' if arch == 'cdna4' else 'gfx942'
    for wave in (32, 64):
        assert f'rj_arith_{name.lower()}_w{wave}_{suffix}' in body
    # Unqualified strict requests (mode 3, and uncaptured DPP/SDWA mode 2 on CDNA4) reject first.
    guard = body.index('RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL')
    reject = body.index('return;', guard)
    execute = body.index('execute_cdna_integral_unary')
    assert guard < reject < execute
    assert (
        'RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL_NO_FP_STATUS' in body[guard:reject]
    ) == (arch != 'cdna4' or not vop3)
    if arch == 'cdna4' and not vop3:
        assert 'amdgpu::SRC_DPP' in body[guard:reject]
        assert 'amdgpu::SRC_SDWA' in body[guard:reject]
    assert 'wf.cu().arithmetic_flags()' in body
    assert 'dst, src' in body
    assert ('RJ_ARITH_ALU_OMOD_2' in body) == vop3
    assert ('RJ_ARITH_ALU_ABS_A' in body) == vop3
    assert ('RJ_ARITH_ALU_NEG_A' in body) == vop3
    assert ('RJ_ARITH_ALU_CLAMP' in body) == vop3
    assert ('RJ_ARITH_ALU_HIGH_A' in body) == (vop3 and name.endswith('F16'))
    assert ('RJ_ARITH_ALU_HIGH_D' in body) == (vop3 and name.endswith('F16'))


@pytest.mark.parametrize('arch', ['rdna3', 'rdna4', 'cdna1', 'cdna2'])
def test_other_architectures_unchanged(arch):
    assert gen_cdna_integral_unary(context(arch, 'V_FLOOR_F32')) is None


@pytest.mark.parametrize('name', ['V_FLOOR_F64', 'V_RCP_F32'])
def test_other_operations_unchanged(name):
    assert gen_cdna_integral_unary(context('cdna4', name)) is None


def test_unexpected_operand_shape_not_claimed():
    ctx = context('cdna4', 'V_FLOOR_F32')
    ctx.src_ops.append('extra')
    assert gen_cdna_integral_unary(ctx) is None
