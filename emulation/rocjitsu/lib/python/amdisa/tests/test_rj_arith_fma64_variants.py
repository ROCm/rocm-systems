# Copyright (c) 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Check implicit accumulator and literal operand placement in CDNA FMA dispatch."""

from types import SimpleNamespace
import pytest
from amdisa.codegen.execute.rj_arith_fma64_variants import gen_fma64_variants


def context(arch, name, operands, vop3=False):
    return SimpleNamespace(
        arch_name=arch,
        inst=SimpleNamespace(name=name),
        dst_ops=['vdst'],
        src_ops=operands,
        is_vop3=vop3,
    )


@pytest.mark.parametrize('arch', ['cdna3', 'cdna4', 'cdna5'])
@pytest.mark.parametrize('vop3', [False, True])
def test_accumulate(arch, vop3):
    body = gen_fma64_variants(context(arch, 'V_FMAC_F64', ['src0', 'src1'], vop3))
    assert f'execute_fmac64<{str(arch != "cdna5").lower()}>' in body
    assert 'vdst, src0, src1' in body
    assert ('RJ_ARITH_ALU_OMOD_2' in body) == vop3
    assert 'RJ_ARITH_ALU_ABS_C' not in body
    assert 'RJ_ARITH_ALU_NEG_C' not in body


@pytest.mark.parametrize(
    'name,inputs,multiply',
    [
        ('V_FMAMK_F64', ['src0', 'literal64', 'vsrc1'], True),
        ('V_FMAAK_F64', ['src0', 'vsrc1', 'literal64'], False),
    ],
)
def test_literal(name, inputs, multiply):
    body = gen_fma64_variants(context('cdna5', name, inputs))
    assert f'execute_literal_fma64<{str(multiply).lower()}>' in body
    assert 'vdst, src0, vsrc1, literal64' in body
    assert gen_fma64_variants(context('cdna4', name, inputs)) is None
    assert gen_fma64_variants(context('cdna5', name, inputs, True)) is None


def test_other_architectures_unchanged():
    assert gen_fma64_variants(context('rdna4', 'V_FMAC_F64', ['a', 'b'])) is None
    assert (
        gen_fma64_variants(context('cdna4', 'V_FMA_F64', ['a', 'b', 'c'], True)) is None
    )
