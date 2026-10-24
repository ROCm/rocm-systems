# Copyright (c) 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT


import pytest

from amdisa.codegen.execute.rj_arith_division import DIVISION_NAMES, gen_division


@pytest.mark.parametrize('arch', ['rdna3', 'rdna4'])
@pytest.mark.parametrize('name', sorted(DIVISION_NAMES))
def test_division_operands_and_guest_policy(arch, name):
    scale = '_SCALE_' in name
    dst = ['vdst', 'sdst'] if scale else ['vdst']
    body = gen_division(arch, name, dst, ['a', 'b', 'c'], True)
    arch_suffix = '_gfx12' if arch == 'rdna4' else ''
    api = 'rj_arith_' + name.lower()
    assert api + '_w32' + arch_suffix in body
    assert api + '_w64' + arch_suffix in body
    assert 'vdst, a, b, c' in body
    assert ('write_wave_mask_scalar(sdst' in body) == scale
    assert ('RJ_ARITH_ALU_ABS_A' in body) == (not scale)
    for flag in ['NEG_A', 'NEG_B', 'NEG_C', 'OMOD_2', 'CLAMP']:
        assert 'RJ_ARITH_ALU_' + flag in body
    assert 'arithmetic_flags()' in body
    assert ('arithmetic_reports_fp_status' in body) == (not scale)
    if name.endswith('F16'):
        assert 'execute_float_ternary16' in body
        assert 'arithmetic_reports_fp_status()' in body
        for letter in 'ABCD':
            assert 'RJ_ARITH_ALU_HIGH_' + letter in body
    if scale:
        assert 'commit_result' in gen_division(
            arch, name, dst, ['a', 'b', 'c'], True, 'commit_result'
        )
    else:
        assert (
            gen_division(arch, name, dst, ['a', 'b', 'c'], True, 'commit_result')
            is None
        )
    assert gen_division('cdna5', name, dst, ['a', 'b', 'c'], True) is None
    assert gen_division(arch, name, dst, ['a', 'b'], True) is None
    assert gen_division(arch, name, dst, ['a', 'b', 'c'], False) is None
