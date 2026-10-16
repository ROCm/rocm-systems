# Copyright (c) 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT


import pytest

from amdisa.codegen.execute.rj_arith_float import FLOAT_NAMES, gen_mullit
from amdisa.codegen.execute.rj_arith_interp import INTERP_NAMES, gen_interp


@pytest.mark.parametrize('arch', ['rdna3', 'rdna4'])
def test_mullit_loose_policy_and_modifiers(arch):
    assert 'V_MULLIT_F32' in FLOAT_NAMES
    body = gen_mullit(arch, 'V_MULLIT_F32', ['d'], ['a', 'b', 'c'], True)
    assert 'execute_float_ternary32' in body
    assert 'arithmetic_flags()' in body
    assert 'arithmetic_reports_fp_status()' in body
    assert 'd, a, b, c' in body
    for modifier in (
        'ABS_A',
        'ABS_B',
        'ABS_C',
        'NEG_A',
        'NEG_B',
        'NEG_C',
        'OMOD_2',
        'CLAMP',
    ):
        assert 'RJ_ARITH_ALU_' + modifier in body
    assert ('rj_arith_v_mullit_f32_w32_gfx12' in body) == (arch == 'rdna4')
    assert gen_mullit('cdna5', 'V_MULLIT_F32', ['d'], ['a', 'b', 'c'], True) is None


@pytest.mark.parametrize('arch', ['rdna3', 'rdna4'])
@pytest.mark.parametrize('name', sorted(INTERP_NAMES))
def test_interp_operand_controls_and_policy(arch, name):
    body = gen_interp(arch, name, ['d'], ['a', 'b', 'c'], 'ENC_VINTERP')
    assert 'rj_arith_' + name.lower() + '_w64' in body
    assert 'arithmetic_flags()' in body
    assert 'reports_state' not in body
    assert 'RJ_ARITH_INTERP_WAIT_EXP_SHIFT' in body
    assert 'd, a, b, c' in body
    assert ('RJ_ARITH_ALU_HIGH_A' in body) == ('F16' in name)
    assert ('RJ_ARITH_ALU_HIGH_C' in body) == ('F16' in name and '_P10_' in name)
    assert ('RJ_ARITH_ALU_HIGH_D' in body) == ('F16' in name and '_P2_' in name)
    assert gen_interp('cdna5', name, ['d'], ['a', 'b', 'c'], 'ENC_VINTERP') is None
    assert gen_interp(arch, name, ['d'], ['a', 'b'], 'ENC_VINTERP') is None
