from pathlib import Path


import re


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


@pytest.mark.parametrize('arch', ['rdna3', 'rdna4'])
def test_generated_graphics_arithmetic_routes(arch):
    root = (
        Path(__file__).resolve().parents[3]
        / 'rocjitsu/src/rocjitsu/isa/arch/amdgpu/generated'
    )
    count = 0
    for file, pattern, bridge in (
        ('vinterp_exec.cpp', r'VInterp\w+', 'arith::execute_interp<'),
        ('vop3_exec.cpp', 'VMullitF32Vop3', 'arith::execute_float_ternary32'),
    ):
        source = (root / arch / file).read_text()
        for match in re.finditer(
            r'void (' + pattern + r')::execute(?:_modifier)?_impl\([^)]*\)\s*\{', source
        ):
            end, depth = match.end(), 1
            while depth:
                depth += (source[end] == '{') - (source[end] == '}')
                end += 1
            body = source[match.end() : end - 1]
            assert bridge in body
            assert 'no-op' not in body
            assert 'execute_v_mullit' not in body
            count += 1
    assert count == 8
