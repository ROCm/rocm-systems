from pathlib import Path


import re


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


@pytest.mark.parametrize('arch', ['rdna3', 'rdna4'])
def test_generated_division_handlers_use_rj_arith(arch):
    root = (
        Path(__file__).resolve().parents[3]
        / 'rocjitsu/src/rocjitsu/isa/arch/amdgpu/generated'
    )
    source = (root / arch / 'vop3_exec.cpp').read_text()
    checked = 0
    for match in re.finditer(
        r'void (VDiv(?:Scale|Fmas|Fixup)F(?:16|32|64)Vop3\w*)::execute(?:_modifier)?_impl\([^)]*\)\s*\{',
        source,
    ):
        end, depth = match.end(), 1
        while depth:
            depth += (source[end] == '{') - (source[end] == '}')
            end += 1
        body = source[match.end() : end - 1]
        assert (
            'arith::execute_float_ternary16'
            if 'F16' in match[1]
            else 'arith::execute_division<'
        ) in body, match[1]
        assert 'try_execute_' not in body
        assert 'amdgpu::execute_v_' not in body
        checked += 1
    assert checked == 8
