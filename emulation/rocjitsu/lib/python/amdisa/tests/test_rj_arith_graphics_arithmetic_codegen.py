# Copyright (c) 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

from pathlib import Path
from types import SimpleNamespace
from amdisa.codegen.execute import ExecuteContext


import re
import pytest
from amdisa.codegen.execute.rj_arith_float import FLOAT_NAMES
from amdisa.codegen.execute.rj_arith_graphics import (
    INTERP_NAMES,
    gen_interp,
    gen_mullit,
)


def context(arch, name, dst, src, *, is_vop3=False, enc_name=""):
    return ExecuteContext(
        inst=SimpleNamespace(name=name),
        sem=None,
        profile=None,
        arch_name=arch,
        dst_ops=dst,
        src_ops=src,
        enc_name=enc_name,
        is_vop3=is_vop3,
        has_abs=is_vop3,
    )


@pytest.mark.parametrize('arch', ['rdna3', 'rdna4', 'cdna5'])
def test_mullit_loose_policy_and_modifiers(arch):
    assert 'V_MULLIT_F32' in FLOAT_NAMES
    body = gen_mullit(
        context(arch, 'V_MULLIT_F32', ['d'], ['a', 'b', 'c'], is_vop3=True)
    )
    assert 'execute_float_ternary32' in body
    assert 'arithmetic_flags()' in body
    assert 'arithmetic_reports_fp_status' not in body
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
    assert ('rj_arith_v_mullit_f32_w32_gfx12' in body) == (arch in ('rdna4', 'cdna5'))
    assert (
        gen_mullit(
            context('cdna2', 'V_MULLIT_F32', ['d'], ['a', 'b', 'c'], is_vop3=True)
        )
        is None
    )


@pytest.mark.parametrize('arch', ['rdna3', 'rdna4'])
@pytest.mark.parametrize('name', sorted(INTERP_NAMES))
def test_interp_operand_controls_and_policy(arch, name):
    body = gen_interp(
        context(arch, name, ['d'], ['a', 'b', 'c'], enc_name='ENC_VINTERP')
    )
    assert 'rj_arith_' + name.lower() + '_w64' in body
    assert 'arithmetic_flags()' in body
    assert 'reports_state' not in body
    assert 'wait_exp' not in body
    assert 'd, a, b, c' in body
    assert ('RJ_ARITH_ALU_HIGH_A' in body) == ('F16' in name)
    assert ('RJ_ARITH_ALU_HIGH_C' in body) == ('F16' in name and '_P10_' in name)
    assert ('RJ_ARITH_ALU_HIGH_D' in body) == ('F16' in name and '_P2_' in name)
    assert (
        gen_interp(
            context('cdna5', name, ['d'], ['a', 'b', 'c'], enc_name='ENC_VINTERP')
        )
        is None
    )
    assert (
        gen_interp(context(arch, name, ['d'], ['a', 'b'], enc_name='ENC_VINTERP'))
        is None
    )


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


@pytest.mark.parametrize('arch', ['rdna3', 'rdna4'])
def test_cube_reuses_ternary_adapter_and_guest_policies(arch):
    from amdisa.codegen.execute.rj_arith_graphics import CUBE_NAMES, gen_cube

    for name in CUBE_NAMES:
        body = gen_cube(context(arch, name, ['dst'], ['x', 'y', 'z'], is_vop3=True))
        arch_suffix = '_gfx12' if arch == 'rdna4' else ''
        api = 'rj_arith_' + name.lower()
        assert api + '_w64' + arch_suffix in body
        assert 'execute_ternary32' in body
        assert 'wf.exec() & wf.vgpr_write_mask()' in body
        assert 'arithmetic_flags()' in body
        assert 'fp_round_mode_f32()' in body
        assert 'RJ_ARITH_FP_IEEE_MODE' in body
        assert ('RJ_ARITH_FP_DX10_CLAMP_OFF' in body) == (arch == 'rdna3')
        assert 'arithmetic_reports_fp_status' not in body
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
        assert (
            gen_cube(context('cdna4', name, ['dst'], ['x', 'y', 'z'], is_vop3=True))
            is None
        )
        assert gen_cube(context(arch, name, ['dst'], ['x', 'y'], is_vop3=True)) is None
        assert (
            gen_cube(context(arch, name, ['dst'], ['x', 'y', 'z'], is_vop3=False))
            is None
        )
