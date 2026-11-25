# Copyright (c) 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

from pathlib import Path
import re
from types import SimpleNamespace

import pytest

from amdisa.codegen.execute import ExecuteContext
from amdisa.codegen.execute.rj_arith_division import DIVISION_NAMES, gen_division


def context(arch, name, dst, src, is_vop3, result_writer=None):
    return ExecuteContext(
        inst=SimpleNamespace(name=name),
        sem=None,
        profile=None,
        arch_name=arch,
        dst_ops=dst,
        src_ops=src,
        enc_name="",
        is_vop3=is_vop3,
        has_abs=is_vop3,
        result_writer=result_writer,
    )


@pytest.mark.parametrize('arch', ['rdna3', 'rdna4'])
@pytest.mark.parametrize('name', sorted(DIVISION_NAMES))
def test_division_operands_and_guest_policy(arch, name):
    scale = '_SCALE_' in name
    dst = ['vdst', 'sdst'] if scale else ['vdst']
    body = gen_division(context(arch, name, dst, ['a', 'b', 'c'], True))
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
    if name.endswith('F16'):
        assert 'execute_float_ternary16' in body
        for letter in 'ABCD':
            assert 'RJ_ARITH_ALU_HIGH_' + letter in body
    if scale:
        assert 'commit_result' in gen_division(
            context(arch, name, dst, ['a', 'b', 'c'], True, 'commit_result')
        )
    else:
        assert (
            gen_division(
                context(arch, name, dst, ['a', 'b', 'c'], True, 'commit_result')
            )
            is None
        )
    for excluded_arch in ('cdna2', 'rdna3_5'):
        assert (
            gen_division(context(excluded_arch, name, dst, ['a', 'b', 'c'], True))
            is None
        )
    assert gen_division(context(arch, name, dst, ['a', 'b'], True)) is None
    assert gen_division(context(arch, name, dst, ['a', 'b', 'c'], False)) is None


@pytest.mark.parametrize('arch', ['rdna3', 'rdna4', 'cdna3', 'cdna4', 'cdna5'])
def test_generated_division_handlers_use_rj_arith(arch):
    root = (
        Path(__file__).resolve().parents[3]
        / 'rocjitsu/src/rocjitsu/isa/arch/amdgpu/generated'
    )
    source = (
        root / arch / ('vop3_exec_alu.cpp' if arch == 'cdna5' else 'vop3_exec.cpp')
    ).read_text()
    checked = 0
    for match in re.finditer(
        r'void (VDiv(?:Scale|Fmas|Fixup)F(?:16|32|64)Vop3\w*)::execute(?:_modifier)?_impl\([^)]*\)\s*\{',
        source,
    ):
        if arch.startswith('cdna') and 'Fmas' not in match[1]:
            continue
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
    # The shared CDNA5 source also contains the gfx1251 FP64 DPP modifier handler.
    assert checked == (3 if arch == 'cdna5' else 2 if arch.startswith('cdna') else 8)


@pytest.mark.parametrize('arch', ['cdna3', 'cdna4', 'cdna5'])
@pytest.mark.parametrize('width', [32, 64])
def test_cdna_fmas_guest_policy(arch, width):
    body = gen_division(
        context(arch, f'V_DIV_FMAS_F{width}', ['vdst'], ['a', 'b', 'c'], True)
    )
    suffix = 'gfx1250' if arch == 'cdna5' else 'gfx942'
    assert f'rj_arith_v_div_fmas_f{width}_w32_{suffix}' in body
    assert f'rj_arith_v_div_fmas_f{width}_w64_gfx942' in body
    assert f'/*rdna3=*/{str(arch != "cdna5").lower()}' in body
    assert 'DivisionOperation::Fmas' in body
    assert 'vdst, a, b, c' in body


@pytest.mark.parametrize('arch', ['cdna3', 'cdna4', 'cdna5'])
@pytest.mark.parametrize('width', [32, 64])
def test_cdna_scale_preserves_inactive_condition(arch, width):
    name = f'V_DIV_SCALE_F{width}'
    body = gen_division(
        context(arch, name, ['vdst', 'sdst'], ['a', 'b', 'c'], True, 'commit_result')
    )
    assert f'rj_arith_v_div_scale_f{width}_w32_gfx942' in body
    assert 'commit_result, /*preserve_inactive_condition=*/true' in body
    assert 'RJ_ARITH_ALU_ABS_A' not in body
    root = (
        Path(__file__).resolve().parents[3]
        / 'rocjitsu/src/rocjitsu/isa/arch/amdgpu/generated'
        / arch
    )
    source = '\n'.join(p.read_text() for p in root.glob('vop3_exec*.cpp'))
    start = source.index(f'void VDivScaleF{width}Vop3SdstEnc::execute_impl')
    end = source.index('\n}', start)
    assert f'rj_arith_v_div_scale_f{width}_w32_gfx942' in source[start:end]
    assert '/*preserve_inactive_condition=*/true' in source[start:end]


@pytest.mark.parametrize('arch', ['cdna3', 'cdna4', 'cdna5'])
@pytest.mark.parametrize('width', [16, 32, 64])
def test_cdna_fixup_policy(arch, width):
    body = gen_division(
        context(arch, f'V_DIV_FIXUP_F{width}', ['vdst'], ['a', 'b', 'c'], True)
    )
    suffix = 'gfx1250' if arch == 'cdna5' else 'gfx942'
    assert f'rj_arith_v_div_fixup_f{width}_w32_{suffix}' in body
    assert f'rj_arith_v_div_fixup_f{width}_w64_gfx942' in body
    assert f'/*rdna3=*/{str(arch != "cdna5").lower()}' in body
    if width == 16:
        # Ordinary FIXUP uses true16 writes; only LEGACY clears the other half.
        assert '/*low_dst_zeroes_high=*/false' in body
    else:
        assert 'DivisionOperation::Fixup' in body


@pytest.mark.parametrize('arch', ['cdna3', 'cdna4'])
def test_legacy_half_fixup_uses_cdna_policy(arch):
    body = gen_division(
        context(arch, 'V_DIV_FIXUP_LEGACY_F16', ['d'], ['a', 'b', 'c'], True)
    )
    for wave in (32, 64):
        assert f'rj_arith_v_div_fixup_legacy_f16_w{wave}' in body
    assert '/*rdna3=*/true' in body
    assert '/*low_dst_zeroes_high=*/true' in body
    for letter in 'ABCD':
        assert f'RJ_ARITH_ALU_HIGH_{letter}' in body
    assert (
        gen_division(
            context('cdna5', 'V_DIV_FIXUP_LEGACY_F16', ['d'], ['a', 'b', 'c'], True)
        )
        is None
    )
