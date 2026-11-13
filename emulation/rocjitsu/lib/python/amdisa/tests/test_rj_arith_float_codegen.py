# Copyright (c) 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT


import re
from pathlib import Path
from types import SimpleNamespace

import pytest

from amdisa.codegen.execute import ExecuteContext
from amdisa.codegen.execute.rj_arith_float import (
    FMA_F32,
    CDNA_MATH_F16,
    CDNA_MATH_F32,
    CDNA_MATH_F64,
    gen_float_unary16,
    gen_float_unary64,
    gen_float_fma32,
    gen_float_binary32,
    gen_float_unary32,
)
from rj_arith_headers import public_headers


def test_other_architectures_keep_existing_path():
    for arch in ('rdna1', 'rdna2', 'rdna3_5', 'cdna1', 'cdna2'):
        assert gen_float_unary32(context(arch, 'V_CEIL_F32', ['dst'], ['src'])) is None


@pytest.mark.parametrize('arch', ['rdna3', 'rdna4'])
def test_generated_float_methods_use_policy_and_no_legacy_fast_path(arch):
    root = (
        Path(__file__).resolve().parents[3]
        / 'rocjitsu/src/rocjitsu/isa/arch/amdgpu/generated'
        / arch
    )
    checked = 0
    for encoding in ('vop1', 'vop3'):
        source = (root / (encoding + '_exec.cpp')).read_text()
        for match in re.finditer(
            r'void (\w+)::execute(?:_modifier)?_impl\([^)]*\)\s*\{', source
        ):
            end, depth = match.end(), 1
            while depth:
                depth += (source[end] == '{') - (source[end] == '}')
                end += 1
            body = source[match.end() : end - 1]
            if 'arith::execute_float_unary32' not in body:
                continue
            assert 'RJ_SIMD' not in body, match[1]
            assert 'try_execute_' not in body, match[1]
            assert 'amdgpu::execute_v_' not in body, match[1]
            assert 'arithmetic_flags()' in body, match[1]
            checked += 1
    assert checked == 60


@pytest.mark.parametrize('arch,count', [('rdna3', 30), ('rdna4', 34)])
def test_generated_binary_float_methods_have_no_legacy_bypass(arch, count):
    root = (
        Path(__file__).resolve().parents[3]
        / 'rocjitsu/src/rocjitsu/isa/arch/amdgpu/generated'
        / arch
    )
    checked = 0
    for encoding in ('vop2', 'vop3'):
        source = (root / (encoding + '_exec.cpp')).read_text()
        for match in re.finditer(
            r'void (\w+)::execute(?:_modifier)?_impl\([^)]*\)\s*\{', source
        ):
            end, depth = match.end(), 1
            while depth:
                depth += (source[end] == '{') - (source[end] == '}')
                end += 1
            body = source[match.end() : end - 1]
            if 'arith::execute_float_binary32' not in body:
                continue
            assert 'RJ_SIMD' not in body, match[1]
            assert 'try_execute_' not in body, match[1]
            assert 'amdgpu::execute_v_' not in body, match[1]
            checked += 1
    assert checked == count


def context(
    arch,
    name,
    dst,
    src,
    *,
    is_vop3=False,
    enc_name='',
    opsel_exprs=('', ''),
    op_sel_hi_2_expr='',
):
    return ExecuteContext(
        inst=SimpleNamespace(name=name),
        sem=None,
        dst_ops=dst,
        src_ops=src,
        profile=None,
        enc_name=enc_name,
        is_vop3=is_vop3,
        has_abs=is_vop3,
        arch_name=arch,
        opsel_exprs=opsel_exprs,
        op_sel_hi_2_expr=op_sel_hi_2_expr,
    )


@pytest.mark.parametrize('arch', ['rdna3', 'rdna4'])
@pytest.mark.parametrize('name', sorted(FMA_F32))
def test_fma_operands_modifiers_and_optional_state(arch, name):
    accumulate = name in ('V_FMAC_F32', 'V_FMAC_DX9_ZERO_F32')
    literal = name in ('V_FMAAK_F32', 'V_FMAMK_F32')
    sources = ['a', 'b'] if accumulate else ['a', 'b', 'c']
    body = gen_float_fma32(context(arch, name, ['dst'], sources, is_vop3=not literal))
    if arch == 'rdna4' and name == 'V_FMAC_DX9_ZERO_F32':
        assert body is None
        return
    assert 'dst, a, b, dst' in body if accumulate else 'dst, a, b, c' in body
    assert ('RJ_ARITH_ALU_NEG_C' in body) == (not accumulate and not literal)
    assert ('RJ_ARITH_ALU_ABS_C' in body) == (not accumulate and not literal)
    suffix = '_gfx12' if arch == 'rdna4' else ''
    operation = 'fma_dx9_zero' if 'DX9_ZERO' in name else 'fma'
    for wave in (32, 64):
        assert f'rj_arith_v_{operation}_f32_w{wave}{suffix},' in body
    assert 'arithmetic_flags()' in body
    assert (
        gen_float_fma32(context('cdna2', name, ['dst'], sources, is_vop3=not literal))
        is None
    )
    assert (
        gen_float_fma32(
            context(arch, name, ['dst'], sources + ['extra'], is_vop3=not literal)
        )
        is None
    )


def test_literal_fma_sources_keep_their_arithmetic_order():
    assert 'dst, a, literal, b' in gen_float_fma32(
        context('rdna4', 'V_FMAMK_F32', ['dst'], ['a', 'literal', 'b'], is_vop3=False)
    )
    assert 'dst, a, b, literal' in gen_float_fma32(
        context('rdna3', 'V_FMAAK_F32', ['dst'], ['a', 'b', 'literal'], is_vop3=False)
    )


@pytest.mark.parametrize('arch,count', [('rdna3', 12), ('rdna4', 10)])
def test_generated_fma_forms_use_rj_arith_without_legacy_arithmetic(arch, count):
    root = (
        Path(__file__).resolve().parents[3]
        / 'rocjitsu/src/rocjitsu/isa/arch/amdgpu/generated'
        / arch
    )
    checked = 0
    for encoding in ('vop2', 'vop3'):
        source = (root / (encoding + '_exec.cpp')).read_text()
        for match in re.finditer(
            r'void (\w+)::execute(?:_modifier)?_impl\([^)]*\)\s*\{', source
        ):
            end, depth = match.end(), 1
            while depth:
                depth += (source[end] == '{') - (source[end] == '}')
                end += 1
            body = source[match.end() : end - 1]
            if 'arith::execute_float_fma32' not in body:
                continue
            assert 'RJ_SIMD' not in body, match[1]
            assert 'try_execute_' not in body, match[1]
            assert 'amdgpu::execute_v_' not in body, match[1]
            assert 'ScopedEnvironment' not in body, match[1]
            checked += 1
    assert checked == count


@pytest.mark.parametrize('arch', ['rdna3', 'rdna4'])
def test_packed_fma_routes_modes_and_inline_sources(arch):
    from amdisa.codegen.execute.rj_arith_float import gen_packed_fma16

    body = gen_packed_fma16(
        context(
            arch,
            'V_PK_FMA_F16',
            ['d'],
            ['a', 'b', 'c'],
            enc_name='ENC_VOP3P',
            opsel_exprs=('low', 'high'),
            op_sel_hi_2_expr='high_c',
        )
    )
    suffix = '_gfx12' if arch == 'rdna4' else ''
    for wave in (32, 64):
        assert f'rj_arith_v_pk_fma_f16_w{wave}{suffix},' in body
    assert 'execute_packed_fma16<false>' in body
    assert 'RJ_ARITH_PK_NEG_LO_C' in body and 'RJ_ARITH_PK_NEG_HI_C' in body
    assert 'pk16_src_needs_narrowing(inst_.src2, c.size_bits())' in body
    assert 'replicate_sources = 0' in body
    body = gen_packed_fma16(
        context(
            arch,
            'V_PK_FMAC_F16',
            ['d'],
            ['a', 'b'],
            enc_name='ENC_VOP2',
            opsel_exprs=('low', 'high'),
            op_sel_hi_2_expr='high_c',
        )
    )
    for wave in (32, 64):
        assert f'rj_arith_v_pk_fmac_f16_w{wave}{suffix},' in body
    assert 'execute_packed_fma16<true>' in body
    assert 'd, a, b, d' in body
    assert 'dot2_src_needs_half_replication(inst_.src0)' in body
    assert 'inst_.src1' not in body


@pytest.mark.parametrize('arch', ['rdna3', 'rdna4'])
@pytest.mark.parametrize('name', ['V_PK_ADD_F16', 'V_PK_MUL_F16'])
def test_packed_binary_routes_modes_inline_sources_and_policy(arch, name):
    from amdisa.codegen.execute.rj_arith_float import gen_packed_binary16

    body = gen_packed_binary16(
        context(
            arch,
            name,
            ['d'],
            ['a', 'b'],
            enc_name='ENC_VOP3P',
            opsel_exprs=('low', 'high'),
        )
    )
    arch_suffix = '_gfx12' if arch == 'rdna4' else ''
    api = 'rj_arith_' + name.lower()
    assert api + '_w32' + arch_suffix + ', ' + api + '_w64' + arch_suffix in body
    for source in 'AB':
        for control in (
            'LO_' + source + '_HIGH',
            'HI_' + source + '_LOW',
            'NEG_LO_' + source,
            'NEG_HI_' + source,
        ):
            assert 'RJ_ARITH_PK_' + control in body
    assert 'RJ_ARITH_PK_CLAMP' in body
    assert 'pk16_src_needs_narrowing(inst_.src0, a.size_bits())' in body
    assert 'pk16_src_needs_narrowing(inst_.src1, b.size_bits())' in body
    assert 'arithmetic_flags()' in body
    assert (
        gen_packed_binary16(
            context(
                'cdna4',
                name,
                ['d'],
                ['a', 'b'],
                enc_name='ENC_VOP3P',
                opsel_exprs=('low', 'high'),
            )
        )
        is None
    )
    assert (
        gen_packed_binary16(
            context(
                arch,
                name,
                ['d'],
                ['a', 'b'],
                enc_name='ENC_VOP3',
                opsel_exprs=('low', 'high'),
            )
        )
        is None
    )


@pytest.mark.parametrize('arch', ['rdna3', 'rdna4'])
def test_generated_packed_binary_methods_have_no_legacy_arithmetic(arch):
    source = (
        Path(__file__).resolve().parents[3]
        / 'rocjitsu/src/rocjitsu/isa/arch/amdgpu/generated'
        / arch
        / 'vop3p_exec.cpp'
    ).read_text()
    checked = 0
    for match in re.finditer(r'void (\w+)::execute_impl\([^)]*\)\s*\{', source):
        end, depth = match.end(), 1
        while depth:
            depth += (source[end] == '{') - (source[end] == '}')
            end += 1
        body = source[match.end() : end - 1]
        if 'arith::execute_packed_binary16' not in body:
            continue
        assert 'try_execute_' not in body, match[1]
        assert 'packed_binary_f16' not in body, match[1]
        assert 'ScopedEnvironment' not in body, match[1]
        checked += 1
    assert checked == (4 if arch == 'rdna3' else 6)


@pytest.mark.parametrize('arch', ['rdna3', 'rdna4'])
@pytest.mark.parametrize(
    'name',
    [
        'V_PK_MIN_F16',
        'V_PK_MAX_F16',
        'V_PK_MIN_NUM_F16',
        'V_PK_MAX_NUM_F16',
        'V_PK_MINIMUM_F16',
        'V_PK_MAXIMUM_F16',
    ],
)
def test_packed_selection_architecture_and_state_binding(arch, name):
    from amdisa.codegen.execute.rj_arith_float import gen_packed_binary16

    body = gen_packed_binary16(
        context(
            arch,
            name,
            ['d'],
            ['a', 'b'],
            enc_name='ENC_VOP3P',
            opsel_exprs=('low', 'high'),
        )
    )
    legacy = name in ('V_PK_MIN_F16', 'V_PK_MAX_F16')
    if legacy != (arch == 'rdna3'):
        assert body is None
        return
    api = 'rj_arith_' + name.lower()
    assert api + '_w32' + ', ' + api + '_w64' in body
    assert 'narrow_sources' in body
    assert 'arithmetic_flags()' in body


@pytest.mark.parametrize(
    'name',
    [
        'V_DOT4_F32_FP8_FP8',
        'V_DOT4_F32_FP8_BF8',
        'V_DOT4_F32_BF8_FP8',
        'V_DOT4_F32_BF8_BF8',
    ],
)
def test_dot4_formats_and_c_modifiers(name):
    from amdisa.codegen.execute.rj_arith_float import gen_float_dot32

    body = gen_float_dot32(
        context(
            'rdna4',
            name,
            ['d'],
            ['a', 'b', 'c'],
            enc_name='ENC_VOP3P',
            opsel_exprs=('low', 'high'),
        )
    )
    assert 'rj_arith_' + name.lower() in body
    assert 'inst_.neg & 4' in body and 'RJ_ARITH_DOT_NEG_C' in body
    assert 'inst_.neg_hi & 4' in body and 'RJ_ARITH_DOT_ABS_C' in body
    assert 'arithmetic_flags()' in body
    assert (
        gen_float_dot32(
            context(
                'rdna3',
                name,
                ['d'],
                ['a', 'b', 'c'],
                enc_name='ENC_VOP3P',
                opsel_exprs=('low', 'high'),
            )
        )
        is None
    )


@pytest.mark.parametrize('arch, count', [('rdna3', 6), ('rdna4', 12)])
def test_generated_float_dot_methods_replace_all_legacy_paths(arch, count):
    root = (
        Path(__file__).resolve().parents[3]
        / 'rocjitsu/src/rocjitsu/isa/arch/amdgpu/generated'
        / arch
    )
    checked = 0
    for encoding in ('vop2', 'vop3p'):
        source = (root / (encoding + '_exec.cpp')).read_text()
        for match in re.finditer(
            r'void (VDot(?:2(?:acc)?F32|4F32)\w+)::execute(?:_modifier)?_impl\([^)]*\)\s*\{',
            source,
        ):
            end, depth = match.end(), 1
            while depth:
                depth += (source[end] == '{') - (source[end] == '}')
                end += 1
            body = source[match.end() : end - 1]
            assert 'arith::execute_' in body, match[1]
            assert 'ROCJITSU_TRY_SIMD' not in body, match[1]
            assert 'amdgpu::execute_v_dot' not in body, match[1]
            assert 'amdgpu::gfx' not in body, match[1]
            checked += 1
    assert checked == count


@pytest.mark.parametrize('arch', ['rdna3', 'rdna4'])
@pytest.mark.parametrize('name', ['V_DOT2_F16_F16', 'V_DOT2_BF16_BF16'])
def test_half_dot_packed_sources_and_partial_accumulator(arch, name):
    from amdisa.codegen.execute.rj_arith_float import gen_half_dot

    body = gen_half_dot(context(arch, name, ['d'], ['a', 'b', 'c'], is_vop3=True))
    arch_suffix = '_gfx12' if arch == 'rdna4' else ''
    api = 'rj_arith_' + name.lower()
    assert api + '_w32' + arch_suffix + ', ' + api + '_w64' + arch_suffix in body
    for source in 'ABC':
        assert 'RJ_ARITH_ALU_ABS_' + source in body
        assert 'RJ_ARITH_ALU_NEG_' + source in body
    assert 'RJ_ARITH_ALU_HIGH_C' in body and 'RJ_ARITH_ALU_HIGH_D' in body
    assert 'RJ_ARITH_ALU_HIGH_A' not in body and 'RJ_ARITH_ALU_HIGH_B' not in body
    assert 'pk16_src_needs_narrowing(inst_.src0, a.size_bits())' in body
    assert 'dot2_src_needs_half_replication(inst_.src1)' in body
    assert 'dot2_src_needs_half_replication(inst_.src2) ? 4u : 0u' in body
    assert 'arithmetic_flags()' in body
    assert (
        gen_half_dot(context('cdna4', name, ['d'], ['a', 'b', 'c'], is_vop3=True))
        is None
    )


@pytest.mark.parametrize('arch', ['rdna3', 'rdna4'])
def test_generated_half_dot_regular_and_dpp_bodies_use_rj_arith(arch):
    source = (
        Path(__file__).resolve().parents[3]
        / 'rocjitsu/src/rocjitsu/isa/arch/amdgpu/generated'
        / arch
        / 'vop3_exec.cpp'
    ).read_text()
    checked = 0
    for match in re.finditer(
        r'void (VDot2(?:F16F16|Bf16Bf16)Vop3)::execute(?:_modifier)?_impl\([^)]*\)\s*\{',
        source,
    ):
        end, depth = match.end(), 1
        while depth:
            depth += (source[end] == '{') - (source[end] == '}')
            end += 1
        body = source[match.end() : end - 1]
        assert 'arith::execute_dot2_f16' in body
        assert 'fp_mode::dot2_' not in body
        assert 'ROCJITSU_TRY_SIMD' not in body
        checked += 1
    assert checked == 4


@pytest.mark.parametrize('arch', ['rdna3', 'rdna4'])
@pytest.mark.parametrize(
    'name', ['V_ADD_F16', 'V_SUB_F16', 'V_SUBREV_F16', 'V_MUL_F16', 'V_LDEXP_F16']
)
def test_half_binary_arithmetic_modes_and_selectors(arch, name):
    from amdisa.codegen.execute.rj_arith_float import gen_float_binary16

    body = gen_float_binary16(context(arch, name, ['d'], ['a', 'b'], is_vop3=True))
    arch_suffix = '_gfx12' if arch == 'rdna4' else ''
    api = 'rj_arith_' + name.lower()
    assert api + '_w32' + arch_suffix + ', ' + api + '_w64' + arch_suffix in body
    for control in (
        'ABS_A',
        'ABS_B',
        'NEG_A',
        'NEG_B',
        'HIGH_A',
        'HIGH_B',
        'HIGH_D',
        'OMOD_2',
        'CLAMP',
    ):
        if name == 'V_LDEXP_F16' and control in ('ABS_B', 'NEG_B'):
            assert 'RJ_ARITH_ALU_' + control not in body
        else:
            assert 'RJ_ARITH_ALU_' + control in body
    assert 'arithmetic_flags()' in body
    vop2 = gen_float_binary16(context(arch, name, ['d'], ['a', 'b'], is_vop3=False))
    assert 'inst_.op' not in vop2 and 'RJ_ARITH_ALU_HIGH' not in vop2
    assert (
        gen_float_binary16(context('rdna3_5', name, ['d'], ['a', 'b'], is_vop3=True))
        is None
    )


@pytest.mark.parametrize('arch', ['rdna3', 'rdna4'])
def test_half_binary_all_encoding_bodies_route_rj_arith(arch):
    root = (
        Path(__file__).resolve().parents[3]
        / 'rocjitsu/src/rocjitsu/isa/arch/amdgpu/generated'
        / arch
    )
    count = 0
    for encoding in ('vop2', 'vop3'):
        source = (root / (encoding + '_exec.cpp')).read_text()
        for match in re.finditer(
            r'void (V(?:Add|Sub|Subrev|Mul|Ldexp)F16Vop[23])::execute(?:_modifier)?_impl\([^)]*\)\s*\{',
            source,
        ):
            end, depth = match.end(), 1
            while depth:
                depth += (source[end] == '{') - (source[end] == '}')
                end += 1
            body = source[match.end() : end - 1]
            assert 'arith::execute_float_binary16' in body
            assert 'fp_mode::arithmetic_f16' not in body
            assert 'ROCJITSU_TRY_SIMD' not in body
            count += 1
    assert count == 20


@pytest.mark.parametrize(
    'arch,names,count',
    [
        ('rdna3', ('V_MIN_F16', 'V_MAX_F16'), 8),
        (
            'rdna4',
            ('V_MIN_NUM_F16', 'V_MAX_NUM_F16', 'V_MINIMUM_F16', 'V_MAXIMUM_F16'),
            12,
        ),
    ],
)
def test_half_selection_uses_architecture_specific_names_and_all_bodies(
    arch, names, count
):
    from amdisa.codegen.execute.rj_arith_float import gen_float_binary16

    root = (
        Path(__file__).resolve().parents[3]
        / 'rocjitsu/src/rocjitsu/isa/arch/amdgpu/generated'
        / arch
    )
    for name in names:
        body = gen_float_binary16(context(arch, name, ['d'], ['a', 'b'], is_vop3=True))
        api = 'rj_arith_' + name.lower()
        assert api + '_w32' + ', ' + api + '_w64' in body
        assert 'arithmetic_flags()' in body
        assert (
            gen_float_binary16(
                context(
                    'rdna4' if arch == 'rdna3' else 'rdna3',
                    name,
                    ['d'],
                    ['a', 'b'],
                    is_vop3=True,
                )
            )
            is None
        )
    checked = 0
    for encoding in ('vop2', 'vop3'):
        source = (root / (encoding + '_exec.cpp')).read_text()
        for match in re.finditer(
            r'void (V(?:Min|Max)(?:Num|imum)?F16Vop[23])::execute(?:_modifier)?_impl\([^)]*\)\s*\{',
            source,
        ):
            end, depth = match.end(), 1
            while depth:
                depth += (source[end] == '{') - (source[end] == '}')
                end += 1
            body = source[match.end() : end - 1]
            assert 'arith::execute_float_binary16' in body
            assert 'std::fmin' not in body and 'std::fmax' not in body
            assert 'ROCJITSU_TRY_SIMD' not in body
            checked += 1
    assert checked == count


@pytest.mark.parametrize('arch', ['rdna3', 'rdna4'])
@pytest.mark.parametrize(
    'name', ['V_FMA_F16', 'V_FMAC_F16', 'V_FMAAK_F16', 'V_FMAMK_F16']
)
def test_half_fma_explicit_accumulator_literal_order_and_modifiers(arch, name):
    from amdisa.codegen.execute.rj_arith_float import gen_float_fma16

    accumulate = name == 'V_FMAC_F16'
    inputs = (
        ['a', 'b']
        if accumulate
        else ['a', 'literal', 'c'] if name == 'V_FMAMK_F16' else ['a', 'b', 'c']
    )
    body = gen_float_fma16(
        context(arch, name, ['d'], inputs, is_vop3=name in ('V_FMA_F16', 'V_FMAC_F16'))
    )
    arch_suffix = '_gfx12' if arch == 'rdna4' else ''
    api = 'rj_arith_v_fma_f16'
    assert api + '_w32' + arch_suffix + ', ' + api + '_w64' + arch_suffix in body
    assert 'd, ' + ', '.join(inputs + ['d'] if accumulate else inputs) in body
    assert 'arithmetic_flags()' in body
    if accumulate:
        assert '((opsel & 8) ? RJ_ARITH_ALU_HIGH_C' in body
        assert 'RJ_ARITH_ALU_NEG_C' not in body and 'RJ_ARITH_ALU_ABS_C' not in body
    elif name == 'V_FMA_F16':
        assert '((opsel & 4) ? RJ_ARITH_ALU_HIGH_C' in body
        assert 'RJ_ARITH_ALU_NEG_C' in body and 'RJ_ARITH_ALU_ABS_C' in body
    assert (
        gen_float_fma16(context('rdna3_5', name, ['d'], inputs, is_vop3=False)) is None
    )


@pytest.mark.parametrize('arch', ['rdna3', 'rdna4'])
def test_all_half_fma_bodies_route_rj_arith(arch):
    root = (
        Path(__file__).resolve().parents[3]
        / 'rocjitsu/src/rocjitsu/isa/arch/amdgpu/generated'
        / arch
    )
    count = 0
    for encoding in ('vop2', 'vop3'):
        source = (root / (encoding + '_exec.cpp')).read_text()
        for match in re.finditer(
            r'void (VFma(?:c|ak|mk)?F16Vop[23])::execute(?:_modifier)?_impl\([^)]*\)\s*\{',
            source,
        ):
            end, depth = match.end(), 1
            while depth:
                depth += (source[end] == '{') - (source[end] == '}')
                end += 1
            body = source[match.end() : end - 1]
            assert 'arith::execute_float_ternary16' in body
            assert 'fp_mode::fma_f16' not in body and 'ROCJITSU_TRY_SIMD' not in body
            count += 1
    assert count == 8


@pytest.mark.parametrize('arch', ['rdna3', 'rdna4'])
def test_trig_all_encodings_route_rj_arith_instead_of_shared_legacy_helpers(arch):
    root = (
        Path(__file__).resolve().parents[3]
        / 'rocjitsu/src/rocjitsu/isa/arch/amdgpu/generated'
        / arch
    )
    count = 0
    for encoding in ('vop1', 'vop3'):
        source = (root / (encoding + '_exec.cpp')).read_text()
        for match in re.finditer(
            r'void V(?:Sin|Cos)F(16|32)Vop[13]::execute(?:_modifier)?_impl\([^)]*\)\s*\{',
            source,
        ):
            end, depth = match.end(), 1
            while depth:
                depth += (source[end] == '{') - (source[end] == '}')
                end += 1
            body = source[match.end() : end - 1]
            assert 'arith::execute_float_unary' + match[1] in body
            assert 'arithmetic_flags()' in body
            assert 'amdgpu::execute_v_' not in body
            assert 'try_execute_' not in body and 'ROCJITSU_TRY_SIMD' not in body
            count += 1
    assert count == 16


@pytest.mark.parametrize('arch,count', [('rdna3', 20), ('rdna4', 36)])
def test_ternary_selection_routes_all_encodings_and_modifiers(arch, count):
    from amdisa.codegen.execute.rj_arith_float import (
        TERNARY_SELECTION_RDNA3,
        TERNARY_SELECTION_RDNA4,
        gen_float_ternary_selection,
    )

    names = TERNARY_SELECTION_RDNA3 if arch == 'rdna3' else TERNARY_SELECTION_RDNA4
    root = (
        Path(__file__).resolve().parents[3]
        / 'rocjitsu/src/rocjitsu/isa/arch/amdgpu/generated'
        / arch
    )
    classes = {
        'V' + ''.join(part.title() for part in name.split('_')[1:]) for name in names
    }
    for name in names:
        body = gen_float_ternary_selection(
            context(arch, name, ['d'], ['a', 'b', 'c'], is_vop3=True)
        )
        assert 'rj_arith_' + name.lower() + '_w64' in body
        assert 'arithmetic_flags()' in body
        for letter in 'ABC':
            for flag in ('ABS_', 'NEG_'):
                assert 'RJ_ARITH_ALU_' + flag + letter in body
        if name.endswith('_F16'):
            for letter in 'ABCD':
                assert 'RJ_ARITH_ALU_HIGH_' + letter in body
        else:
            assert 'RJ_ARITH_ALU_HIGH' not in body
        assert (
            gen_float_ternary_selection(
                context('cdna4', name, ['d'], ['a', 'b', 'c'], is_vop3=True)
            )
            is None
        )
    source = (root / 'vop3_exec.cpp').read_text()
    seen = 0
    for match in re.finditer(
        r'void (V\w+F(?:16|32))Vop3::execute(?:_modifier)?_impl\([^)]*\)\s*\{', source
    ):
        if match[1] not in classes:
            continue
        end, depth = match.end(), 1
        while depth:
            depth += (source[end] == '{') - (source[end] == '}')
            end += 1
        body = source[match.end() : end - 1]
        assert 'arith::execute_float_ternary' in body
        assert 'ROCJITSU_TRY_SIMD' not in body
        seen += 1
    assert seen == count


@pytest.mark.parametrize('arch', ['rdna3', 'rdna4'])
@pytest.mark.parametrize('name', ['V_LDEXP_F64', 'V_TRIG_PREOP_F64'])
def test_float64_word_uses_pair_and_integer_adapter(arch, name):
    from amdisa.codegen.execute.rj_arith_float import gen_float64_word

    body = gen_float64_word(context(arch, name, ['dst'], ['a', 'b'], is_vop3=True))
    assert 'execute_float64_word' in body
    assert 'arithmetic_flags()' in body
    assert 'RJ_ARITH_ALU_ABS_B' not in body
    assert 'RJ_ARITH_ALU_NEG_B' not in body
    assert (
        gen_float64_word(context('cdna2', name, ['dst'], ['a', 'b'], is_vop3=True))
        is None
    )
    assert gen_float64_word(context(arch, name, ['dst'], ['a'], is_vop3=True)) is None
    assert ('/*ldexp=*/true' in body) == (name == 'V_LDEXP_F64')
    path = (
        Path(__file__).resolve().parents[3]
        / 'rocjitsu/src/rocjitsu/isa/arch/amdgpu/generated'
        / arch
        / 'vop3_exec.cpp'
    )
    source = path.read_text()
    cls = 'VLdexpF64' if name == 'V_LDEXP_F64' else 'VTrigPreopF64'
    start = source.index('void ' + cls + 'Vop3::execute_impl')
    end = source.index('\n}', start)
    body = source[start:end]
    assert 'execute_float64_word' in body
    assert 'try_execute_' not in body
    assert 'amdgpu::execute_v_' not in body


@pytest.mark.parametrize('arch', ['rdna3', 'rdna4'])
def test_fp64_selection_pair_adapter(arch):
    from amdisa.codegen.execute.rj_arith_float import SELECTION_F64, gen_float_binary64

    for name in SELECTION_F64[arch]:
        body = gen_float_binary64(
            context(arch, name, ['dst'], ['a', 'b'], is_vop3=True)
        )
        assert 'execute_float_binary64' in body
        assert 'rj_arith_' + name.lower() + '_w64' in body
        assert 'arithmetic_flags()' in body
        for modifier in ('ABS_A', 'ABS_B', 'NEG_A', 'NEG_B', 'OMOD_2', 'CLAMP'):
            assert 'RJ_ARITH_ALU_' + modifier in body
        assert (
            gen_float_binary64(
                context('cdna2', name, ['dst'], ['a', 'b'], is_vop3=True)
            )
            is None
        )
        assert (
            gen_float_binary64(context(arch, name, ['dst'], ['a'], is_vop3=True))
            is None
        )
        assert 'modifiers = 0' in gen_float_binary64(
            context(arch, name, ['dst'], ['a', 'b'], is_vop3=False)
        )


@pytest.mark.parametrize('arch', ['rdna3', 'rdna4', 'cdna3', 'cdna4', 'cdna5'])
def test_fp64_binary_arithmetic_adapter(arch):
    from amdisa.codegen.execute.rj_arith_float import gen_float_binary64

    for name in ('V_ADD_F64', 'V_MUL_F64'):
        for vop3 in (False, True):
            body = gen_float_binary64(
                context(arch, name, ['dst'], ['a', 'b'], is_vop3=vop3)
            )
            arch_suffix = '_gfx12' if arch in ('rdna4', 'cdna5') else ''
            if name in ('V_ADD_F64', 'V_MUL_F64') and arch.startswith('cdna'):
                arch_suffix = '_gfx1250' if arch == 'cdna5' else '_gfx942'
            api = 'rj_arith_' + name.lower()
            for wave in (32, 64):
                assert api + f'_w{wave}' + arch_suffix + ',' in body
            assert 'execute_float_binary64' in body
            assert 'arithmetic_flags()' in body


@pytest.mark.parametrize('arch', ['rdna3', 'rdna4', 'cdna3', 'cdna4', 'cdna5'])
def test_byte_conversion_policy_and_selected_read_byte(arch):
    from amdisa.codegen.execute.rj_arith_float import gen_byte_nibble_conversion

    for index in range(5):
        byte = index % 4
        name = f'V_CVT_F32_UBYTE{index}' if index < 4 else 'V_CVT_OFF_F32_I4'
        for vop3 in (False, True):
            body = gen_byte_nibble_conversion(
                context(arch, name, ['dst'], ['src'], is_vop3=vop3)
            )
            arch_suffix = '_gfx12' if arch in ('rdna4', 'cdna5') else ''
            api = 'rj_arith_' + name.lower()
            assert api + '_w64' + arch_suffix in body
            assert 'execute_unary32' in body
            assert f'dst, src, {1 << byte}' in body
            assert 'arithmetic_flags()' in body
            assert 'RJ_ARITH_FP_IEEE_MODE' in body
            assert 'RJ_ARITH_FP_FLUSH_OUTPUT_DENORMALS' in body
            assert ('inst_.omod' in body) == vop3
            assert 'inst_.abs' not in body and 'inst_.neg' not in body
            assert ('execute_byte_conversion_sdwa' in body) == (
                arch in ('cdna3', 'cdna4') and not vop3
            )


@pytest.mark.parametrize('packed', [False, True])
def test_fp8_widen_selectors_and_pair_result(packed):
    from amdisa.codegen.execute.rj_arith_float import gen_fp8_widen

    for kind in ('FP8', 'BF8'):
        name = 'V_CVT_' + ('PK_' if packed else '') + 'F32_' + kind
        for vop3 in (False, True):
            body = gen_fp8_widen(context('rdna4', name, ['dst'], ['src'], is_vop3=vop3))
            assert 'rj_arith_' + name.lower() + '_w64' in body
            assert ('execute_widen32_to_pair' if packed else 'execute_unary32') in body
            assert 'arithmetic_flags()' in body
            assert ('amdgpu::vop3_opsel(inst_)' in body) == vop3
            assert 'source_bytes' in body
            assert 'arithmetic_reports_fp_status' not in body
            assert 'inst_.clamp' not in body
        assert (
            gen_fp8_widen(context('rdna3', name, ['dst'], ['src'], is_vop3=True))
            is None
        )
        for arch, suffix in [
            ('cdna3', 'gfx942'),
            ('cdna4', 'gfx950'),
            ('cdna5', 'gfx950'),
        ]:
            if arch == 'cdna5' and name == 'V_CVT_F32_FP8':
                suffix = 'gfx1250'
            for vop3 in (False, True):
                body = gen_fp8_widen(
                    context(arch, name, ['dst'], ['src'], is_vop3=vop3)
                )
                assert f'rj_arith_{name.lower()}_w64_{suffix}' in body
                assert ('inst_.clamp' in body) == (vop3 and suffix == 'gfx1250')
                assert ('execute_fp8_widen_sdwa' in body) == (
                    not packed and not vop3 and arch != 'cdna5'
                )
        assert gen_fp8_widen(context('rdna4', name, ['dst'], [], is_vop3=True)) is None


@pytest.mark.parametrize('stochastic', [False, True])
def test_fp8_narrow_modes_and_partial_destination(stochastic):
    from amdisa.codegen.execute.rj_arith_float import gen_fp8_narrow

    for kind in ('FP8', 'BF8'):
        name = 'V_CVT_' + ('SR_' if stochastic else 'PK_') + kind + '_F32'
        body = gen_fp8_narrow(context('rdna4', name, ['dst'], ['a', 'b'], is_vop3=True))
        assert 'rj_arith_' + name.lower() + '_w64' in body
        assert 'execute_binary32' in body
        assert 'arithmetic_flags()' in body
        assert 'RJ_ARITH_FP16_OVFL' in body
        assert 'RJ_ARITH_ALU_ABS_A' in body and 'RJ_ARITH_ALU_NEG_A' in body
        assert ('RJ_ARITH_ALU_ABS_B' in body) != stochastic
        assert ('RJ_ARITH_ALU_NEG_B' in body) != stochastic
        assert 'destination_bytes' in body
        assert 'arithmetic_reports_fp_status' not in body
        assert 'inst_.clamp' not in body and 'inst_.omod' not in body
        assert (
            gen_fp8_narrow(context('rdna3', name, ['dst'], ['a', 'b'], is_vop3=True))
            is None
        )
        for arch, suffix in [
            ('cdna3', 'gfx942'),
            ('cdna4', 'gfx950'),
            ('cdna5', 'gfx950'),
        ]:
            if arch == 'cdna5':
                suffix = 'gfx1250'
            body = gen_fp8_narrow(
                context(arch, name, ['dst'], ['a', 'b'], is_vop3=True)
            )
            assert f'rj_arith_{name.lower()}_w64_{suffix}' in body
            assert 'amdgpu::vop3_opsel(inst_)' in body
            assert ('inst_.clamp' in body) == (suffix == 'gfx1250' and kind == 'FP8')
        assert (
            gen_fp8_narrow(context('rdna4', name, ['dst'], ['a', 'b'], is_vop3=False))
            is None
        )


@pytest.mark.parametrize('arch', ['rdna3', 'rdna4', 'cdna3', 'cdna4', 'cdna5'])
def test_conversion64_operand_layouts_and_policy(arch):
    from amdisa.codegen.execute.rj_arith_float import (
        CONVERSION64_NAMES,
        gen_conversion64,
    )

    for name in CONVERSION64_NAMES:
        output, source = name.split('_')[-2:]
        from_integer = source in ('I32', 'U32')
        for vop3 in (False, True):
            body = gen_conversion64(context(arch, name, ['dst'], ['src'], is_vop3=vop3))
            arch_suffix = '_gfx12' if arch in ('rdna4', 'cdna5') else ''
            api = 'rj_arith_' + name.lower()
            assert api + '_w64' + arch_suffix in body
            assert 'arithmetic_flags()' in body
            assert (
                'execute_widen32_to_pair'
                if from_integer
                else 'execute_float_conversion'
            ) in body
            assert 'arithmetic_reports_fp_status' not in body
            assert ('RJ_ARITH_ALU_ABS_A' in body) == (vop3 and not from_integer)
            assert ('RJ_ARITH_ALU_NEG_A' in body) == (vop3 and not from_integer)
            assert ('inst_.omod' in body) == vop3
            assert ('RJ_ARITH_FP_FLUSH_OUTPUT_DENORMALS' in body) == output.startswith(
                'F'
            )
        assert gen_conversion64(context(arch, name, ['dst'], [], is_vop3=True)) is None


@pytest.mark.parametrize('arch', ['rdna3', 'rdna4', 'cdna3', 'cdna4', 'cdna5'])
def test_conversion32_modifiers_and_state_policy(arch):
    from amdisa.codegen.execute.rj_arith_float import (
        CONVERSION32_NAMES,
        gen_conversion32,
    )

    for name in CONVERSION32_NAMES:
        to_float = name in ('V_CVT_F32_I32', 'V_CVT_F32_U32')
        fixed_round = name in ('V_CVT_NEAREST_I32_F32', 'V_CVT_FLOOR_I32_F32')
        for vop3 in (False, True):
            body = gen_conversion32(context(arch, name, ['dst'], ['src'], is_vop3=vop3))
            api = 'rj_arith_' + name.lower()
            arch_suffix = (
                '_gfx12' if arch in ('rdna4', 'cdna5') and not fixed_round else ''
            )
            assert api + '_w64' + arch_suffix in body
            assert 'float32_flags' in body
            assert 'arithmetic_flags()' in body
            assert 'arithmetic_reports_fp_status' not in body
            assert ('inst_.omod' in body) == (vop3 and not fixed_round)
            assert ('RJ_ARITH_ALU_ABS_A' in body) == (vop3 and not to_float)
            assert ('RJ_ARITH_ALU_NEG_A' in body) == (vop3 and not to_float)
        assert gen_conversion32(context(arch, name, ['dst'], [], is_vop3=True)) is None


@pytest.mark.parametrize('arch', ['rdna3', 'rdna4'])
def test_conversion16_halves_and_guest_modes(arch):
    from amdisa.codegen.execute.rj_arith_float import (
        CONVERSION16_NAMES,
        gen_conversion16,
    )

    for name in CONVERSION16_NAMES:
        output, source = name.split('_')[-2:]
        for vop3 in (False, True):
            body = gen_conversion16(context(arch, name, ['dst'], ['src'], is_vop3=vop3))
            arch_suffix = '_gfx12' if arch == 'rdna4' else ''
            api = 'rj_arith_' + name.lower()
            assert api + '_w64' + arch_suffix in body
            assert 'execute_float_conversion' in body
            assert 'arithmetic_flags()' in body
            assert 'arithmetic_reports_fp_status' not in body
            assert ('RJ_ARITH_FP16_OVFL' in body) == (output == 'F16')
            assert ('RJ_ARITH_FP_DX10_CLAMP_OFF' in body) == (
                arch == 'rdna3' and output.startswith('F')
            )
            if output.startswith('F'):
                rounding = (
                    'fp_round_mode_f32' if output == 'F32' else 'fp_round_mode_f16_f64'
                )
                assert f'wf.{rounding}()' in body
            assert ('((opsel & 1)' in body) == (vop3 and source != 'F32')
            assert ('((opsel & 8)' in body) == (vop3 and output != 'F32')
            assert ('RJ_ARITH_ALU_ABS_A' in body) == (vop3 and source.startswith('F'))
            assert 'source_bytes, destination_bytes' in body
            assert ('e32_word_operand(src)' in body) == (not vop3)
            assert ('e32_word_operand(dst)' in body) == (not vop3)
            if not vop3:
                assert 'flags, modifiers, word_destination, word_source,' in body
                assert ('src.encoding_value() & 0x80' in body) == (source != 'F32')
                assert ('dst.encoding_value() & 0x80' in body) == (output != 'F32')
        assert gen_conversion16(context(arch, name, ['dst'], [], is_vop3=True)) is None


@pytest.mark.parametrize('arch', ['rdna3', 'rdna4', 'cdna3', 'cdna4', 'cdna5'])
def test_packed_conversion_uses_fixed_rounding_and_optional_state(arch):
    from amdisa.codegen.execute.rj_arith_float import (
        PACKED_CONVERSION_NAMES,
        gen_packed_conversion,
    )

    for name in PACKED_CONVERSION_NAMES:
        for vop3 in (False, True):
            body = gen_packed_conversion(
                context(arch, name, ['dst'], ['a', 'b'], is_vop3=vop3)
            )
            arch_suffix = '_gfx12' if arch in ('rdna4', 'cdna5') else ''
            api = 'rj_arith_' + name.lower()
            assert api + '_w64' + arch_suffix in body
            assert 'arithmetic_flags()' in body
            assert 'arithmetic_reports_fp_status' not in body
            assert 'execute_float_pack' in body
            assert '/*half_sources=*/false' in body
            assert 'RJ_ARITH_ALU_HIGH' not in body
            assert ('RJ_ARITH_ALU_ABS_B' in body) == vop3
            assert ('RJ_ARITH_ALU_NEG_A' in body) == vop3
            assert ('RJ_ARITH_ALU_CLAMP' in body) == vop3
            assert ('RJ_ARITH_ALU_OMOD_2' in body) == (vop3 and '_RTZ_' in name)
        if name == 'V_CVT_PK_RTZ_F16_F32':
            assert gen_packed_conversion(
                context(
                    arch,
                    name.replace('_PK_RTZ_', '_PKRTZ_'),
                    ['dst'],
                    ['a', 'b'],
                    is_vop3=True,
                )
            ) == gen_packed_conversion(
                context(arch, name, ['dst'], ['a', 'b'], is_vop3=True)
            )
        assert (
            gen_packed_conversion(context(arch, name, ['dst'], ['a'], is_vop3=True))
            is None
        )


@pytest.mark.parametrize('arch', ['rdna3', 'rdna4', 'cdna3', 'cdna4', 'cdna5'])
def test_normalized_conversion_layouts_and_state_contract(arch):
    from amdisa.codegen.execute.rj_arith_float import NORMALIZED_NAMES, gen_normalized

    for name in NORMALIZED_NAMES:
        packed = '_PK_' in name
        half = name.endswith('_F16')
        sources = ['a', 'b'] if packed else ['a']
        for vop3 in (False, True):
            body = gen_normalized(context(arch, name, ['d'], sources, is_vop3=vop3))
            api = 'rj_arith_' + name.lower()
            arch_suffix = (
                '_gfx12'
                if packed and half and arch in ('rdna4', 'cdna5')
                else '_gfx942' if not packed and arch in ('cdna3', 'cdna4') else ''
            )
            assert api + '_w64' + arch_suffix in body
            assert 'arithmetic_flags()' in body
            assert 'arithmetic_reports_fp_status' not in body
            assert ('RJ_ARITH_ALU_HIGH_A' in body) == (half and (vop3 or not packed))
            assert ('RJ_ARITH_ALU_HIGH_B' in body) == (packed and half and vop3)
            assert ('RJ_ARITH_ALU_HIGH_D' in body) == (not packed)
            assert 'RJ_ARITH_ALU_OMOD_2' not in body
            assert ('RJ_ARITH_ALU_CLAMP' in body) == (packed and vop3)
            assert 'fp_round_mode' not in body
        if packed and arch in ('cdna3', 'cdna4'):
            assert gen_normalized(
                context(
                    arch,
                    name.replace('_PK_NORM_', '_PKNORM_'),
                    ['d'],
                    sources,
                    is_vop3=True,
                )
            ) == gen_normalized(context(arch, name, ['d'], sources, is_vop3=True))
        assert gen_normalized(context(arch, name, ['d'], [], is_vop3=True)) is None


@pytest.mark.parametrize('arch', ['rdna3', 'rdna4'])
def test_generated_float_pack_methods_have_no_legacy_bypass(arch):
    root = (
        Path(__file__).resolve().parents[3]
        / 'rocjitsu/src/rocjitsu/isa/arch/amdgpu/generated'
        / arch
    )
    checked = 0
    for encoding in ('vop2', 'vop3'):
        source = (root / (encoding + '_exec.cpp')).read_text()
        for match in re.finditer(
            r'void (\w+)::execute(?:_modifier)?_impl\([^)]*\)\s*\{', source
        ):
            end, depth = match.end(), 1
            while depth:
                depth += (source[end] == '{') - (source[end] == '}')
                end += 1
            body = source[match.end() : end - 1]
            if 'arith::execute_float_pack' not in body:
                continue
            assert 'RJ_SIMD' not in body, match[1]
            assert 'try_execute_' not in body, match[1]
            assert 'amdgpu::execute_v_' not in body, match[1]
            checked += 1
    assert checked == 14


@pytest.mark.parametrize('arch', ['rdna3', 'rdna4', 'cdna3', 'cdna4', 'cdna5'])
def test_bit_pack_input_layouts_and_state_policy(arch):
    from amdisa.codegen.execute.rj_arith_float import BIT_PACK_NAMES, gen_bit_pack

    for name in BIT_PACK_NAMES:
        half = name == 'V_PACK_B32_F16'
        sources = ['a', 'b'] if half else ['a', 'b', 'c']
        body = gen_bit_pack(context(arch, name, ['d'], sources, is_vop3=True))
        arch_suffix = '_gfx12' if arch in ('rdna4', 'cdna5') else ''
        api = 'rj_arith_' + name.lower()
        assert api + '_w64' + arch_suffix in body
        assert 'arithmetic_flags()' in body
        assert 'arithmetic_reports_fp_status' not in body
        assert ('execute_float_pack' in body) == half
        assert ('execute_byte_pack' in body) == (not half)
        assert ('RJ_ARITH_ALU_HIGH_B' in body) == half
        assert ('RJ_ARITH_ALU_ABS_B' in body) == half
        assert ('RJ_ARITH_ALU_CLAMP' in body) == (not half)
        assert 'RJ_ARITH_ALU_OMOD' not in body
        assert 'RJ_ARITH_ALU_NEG_C' not in body
        assert 'RJ_ARITH_ALU_HIGH_D' not in body

        assert gen_bit_pack(context(arch, name, ['d'], sources, is_vop3=False)) is None
        assert gen_bit_pack(context(arch, name, ['d'], [], is_vop3=True)) is None


@pytest.mark.parametrize('arch', ['rdna3', 'rdna4', 'cdna3', 'cdna4', 'cdna5'])
def test_fma64_pair_layout_modifiers_and_policy(arch):
    from amdisa.codegen.execute.rj_arith_float import FLOAT_NAMES, gen_float_fma64

    body = gen_float_fma64(
        context(arch, 'V_FMA_F64', ['d'], ['a', 'b', 'c'], is_vop3=True)
    )
    arch_suffix = (
        ('_gfx1250' if arch == 'cdna5' else '_gfx942')
        if arch.startswith('cdna')
        else ('_gfx12' if arch == 'rdna4' else '')
    )
    api = 'rj_arith_v_fma_f64'
    for wave in (32, 64):
        assert api + f'_w{wave}' + arch_suffix + ',' in body
    assert 'execute_float_ternary64' in body
    assert 'd, a, b, c,' in body
    for source in 'ABC':
        assert 'RJ_ARITH_ALU_ABS_' + source in body
        assert 'RJ_ARITH_ALU_NEG_' + source in body
    assert 'RJ_ARITH_ALU_OMOD_2' in body
    assert 'RJ_ARITH_ALU_CLAMP' in body
    assert 'arithmetic_flags()' in body
    assert 'V_FMA_F64' in FLOAT_NAMES
    assert (
        gen_float_fma64(
            context('cdna2', 'V_FMA_F64', ['d'], ['a', 'b', 'c'], is_vop3=True)
        )
        is None
    )
    assert (
        gen_float_fma64(context(arch, 'V_FMA_F64', ['d'], ['a', 'b'], is_vop3=True))
        is None
    )
    assert (
        gen_float_fma64(
            context(arch, 'V_FMA_F64', ['d'], ['a', 'b', 'c'], is_vop3=False)
        )
        is None
    )

    root = (
        Path(__file__).resolve().parents[3]
        / 'rocjitsu/src/rocjitsu/isa/arch/amdgpu/generated'
    )
    source = '\n'.join(p.read_text() for p in (root / arch).glob('vop3_exec*.cpp'))
    matches = list(
        re.finditer(
            r'void VFmaF64Vop3::execute(?:_modifier)?_impl\([^)]*\)\s*\{', source
        )
    )
    assert matches
    for match in matches:
        end, depth = match.end(), 1
        while depth:
            depth += (source[end] == '{') - (source[end] == '}')
            end += 1
        body = source[match.end() : end - 1]
        assert 'execute_float_ternary64' in body
        for wave in (32, 64):
            assert api + f'_w{wave}' + arch_suffix + ',' in body
        assert 'try_execute_' not in body
        assert 'amdgpu::execute_v_' not in body


@pytest.mark.parametrize('arch', ['rdna3', 'rdna4', 'cdna5'])
def test_mixed_fma_formats_abs_partial_outputs_and_routing(arch):
    from amdisa.codegen.execute.rj_arith_float import MIXED_FMA_NAMES, gen_mixed_fma

    for name in MIXED_FMA_NAMES:
        body = gen_mixed_fma(
            context(
                arch,
                name,
                ['d'],
                ['a', 'b', 'c'],
                enc_name='ENC_VOP3P',
                opsel_exprs=('low', 'high'),
                op_sel_hi_2_expr='high_c',
            )
        )
        assert 'execute_mixed_fma' in body
        assert 'arithmetic_flags()' in body
        for operand in 'ABC':
            assert 'RJ_ARITH_MIX_F16_' + operand in body
            assert 'RJ_ARITH_ALU_HIGH_' + operand in body
            assert 'RJ_ARITH_ALU_ABS_' + operand in body
        assert 'inst_.neg_hi' in body and 'is_inline_float_src' in body
        assert 'RJ_ARITH_ALU_OMOD' not in body
        byte_mask = (
            '0xF' if name == 'V_FMA_MIX_F32' else '0xC' if 'MIXHI' in name else '0x3'
        )
        assert 'narrow_sources, ' + byte_mask in body
        assert (
            gen_mixed_fma(
                context(
                    'cdna4',
                    name,
                    ['d'],
                    ['a', 'b', 'c'],
                    enc_name='ENC_VOP3P',
                    opsel_exprs=('low', 'high'),
                    op_sel_hi_2_expr='high_c',
                )
            )
            is None
        )
    root = (
        Path(__file__).resolve().parents[3]
        / 'rocjitsu/src/rocjitsu/isa/arch/amdgpu/generated'
    )
    source = (root / arch / 'vop3p_exec.cpp').read_text()
    checked = 0
    for match in re.finditer(
        r'void (VFmaMix(?:F32|loF16|hiF16)Vop3p)::execute(?:_modifier)?_impl\([^)]*\)\s*\{',
        source,
    ):
        end, depth = match.end(), 1
        while depth:
            depth += (source[end] == '{') - (source[end] == '}')
            end += 1
        body = source[match.end() : end - 1]
        assert 'execute_mixed_fma' in body
        assert 'try_execute_' not in body and 'amdgpu::execute_v_' not in body
        checked += 1
    assert checked == 6


@pytest.mark.parametrize('arch', ['rdna3', 'rdna4'])
def test_fused_public_header_bindings(arch):
    root = (
        Path(__file__).resolve().parents[3]
        / 'rocjitsu/src/rocjitsu/isa/arch/amdgpu/generated'
        / arch
    )
    declarations = public_headers()
    functions = set()
    for path in root.glob('vop*_exec.cpp'):
        functions.update(re.findall(r'\brj_arith_v_[a-z0-9_]+', path.read_text()))
    assert functions
    for function in functions:
        assert re.search(r'\bint ' + function + r'\(', declarations), function


@pytest.mark.parametrize('arch,expected', [('rdna3', 'true'), ('rdna4', 'false')])
@pytest.mark.parametrize(
    'name,gfx12',
    [
        ('V_ADD_F32', True),
        ('V_MUL_F32', True),
        ('V_LDEXP_F32', True),
        ('V_MIN_NUM_F32', False),
        ('V_MAXIMUM_F32', False),
    ],
)
def test_binary32_architecture_argument(arch, expected, name, gfx12):
    body = gen_float_binary32(context(arch, name, ['d'], ['a', 'b'], is_vop3=True))
    if arch == 'rdna3' and not gfx12:
        assert body is None  # These selection mnemonics are RDNA4-only.
        return
    suffix = '_gfx12' if arch == 'rdna4' and gfx12 else ''
    for wave in (32, 64):
        assert f'rj_arith_{name.lower()}_w{wave}{suffix},' in body
    assert f'/*compatibility_mul=*/{str(name == "V_MUL_F32").lower()}' in body
    assert f'/*rdna3=*/{expected}' in body


@pytest.mark.parametrize('arch', ['rdna3', 'rdna4'])
def test_frexp32_architecture_suffix(arch):
    name = 'V_FREXP_EXP_I32_F32'
    from amdisa.codegen.execute.rj_arith_float import gen_float_unary32

    body = gen_float_unary32(context(arch, name, ['d'], ['a']))
    suffix = '_gfx12' if arch == 'rdna4' else ''
    for wave in (32, 64):
        assert f'rj_arith_{name.lower()}_w{wave}{suffix},' in body


@pytest.mark.parametrize('arch', ['rdna3', 'rdna4', 'rdna3_5', 'cdna5'])
@pytest.mark.parametrize(
    'name', ['V_DOT4_I32_IU8', 'V_DOT4_U32_U8', 'V_DOT8_I32_IU4', 'V_DOT8_U32_U4']
)
def test_integer_dot_dispatch(arch, name):
    from amdisa.codegen import CodeGenerator
    from amdisa.gpuisa import Instruction, Operand
    from amdisa.isa_profile import (
        Rdna3Profile,
        Rdna4Profile,
        Rdna3_5Profile,
        Cdna5Profile,
    )
    from amdisa.semantics import derive_semantics

    profiles = {
        'rdna3': Rdna3Profile,
        'rdna4': Rdna4Profile,
        'rdna3_5': Rdna3_5Profile,
        'cdna5': Cdna5Profile,
    }
    generator = object.__new__(CodeGenerator)
    generator.isa_spec = SimpleNamespace(
        arch_name=arch, profile=profiles[arch](), inst_encodings=[], encoding_map={}
    )
    operands = [Operand('vdst', 32, 'OPR_VGPR', False, True, False, False, 0)]
    operands += [
        Operand(f'src{i}', 32, 'OPR_SRC', True, False, False, False, i + 1)
        for i in range(3)
    ]
    inst = Instruction(name, 'ENC_VOP3P', 0, operands)
    body = generator._gen_execute_body(
        inst, derive_semantics(name, 'ENC_VOP3P'), 'ENC_VOP3P'
    )
    if arch not in ('rdna3', 'rdna4'):
        assert 'arith::execute_ternary32' not in body
        assert 'rj_arith_' not in body
        if arch == 'cdna5' and 'DOT4' in name:
            assert 'inst_.clamp && amdgpu::dot4_clamp_supported(wf)' in body
        return
    assert 'arith::execute_ternary32' in body
    assert 'wf.cu().arithmetic_flags(),' in body
    for flag in ('CLAMP', 'LO_A_HIGH', 'LO_B_HIGH', 'HI_A_LOW', 'HI_B_LOW'):
        assert 'RJ_ARITH_DOT_' + flag in body
    for operand in ('A', 'B'):
        assert ('RJ_ARITH_DOT_SIGNED_' + operand in body) == ('_I32_' in name)
    for wave in (32, 64):
        assert f'rj_arith_{name.lower()}_w{wave},' in body
    assert 'ROCJITSU_TRY_SIMD' not in body
    assert 'amdgpu::execute_v_dot' not in body


@pytest.mark.parametrize(
    'arch,encoding',
    [
        ('cdna5', 'ENC_VOP3P'),
        ('rdna3_5', 'ENC_VOP3P'),
        ('rdna3', 'ENC_VOP3'),
        ('rdna4', 'ENC_VOP3'),
    ],
)
def test_integer_dot_helper_declines_other_architectures_and_encodings(arch, encoding):
    from amdisa.codegen.execute.rj_arith_vector import gen_integer_dot

    assert (
        gen_integer_dot(
            context(
                arch,
                'V_DOT4_I32_IU8',
                ['d'],
                ['a', 'b', 'c'],
                enc_name=encoding,
                opsel_exprs=('low', 'high'),
            )
        )
        is None
    )


@pytest.mark.parametrize('arch', ['rdna3', 'rdna4'])
@pytest.mark.parametrize('name', ['V_RCP_IFLAG_F32', 'V_SQRT_F32', 'V_RCP_F32'])
def test_unary_status_scope_dispatch(arch, name):
    from amdisa.codegen import CodeGenerator
    from amdisa.gpuisa import Instruction, Operand
    from amdisa.isa_profile import Rdna3Profile, Rdna4Profile
    from amdisa.semantics import derive_semantics

    generator = object.__new__(CodeGenerator)
    profile = Rdna3Profile() if arch == 'rdna3' else Rdna4Profile()
    generator.isa_spec = SimpleNamespace(
        arch_name=arch, profile=profile, inst_encodings=[], encoding_map={}
    )
    inst = Instruction(
        name,
        'ENC_VOP1',
        0,
        [
            Operand('vdst', 32, 'OPR_VGPR', False, True, False, False, 0),
            Operand('src0', 32, 'OPR_SRC', True, False, False, False, 1),
        ],
    )
    body = generator._gen_execute_body(
        inst, derive_semantics(name, 'ENC_VOP1'), 'ENC_VOP1'
    )
    assert 'arith::execute_float_unary32' in body
    assert ('/*compatibility_all_exec=*/true' in body) == (name != 'V_RCP_F32')


@pytest.mark.parametrize('arch', ['rdna3', 'rdna4', 'cdna5'])
def test_pseudo_scalar_routing_and_semantics(arch):
    from amdisa.codegen.execute.rj_arith_float import (
        PSEUDO_SCALAR_NAMES,
        gen_pseudo_scalar,
    )

    for name in PSEUDO_SCALAR_NAMES:
        body = gen_pseudo_scalar(context(arch, name, ['d'], ['a'], is_vop3=True))
        if arch not in ('rdna4', 'cdna5'):
            assert body is None
            continue
        assert 'execute_pseudo_scalar' in body
        assert 'rj_arith_' + name.lower() in body
        assert ('_gfx1250' in body) == (arch == 'cdna5')
        assert 'arithmetic_flags()' in body
        assert 'arithmetic_reports_fp_status' not in body
        assert 'wave64' not in body and 'exec_mask' not in body
        assert 'OPSEL' not in body and 'HIGH_A' not in body
        assert (
            gen_pseudo_scalar(context(arch, name, ['d'], ['a'], is_vop3=False)) is None
        )
    if arch == 'rdna4':
        root = (
            Path(__file__).resolve().parents[3]
            / 'rocjitsu/src/rocjitsu/isa/arch/amdgpu/generated'
        )
        source = (root / arch / 'vop3_exec.cpp').read_text()
        bodies = re.findall(
            r'void VS(?:Exp|Log|Rcp|Rsq|Sqrt)F(?:16|32)Vop3::execute_impl\([^)]*\)\s*\{([^}]+)\}',
            source,
        )
        assert len(bodies) == 10
        for body in bodies:
            assert 'execute_pseudo_scalar' in body
            assert 'transcendental::' not in body and 'amdgpu::execute_v_' not in body


@pytest.mark.parametrize('arch', ['rdna4', 'cdna3', 'cdna4', 'cdna5'])
def test_f16_to_fp8_migration(arch):
    from amdisa.codegen.execute.rj_arith_float import gen_fp8_narrow

    for stochastic in (False, True):
        for kind in ('FP8', 'BF8'):
            name = 'V_CVT_' + ('SR_' if stochastic else 'PK_') + kind + '_F16'
            body = gen_fp8_narrow(
                context(
                    arch,
                    name,
                    ['d'],
                    ['a', 'seed'] if stochastic else ['a'],
                    is_vop3=True,
                )
            )
            if arch != 'cdna5':
                assert body is None
                continue
            assert 'FP16_OVFL' not in body
            assert 'rj_arith_' + name.lower() + '_w64' in body
            assert ('execute_binary32' in body) == stochastic
            assert ('execute_unary32' in body) == (not stochastic)
            assert ('ALU_HIGH_A' in body) == stochastic
            assert ('ALU_CLAMP' in body) == (kind == 'FP8')
            assert 'destination_bytes' in body


@pytest.mark.parametrize('arch', ['cdna3', 'cdna4', 'cdna5'])
def test_fp8_to_f16_migration(arch):
    from amdisa.codegen.execute.rj_arith_float import gen_fp8_widen

    for packed in (False, True):
        for kind in ('FP8', 'BF8'):
            name = 'V_CVT_' + ('PK_' if packed else '') + 'F16_' + kind
            body = gen_fp8_widen(context(arch, name, ['d'], ['a'], is_vop3=True))
            if arch != 'cdna5':
                assert body is None
                continue
            assert 'execute_unary32' in body
            assert 'FP16_OVFL' in body
            assert 'rj_arith_' + name.lower() + '_w32' in body
            assert ('ALU_HIGH_D' in body) == (not packed)
            assert ('ALU_CLAMP' in body) == (not packed and kind == 'FP8')
            assert ('source_bytes, (selector & 8) ? 0xC : 0x3' in body) == (not packed)


def test_fp8_to_f16_vop1_true16_selection():
    from amdisa.codegen.execute.rj_arith_float import gen_fp8_widen

    generated = (
        Path(__file__).resolve().parents[3]
        / 'rocjitsu/src/rocjitsu/isa/arch/amdgpu/generated/cdna5/vop1_exec.cpp'
    ).read_text()
    for packed in (False, True):
        for kind in ('FP8', 'BF8'):
            name = 'V_CVT_' + ('PK_' if packed else '') + 'F16_' + kind
            body = gen_fp8_widen(context('cdna5', name, ['d'], ['a']))
            assert 'e32_word_operand(a)' in body
            assert 'e32_word_operand(d)' in body
            assert f'e32_packed_vgpr_operand({"a" if packed else "d"})' in body
            assert 'inst_.opsel' not in body
            assert 'ALU_CLAMP' not in body
            assert 'rj_arith_' + name.lower() + '_w32' in generated


@pytest.mark.parametrize('arch', ['cdna3', 'cdna4'])
@pytest.mark.parametrize(
    'name', ['V_MAD_MIX_F32', 'V_MAD_MIXLO_F16', 'V_MAD_MIXHI_F16']
)
def test_cdna_mixed_mad_bridge(arch, name):
    from amdisa.codegen.execute.rj_arith_float import gen_mixed_fma

    body = gen_mixed_fma(
        context(
            arch,
            name,
            ['dst'],
            ['a', 'b', 'c'],
            enc_name='ENC_VOP3P',
            opsel_exprs=('inst_.op_sel', 'inst_.op_sel_hi'),
            op_sel_hi_2_expr='inst_.op_sel_hi_2',
        )
    )
    assert f'rj_arith_{name.lower()}_w32' in body
    assert f'rj_arith_{name.lower()}_w64' in body
    assert 'is_inline_float_src' in body


@pytest.mark.parametrize(
    'name', ['V_FMA_MIX_F32_BF16', 'V_FMA_MIXLO_BF16', 'V_FMA_MIXHI_BF16']
)
def test_cdna_mixed_bf16_preserves_raw_inline_halves(name):
    from amdisa.codegen.execute.rj_arith_float import gen_mixed_fma

    body = gen_mixed_fma(
        context(
            'cdna5',
            name,
            ['d'],
            ['a', 'b', 'c'],
            enc_name='ENC_VOP3P',
            opsel_exprs=('low', 'high'),
            op_sel_hi_2_expr='high_c',
        )
    )
    assert 'execute_mixed_bf16' in body
    assert 'rj_arith_' + name.lower() + '_w32' in body
    assert 'narrow_sources' not in body and 'is_inline_float_src' not in body
    for source in 'ABC':
        assert 'RJ_ARITH_MIX_BF16_' + source in body
        assert 'RJ_ARITH_ALU_ABS_' + source in body
    for arch in ('cdna3', 'cdna4'):
        assert (
            gen_mixed_fma(
                context(
                    arch,
                    name,
                    ['d'],
                    ['a', 'b', 'c'],
                    enc_name='ENC_VOP3P',
                    opsel_exprs=('low', 'high'),
                    op_sel_hi_2_expr='high_c',
                )
            )
            is None
        )


@pytest.mark.parametrize('arch', ['cdna3', 'cdna4'])
def test_cdna_float_dot_bindings(arch):
    from amdisa.codegen.execute.rj_arith_float import gen_float_dot32

    for fmt in ('F16', 'BF16'):
        for accumulate in (False, True):
            name = f'V_DOT2{"C" if accumulate else ""}_F32_{fmt}'
            for encoding in (
                ('ENC_VOP2', 'ENC_VOP3') if accumulate else ('ENC_VOP3P',)
            ):
                body = gen_float_dot32(
                    context(
                        arch,
                        name,
                        ['d'],
                        ['a', 'b'] if accumulate else ['a', 'b', 'c'],
                        enc_name=encoding,
                        opsel_exprs=('low', 'high'),
                    )
                )
                if arch == 'cdna3' and fmt == 'BF16':
                    assert body is None
                    continue
                suffix = 'gfx942' if fmt == 'F16' else 'gfx950'
                assert f'rj_arith_v_dot2_f32_{fmt.lower()}_w64_{suffix},' in body
                assert 'const unsigned replicate_sources = 0;' in body
                if accumulate:
                    assert 'const unsigned narrow_sources = 0;' in body
                    assert 'const uint64_t modifiers = 0;' in body
                    assert ', d, ' in body
                else:
                    assert 'pk16_src_needs_narrowing' in body


@pytest.mark.parametrize('arch', ['cdna3', 'cdna4', 'cdna5'])
def test_packed_f32_fma_pair_selectors(arch):
    from amdisa.codegen.execute.rj_arith_packed_f32 import gen_packed_f32

    body = gen_packed_f32(
        context(
            arch,
            'V_PK_FMA_F32',
            ['dst'],
            ['a', 'b', 'c'],
            enc_name='ENC_VOP3P',
            opsel_exprs=('inst_.op_sel', 'inst_.op_sel_hi'),
            op_sel_hi_2_expr='inst_.op_sel_hi_2',
        )
    )
    assert 'dst, a, b, c' in body
    assert 'inst_.op_sel_hi_2 ? 0u : RJ_ARITH_PK_HI_C_LOW' in body
    assert 'RJ_ARITH_PK_NEG_HI_C' in body
    assert (
        'rj_arith_v_pk_fma_f32_w32_' + ('gfx1250' if arch == 'cdna5' else 'gfx942')
        in body
    )
    assert ('rj_arith_v_pk_fma_f32_w64_gfx942' in body) == (arch != 'cdna5')
    assert (
        gen_packed_f32(context(arch, 'V_PK_FMA_F32', ['dst'], ['a', 'b', 'c'])) is None
    )


@pytest.mark.parametrize('operation', ['Add', 'Mul', 'Fma', 'MinNum', 'MaxNum'])
def test_cdna_packed_bf16_uses_raw_word_adapter(operation):
    root = Path(__file__).resolve().parents[3] / 'rocjitsu/src/rocjitsu/isa/arch/amdgpu'
    source = (root / 'generated/cdna5/vop3p_exec.cpp').read_text()
    body = re.search(
        rf'void VPk{operation}Bf16Vop3p::execute_impl\([^)]*\)\s*\{{(.*?)\n\}}',
        source,
        re.S,
    )[1]
    assert f'execute_packed_bf16<{str(operation == "Fma").lower()}>' in body
    mnemonic = operation.lower().replace('num', '_num')
    assert f'rj_arith_v_pk_{mnemonic}_bf16_w32' in body
    for operand in 'ABC' if operation == 'Fma' else 'AB':
        for flag in ('LO_{}_HIGH', 'HI_{}_LOW', 'NEG_LO_{}', 'NEG_HI_{}'):
            assert 'RJ_ARITH_PK_' + flag.format(operand) in body
    assert 'RJ_ARITH_PK_CLAMP' in body
    assert 'fp_mode::' not in body


@pytest.mark.parametrize('arch', ['cdna3', 'cdna4'])
def test_cdna_byte_accumulator_keeps_legacy_policy(arch):
    from amdisa.codegen.execute.rj_arith_float import gen_bit_pack

    body = gen_bit_pack(
        context(arch, 'V_CVT_PKACCUM_U8_F32', ['d'], ['a', 'b'], is_vop3=True)
    )
    assert 'rj_arith_v_cvt_pkaccum_u8_f32_w64' in body
    assert 'd, a, b, d' in body
    assert 'fp_round_mode' not in body
    assert 'RJ_ARITH_ALU_' not in body


@pytest.mark.parametrize('arch', ['cdna3', 'cdna4', 'cdna5'])
def test_float16_conversion_architectures_and_layouts(arch):
    from amdisa.codegen.execute.rj_arith_float import (
        FLOAT16_CONVERSION_NAMES,
        gen_float16_conversion,
    )

    for name in FLOAT16_CONVERSION_NAMES:
        widen = name == 'V_CVT_F32_BF16'
        packed_sr = name in ('V_CVT_SR_PK_BF16_F32', 'V_CVT_SR_PK_F16_F32')
        single_sr = name in ('V_CVT_SR_BF16_F32', 'V_CVT_SR_F16_F32')
        sources = ['a'] if widen else ['a', 'b', 'seed'] if packed_sr else ['a', 'b']
        for vop3 in (False, True):
            body = gen_float16_conversion(
                context(arch, name, ['d'], sources, is_vop3=vop3)
            )
            supported = (
                arch != 'cdna3'
                and (widen or vop3)
                and not (
                    (single_sr and arch != 'cdna4') or (packed_sr and arch != 'cdna5')
                )
            )
            if not supported:
                assert body is None
                continue
            assert 'rj_arith_' + name.lower() + '_w64' in body
            assert 'FP16_OVFL' in body
            assert ('execute_bf16_widen_sdwa' in body) == (
                widen and not vop3 and arch == 'cdna4'
            )
            assert ('execute_ternary32' in body) == packed_sr
            assert ('? 0xC : 0x3' in body) == (widen or single_sr)
            assert ('inst_.abs & 1' in body) == (not widen and not packed_sr)
            assert ('inst_.abs & 2' in body) == (
                not widen and not packed_sr and not single_sr
            )


@pytest.mark.parametrize('arch', ['cdna3', 'cdna4', 'cdna5'])
def test_cdna_half_conversion_layout_and_sdwa(arch):
    from amdisa.codegen.execute.rj_arith_float import (
        CONVERSION16_NAMES,
        gen_conversion16,
    )

    for name in CONVERSION16_NAMES:
        output = name.split('_')[-2]
        for vop3 in (False, True):
            body = gen_conversion16(context(arch, name, ['dst'], ['src'], is_vop3=vop3))
            suffix = (
                '_gfx12' if arch == 'cdna5' else '_gfx942' if output != 'F32' else ''
            )
            assert 'rj_arith_' + name.lower() + '_w64' + suffix in body
            assert ('execute_conversion16_sdwa' in body) == (
                arch != 'cdna5' and not vop3
            )
            if arch != 'cdna5' and output != 'F32':
                assert 'RJ_ARITH_ALU_HIGH_D) ? 0xC : 0xF' in body


@pytest.mark.parametrize('arch', ['cdna3', 'cdna4', 'cdna5'])
def test_cdna_conversion32_legacy_names_and_sdwa(arch):
    from amdisa.codegen.execute.rj_arith_float import gen_conversion32

    for old, current in [('RPI', 'nearest'), ('FLR', 'floor')]:
        for vop3 in (False, True):
            body = gen_conversion32(
                context(arch, f'V_CVT_{old}_I32_F32', ['dst'], ['src'], is_vop3=vop3)
            )
            assert f'rj_arith_v_cvt_{current}_i32_f32_w64' in body
            assert ('execute_conversion32_sdwa' in body) == (
                arch != 'cdna5' and not vop3
            )


@pytest.mark.parametrize('arch', ['cdna3', 'cdna4', 'cdna5'])
def test_cdna_integer_conversion_bindings(arch):
    from amdisa.codegen.execute.rj_arith_vector import (
        gen_vector32,
        gen_widening_integer,
    )

    for signed in ('I', 'U'):
        name = f'V_CVT_PK_{signed}16_{signed}32'
        body = gen_widening_integer(
            context(arch, name, ['d'], ['a', 'b'], is_vop3=True)
        )
        assert f'rj_arith_{name.lower()}_w64,' in body
        assert '0u, d, a, b);' in body
        name = f'V_CVT_{signed}32_{signed}16'
        body = gen_vector32(context(arch, name, ['d'], ['a']))
        if arch != 'cdna5':
            assert body is None
        else:
            assert f'rj_arith_{name.lower()}_w32,' in body
            assert 'e32_word_operand(a)' in body
            assert 'RJ_ARITH_ALU_HIGH_A' in body


@pytest.mark.parametrize('arch', ['cdna3', 'cdna4', 'cdna5'])
def test_cdna_fp32_arithmetic_routes_guest_modes_and_state(arch):
    from amdisa.codegen.execute.rj_arith_float import CDNA_FP32_ARITHMETIC

    root = (
        Path(__file__).resolve().parents[3]
        / 'rocjitsu/src/rocjitsu/isa/arch/amdgpu/generated'
        / arch
    )
    generated = '\n'.join(p.read_text() for p in root.glob('vop*_exec*.cpp'))
    for name in sorted(CDNA_FP32_ARITHMETIC):
        accumulate = name == 'V_FMAC_F32'
        fma = name in FMA_F32
        sources = ['a', 'b'] if accumulate or not fma else ['a', 'b', 'c']
        emitter = gen_float_fma32 if fma else gen_float_binary32
        body = emitter(context(arch, name, ['d'], sources, is_vop3=True))
        api = 'rj_arith_' + ('v_fma_f32' if fma else name.lower())
        suffix = '_gfx12' if arch == 'cdna5' else ''
        for wave in (32, 64):
            assert api + '_w' + str(wave) + suffix in body
            assert api + '_w' + str(wave) + suffix in generated
        assert 'arithmetic_flags()' in body
        if fma:
            assert ('d, a, b, d' if accumulate else 'd, a, b, c') in body
        else:
            assert 'compatibility_mul=*/' + str(name == 'V_MUL_F32').lower() in body
        if fma:
            assert body.endswith(str(arch != 'cdna5').lower() + ');')
        else:
            assert 'rdna3=*/' + str(arch != 'cdna5').lower() in body


@pytest.mark.parametrize('arch', ['cdna3', 'cdna4', 'cdna5'])
def test_cdna_ldexp64_guest_policy(arch):
    from amdisa.codegen.execute.rj_arith_float import gen_float64_word

    body = gen_float64_word(
        context(arch, 'V_LDEXP_F64', ['d'], ['a', 'exponent'], is_vop3=True)
    )
    suffix = '_gfx12' if arch == 'cdna5' else '_gfx942'
    assert 'rj_arith_v_ldexp_f64_w64' + suffix in body
    assert 'd, a, exponent, ' + str(arch != 'cdna5').lower() in body
    assert 'RJ_ARITH_ALU_NEG_B' not in body
    assert 'RJ_ARITH_ALU_ABS_B' not in body


@pytest.mark.parametrize('arch', ['cdna3', 'cdna4', 'cdna5'])
def test_cdna_unary_methods_use_arithmetic_bridge(arch):
    root = (
        Path(__file__).resolve().parents[3]
        / 'rocjitsu/src/rocjitsu/isa/arch/amdgpu/generated'
        / arch
    )
    body = '\n'.join(p.read_text() for p in root.glob('*exec*.cpp'))
    for name in (
        'VTruncF64',
        'VCeilF64',
        'VRndneF64',
        'VFloorF64',
        'VFractF64',
        'VFrexpMantF64',
        'VFrexpExpI32F64',
        'VFrexpExpI32F32',
    ):
        matches = list(
            re.finditer(r'void ' + name + r'Vop[13]::execute_impl\([^)]*\)\s*\{', body)
        )
        assert len(matches) == 2, (arch, name)
        for match in matches:
            start, end, depth = match.end(), match.end(), 1
            while depth:
                depth += (body[end] == '{') - (body[end] == '}')
                end += 1
            assert any(
                helper in body[start:end]
                for helper in (
                    'amdgpu::arith::execute_float_unary',
                    'amdgpu::arith::execute_cdna_float_unary',
                )
            ), (arch, name)


@pytest.mark.parametrize(
    'arch,name,suffix',
    [
        ('cdna3', 'V_MUL_LEGACY_F32', ''),
        ('cdna4', 'V_MUL_LEGACY_F32', ''),
        ('cdna5', 'V_MUL_DX9_ZERO_F32', '_gfx1250'),
    ],
)
def test_legacy_multiply_migration(arch, name, suffix):
    body = gen_float_binary32(context(arch, name, ['dst'], ['a', 'b'], is_vop3=True))
    for wave in (32, 64):
        assert f'rj_arith_{name.lower()}_w{wave}{suffix}' in body
    assert 'RJ_ARITH_ALU_OMOD_2' in body
    assert '/*compatibility_mul=*/false' in body
    assert f'/*rdna3=*/{str(arch != "cdna5").lower()}' in body
    assert gen_float_binary32(context('cdna2', name, ['dst'], ['a', 'b'])) is None


def test_cdna_dx9_fma_reuses_gfx12_policy():
    body = gen_float_fma32(
        context('cdna5', 'V_FMA_DX9_ZERO_F32', ['d'], ['a', 'b', 'c'], is_vop3=True)
    )
    assert 'execute_float_fma32<true>' in body
    assert 'rj_arith_v_fma_dx9_zero_f32_w32_gfx12' in body
    assert 'arithmetic_flags()' in body
    assert 'RJ_ARITH_ALU_OMOD_2' in body and 'RJ_ARITH_ALU_ABS_C' in body
    for arch in ('cdna3', 'cdna4'):
        assert (
            gen_float_fma32(
                context(
                    arch, 'V_FMA_DX9_ZERO_F32', ['d'], ['a', 'b', 'c'], is_vop3=True
                )
            )
            is None
        )


def test_rdna2_ordinary_fma_keeps_existing_handler():
    from amdisa.codegen.execute.rj_arith_float import gen_float_fma32

    for name, sources, vop3 in (
        ('V_FMA_F32', ['a', 'b', 'c'], True),
        ('V_FMAC_F32', ['a', 'b'], False),
        ('V_FMAC_F32', ['a', 'b'], True),
        ('V_FMAMK_F32', ['a', 'b', 'c'], False),
        ('V_FMAAK_F32', ['a', 'b', 'c'], False),
    ):
        assert (
            gen_float_fma32(context('rdna2', name, ['d'], sources, is_vop3=vop3))
            is None
        )


@pytest.mark.parametrize('arch', ['cdna3', 'cdna4'])
def test_cdna_nan_quieting_adapters(arch):
    from amdisa.codegen.execute.rj_arith_float import (
        gen_float_unary64,
        gen_float_binary32,
    )

    for name in ('TRUNC', 'CEIL', 'RNDNE', 'FLOOR', 'FRACT', 'FREXP_MANT'):
        body = gen_float_unary64(
            context(arch, f'V_{name}_F64', ['d'], ['a'], is_vop3=True)
        )
        for lanes in (32, 64):
            assert f'rj_arith_v_{name.lower()}_f64_w{lanes}_gfx942,' in body
    body = gen_float_binary32(
        context(arch, 'V_LDEXP_F32', ['d'], ['a', 'b'], is_vop3=True)
    )
    for lanes in (32, 64):
        assert f'rj_arith_v_ldexp_f32_w{lanes}_gfx942,' in body


@pytest.mark.parametrize('arch', ['cdna3', 'cdna4', 'cdna5'])
@pytest.mark.parametrize('name', sorted(CDNA_MATH_F32))
@pytest.mark.parametrize('vop3', [False, True])
def test_cdna_math_preserves_destination_and_status_policy(arch, name, vop3):
    body = gen_float_unary32(context(arch, name, ['dst'], ['src'], is_vop3=vop3))
    suffix = 'gfx1250' if arch == 'cdna5' else 'gfx942'
    assert f'rj_arith_{name.lower()}_w32_{suffix}' in body
    assert ('nullptr' if arch == 'cdna5' else f'_w64_{suffix}') in body
    assert 'guest_rounding=*/false' in body
    if name in {'V_RCP_IFLAG_F32', 'V_SQRT_F32'}:
        assert 'compatibility_all_exec=*/true' in body
    if arch != 'cdna5' and not vop3:
        policy = 'MODE' if name in {'V_SIN_F32', 'V_RCP_IFLAG_F32'} else 'FLUSH_NEAREST'
        assert (
            f'execute_cdna_float_unary<amdgpu::sdwa::ResultFormat::F32, amdgpu::sdwa::OutputPolicy::{policy}>'
            in body
        )
    else:
        assert 'execute_cdna_float_unary' not in body


@pytest.mark.parametrize('arch', ['cdna3', 'cdna4', 'cdna5'])
@pytest.mark.parametrize('name', sorted(CDNA_MATH_F64))
@pytest.mark.parametrize('vop3', [False, True])
def test_cdna_fp64_math_uses_shared_guest_policy(arch, name, vop3):
    body = gen_float_unary64(context(arch, name, ['dst'], ['src'], is_vop3=vop3))
    suffix = '_gfx1250' if arch == 'cdna5' else '_gfx942'
    for wave in ((32,) if arch == 'cdna5' else (32, 64)):
        assert f'rj_arith_{name.lower()}_w{wave}{suffix},' in body
    assert 'arithmetic_flags()' in body
    assert ('RJ_ARITH_ALU_OMOD_2' in body) == vop3
    assert ('RJ_ARITH_ALU_ABS_A' in body) == vop3
    assert ('dst, src, false)' if arch == 'cdna5' else 'dst, src, true)') in body
    assert gen_float_unary64(context('cdna2', name, ['dst'], ['src'])) is None


@pytest.mark.parametrize('arch', ['cdna3', 'cdna4', 'cdna5'])
@pytest.mark.parametrize('name', sorted(CDNA_MATH_F16))
@pytest.mark.parametrize('vop3', [False, True])
def test_cdna_half_math_binding(arch, name, vop3):
    body = gen_float_unary16(context(arch, name, ['dst'], ['src'], is_vop3=vop3))
    suffix = 'gfx1250' if arch == 'cdna5' else 'gfx942'
    assert f'rj_arith_{name.lower()}_w32_{suffix}' in body
    assert ('nullptr' if arch == 'cdna5' else f'_w64_{suffix}') in body
    assert 'guest_rounding=*/false' in body
    if arch != 'cdna5':
        if vop3:
            assert 'HalfUnaryWriter{true}' in body
        else:
            assert 'sdwa_half_math_writer(*this)' in body
            assert 'sdwa_omod_' in body and 'sdwa_clamp_' in body


@pytest.mark.parametrize('arch', ['cdna3', 'cdna4'])
@pytest.mark.parametrize('name', ['V_EXP_LEGACY_F32', 'V_LOG_LEGACY_F32'])
@pytest.mark.parametrize('vop3', [False, True])
def test_cdna_legacy_math_keeps_sdwa_policy(arch, name, vop3):
    body = gen_float_unary32(context(arch, name, ['dst'], ['src'], is_vop3=vop3))
    assert 'legacy_f32_w32_gfx942' in body
    assert 'legacy_f32_w64_gfx942' in body
    assert (
        'execute_cdna_float_unary<amdgpu::sdwa::ResultFormat::F32, amdgpu::sdwa::OutputPolicy::FLUSH_NEAREST>'
        in body
    ) == (not vop3)
    assert (
        gen_float_unary32(context('cdna5', name, ['dst'], ['src'], is_vop3=vop3))
        is None
    )


@pytest.mark.parametrize('arch', ['cdna3', 'cdna4', 'cdna5'])
def test_cdna_trig_preop_preserves_omod_rounding(arch):
    from amdisa.codegen.execute.rj_arith_float import gen_float64_word

    body = gen_float64_word(
        context(arch, 'V_TRIG_PREOP_F64', ['dst'], ['a', 'b'], is_vop3=True)
    )
    suffix = 'gfx1250' if arch == 'cdna5' else 'gfx942'
    assert f'rj_arith_v_trig_preop_f64_w32_{suffix}' in body
    assert '/*preop_rounding=*/true' in body
    assert ('nullptr' in body) == (arch == 'cdna5')
    ldexp = gen_float64_word(
        context(arch, 'V_LDEXP_F64', ['dst'], ['a', 'b'], is_vop3=True)
    )
    assert '/*ldexp=*/true' in ldexp
    assert '/*preop_rounding=*/true' not in ldexp

    generated = (
        Path(__file__).resolve().parents[3]
        / 'rocjitsu/src/rocjitsu/isa/arch/amdgpu/generated'
        / arch
    )
    filename = 'vop3_exec_misc.cpp' if arch == 'cdna5' else 'vop3_exec.cpp'
    source = (generated / filename).read_text()
    start = source.index('void VTrigPreopF64Vop3::execute_impl')
    body = source[start : source.index('\n}', start)]
    assert f'rj_arith_v_trig_preop_f64_w32_{suffix}' in body
    assert '/*preop_rounding=*/true' in body


@pytest.mark.parametrize(
    'op', ['RCP', 'RSQ', 'SQRT', 'EXP', 'LOG', 'SIN', 'COS', 'TANH']
)
@pytest.mark.parametrize('vop3', [False, True])
def test_gfx1250_bf16_math_uses_wave32_api(op, vop3):
    from amdisa.codegen.execute.rj_arith_float import gen_float_unary16

    ctx = context('cdna5', f'V_{op}_BF16', ['dst'], ['src'], is_vop3=vop3)
    body = gen_float_unary16(ctx)
    assert f'rj_arith_v_{op.lower()}_bf16_w32_gfx1250, nullptr' in body
    assert '/*guest_rounding=*/false' in body
    assert ('RJ_ARITH_ALU_HIGH_D' in body) == vop3
    assert ('RJ_ARITH_ALU_OMOD_2' in body) == vop3


@pytest.mark.parametrize('width', [16, 32])
@pytest.mark.parametrize('vop3', [False, True])
def test_gfx1250_tanh_uses_wave32_api(width, vop3):
    from amdisa.codegen.execute.rj_arith_float import gen_float_unary16

    handler = gen_float_unary16 if width == 16 else gen_float_unary32
    body = handler(context('cdna5', f'V_TANH_F{width}', ['dst'], ['src'], is_vop3=vop3))
    assert f'rj_arith_v_tanh_f{width}_w32_gfx1250, nullptr' in body
    assert '/*guest_rounding=*/false' in body
    assert ('RJ_ARITH_ALU_OMOD_2' in body) == vop3
