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

# Expected architecture routing of the shared entry points. RDNA3 owns the unsuffixed
# entries, RDNA4 the _gfx12 entries, CDNA3 (gfx942) the _gfx942 entries, and CDNA4
# (gfx950) the _gfx950 entries; all four accept bit-exact semantics. The remaining
# architectures must reach a suffixed loose-only wrapper. These are spelled out rather
# than imported from the generator so that changing its map fails these tests.
CONVERSION_SUFFIX = {
    'rdna3': '',
    'rdna4': '_gfx12',
    'cdna1': '_gfx10',
    'cdna2': '_gfx10',
    'cdna3': '_gfx942',
    'cdna4': '_gfx950',
    'cdna5': '_gfx1250',
    'rdna1': '_gfx10',
    'rdna2': '_gfx10',
    'rdna3_5': '_gfx10',
}

# Byte and nibble conversions predate the per-architecture split: every architecture shares
# the two RDNA entries, which carry no MODE-dependent behaviour to qualify.
BYTE_SUFFIX = {
    'rdna3': '',
    'rdna4': '_gfx12',
    'cdna3': '',
    'cdna4': '',
    'cdna5': '_gfx12',
    'cdna1': '',
    'cdna2': '',
    'rdna1': '',
    'rdna2': '',
    'rdna3_5': '',
}

FMA64_SUFFIX = {
    'rdna2': '_gfx10',
    'rdna3': '',
    'rdna4': '_gfx12',
    'cdna3': '_gfx942',
    'cdna4': '_gfx950',
    'cdna5': '_gfx1250',
    'cdna1': '_gfx950',
    'cdna2': '_gfx950',
    'rdna1': '_gfx950',
    'rdna3_5': '_gfx950',
}


def assert_routes_to(body, api, suffix):
    """Assert both wave entry points, matching complete identifiers.

    A plain substring check lets a longer suffix satisfy a shorter expectation, so an
    exact-capable `_w64_gfx12` would pass a `_w64_gfx1250` expectation and an unsuffixed
    RDNA entry would pass a `_gfx942` one. Either way strict exact could be re-enabled on
    an architecture whose captures have not been reviewed.
    """
    for wave in ('_w32', '_w64'):
        expected = api + wave + suffix
        assert re.search(
            rf'\b{re.escape(expected)}\b', body
        ), f'{expected} not routed; body routes {sorted(set(re.findall(r"rj_arith_\w+", body)))}'


def test_other_architectures_keep_existing_path():
    for arch in ('rdna1', 'rdna2', 'rdna3_5', 'cdna1', 'cdna2'):
        assert (
            gen_float_unary32(context(arch, 'V_CEIL_F32', ['dst'], ['src'])) is not None
        )


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
    ) == ('DX9_ZERO' in name)
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


@pytest.mark.parametrize('arch, count', [('rdna3', 6), ('rdna3_5', 6), ('rdna4', 12)])
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


@pytest.mark.parametrize('arch', ['rdna3', 'rdna3_5', 'rdna4'])
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


@pytest.mark.parametrize('arch', ['rdna3', 'rdna3_5', 'rdna4'])
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
    older = gen_float_binary16(context('rdna1', name, ['d'], ['a', 'b'], is_vop3=True))
    assert 'rj_arith_' + name.lower() + '_w32_' in older


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


@pytest.mark.parametrize('arch', ['rdna1', 'rdna2', 'rdna3_5', 'rdna3', 'rdna4'])
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
        cdna2 = gen_float_ternary_selection(
            context('cdna2', name, ['d'], ['a', 'b', 'c'], is_vop3=True)
        )
        if name in {
            'V_MIN3_F16',
            'V_MAX3_F16',
            'V_MED3_F16',
            'V_MIN3_F32',
            'V_MAX3_F32',
            'V_MED3_F32',
        }:
            assert '_gfx942' in cdna2
        else:
            assert cdna2 is None
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
    inherited = gen_float64_word(
        context('cdna2', name, ['dst'], ['a', 'b'], is_vop3=True)
    )
    # The independent FP64 arithmetic and native-math owners are both present.
    assert inherited is not None
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
            arch_suffix = CONVERSION_SUFFIX[arch]
            api = 'rj_arith_' + name.lower()
            for wave in (32, 64):
                assert api + f'_w{wave}' + arch_suffix + ',' in body
            assert 'execute_float_binary64' in body
            assert 'arithmetic_flags()' in body


@pytest.mark.parametrize(
    'arch',
    [
        'cdna1',
        'cdna2',
        'rdna1',
        'rdna2',
        'rdna3_5',
        'rdna3',
        'rdna4',
        'cdna3',
        'cdna4',
        'cdna5',
    ],
)
def test_byte_conversion_policy_and_selected_read_byte(arch):
    from amdisa.codegen.execute.rj_arith_float import gen_byte_nibble_conversion

    for index in range(5):
        byte = index % 4
        name = f'V_CVT_F32_UBYTE{index}' if index < 4 else 'V_CVT_OFF_F32_I4'
        for vop3 in (False, True):
            body = gen_byte_nibble_conversion(
                context(arch, name, ['dst'], ['src'], is_vop3=vop3)
            )
            assert_routes_to(body, 'rj_arith_' + name.lower(), BYTE_SUFFIX[arch])
            assert 'execute_unary32' in body
            assert f'dst, src, {1 << byte}' in body
            assert 'arithmetic_flags()' in body
            assert 'RJ_ARITH_FP_IEEE_MODE' in body
            assert 'RJ_ARITH_FP_FLUSH_OUTPUT_DENORMALS' in body
            assert ('inst_.omod' in body) == vop3
            assert 'inst_.abs' not in body and 'inst_.neg' not in body
            assert ('execute_byte_conversion_sdwa' in body) == (
                arch in ('cdna1', 'cdna2', 'rdna1', 'rdna2', 'cdna3', 'cdna4')
                and not vop3
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
            ('cdna5', 'gfx1250'),
        ]:
            for vop3 in (False, True):
                body = gen_fp8_widen(
                    context(arch, name, ['dst'], ['src'], is_vop3=vop3)
                )
                assert f'rj_arith_{name.lower()}_w64_{suffix}' in body
                assert ('inst_.clamp' in body) == (
                    vop3 and arch == 'cdna5' and name == 'V_CVT_F32_FP8'
                )
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
                '(wf.fp_denorm_mode_f32() & 2) != (wf.fp_denorm_mode_f16_f64() & 2)'
                in body
            ) == (arch == 'cdna4')
            if arch == 'cdna4':
                assert body.index('UnsupportedOperandValue') < body.index(
                    'execute_binary32'
                )
        assert (
            gen_fp8_narrow(context('rdna4', name, ['dst'], ['a', 'b'], is_vop3=False))
            is None
        )


@pytest.mark.parametrize(
    'arch',
    [
        'cdna1',
        'cdna2',
        'rdna1',
        'rdna2',
        'rdna3_5',
        'rdna3',
        'rdna4',
        'cdna3',
        'cdna4',
        'cdna5',
    ],
)
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
            assert_routes_to(body, 'rj_arith_' + name.lower(), CONVERSION_SUFFIX[arch])
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


@pytest.mark.parametrize(
    'arch',
    [
        'cdna1',
        'cdna2',
        'rdna1',
        'rdna2',
        'rdna3_5',
        'rdna3',
        'rdna4',
        'cdna3',
        'cdna4',
        'cdna5',
    ],
)
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
            # Fixed-rounding forms share one RDNA entry; CDNA still reaches its wrapper.
            arch_suffix = (
                '' if fixed_round and arch == 'rdna4' else CONVERSION_SUFFIX[arch]
            )
            assert_routes_to(body, 'rj_arith_' + name.lower(), arch_suffix)
            assert 'float32_flags' in body
            assert 'arithmetic_flags()' in body
            assert 'arithmetic_reports_fp_status' not in body
            assert ('inst_.omod' in body) == (vop3 and not fixed_round)
            assert ('RJ_ARITH_ALU_ABS_A' in body) == (vop3 and not to_float)
            assert ('RJ_ARITH_ALU_NEG_A' in body) == (vop3 and not to_float)
        assert gen_conversion32(context(arch, name, ['dst'], [], is_vop3=True)) is None


@pytest.mark.parametrize('arch', ['rdna3_5', 'rdna3', 'rdna4'])
def test_conversion16_halves_and_guest_modes(arch):
    from amdisa.codegen.execute.rj_arith_float import (
        CONVERSION16_NAMES,
        gen_conversion16,
    )

    for name in CONVERSION16_NAMES:
        output, source = name.split('_')[-2:]
        for vop3 in (False, True):
            body = gen_conversion16(context(arch, name, ['dst'], ['src'], is_vop3=vop3))
            assert_routes_to(body, 'rj_arith_' + name.lower(), CONVERSION_SUFFIX[arch])
            assert 'execute_float_conversion' in body
            assert 'arithmetic_flags()' in body
            assert 'arithmetic_reports_fp_status' not in body
            assert ('RJ_ARITH_FP16_OVFL' in body) == (output == 'F16')
            assert ('RJ_ARITH_FP_DX10_CLAMP_OFF' in body) == (
                arch != 'rdna4' and output.startswith('F')
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
            assert_routes_to(body, 'rj_arith_' + name.lower(), CONVERSION_SUFFIX[arch])
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
            # Packed-half forms have an RDNA4-specific entry; the other RDNA forms share
            # one entry. Every CDNA form reaches its own loose-only wrapper.
            arch_suffix = (
                CONVERSION_SUFFIX[arch]
                if arch not in ('rdna3', 'rdna4')
                else '_gfx12' if packed and half and arch == 'rdna4' else ''
            )
            assert_routes_to(body, 'rj_arith_' + name.lower(), arch_suffix)
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
        assert_routes_to(body, 'rj_arith_' + name.lower(), CONVERSION_SUFFIX[arch])
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


@pytest.mark.parametrize(
    'arch',
    [
        'rdna1',
        'rdna2',
        'rdna3',
        'rdna3_5',
        'rdna4',
        'cdna1',
        'cdna2',
        'cdna3',
        'cdna4',
        'cdna5',
    ],
)
def test_fma64_pair_layout_modifiers_and_policy(arch):
    from amdisa.codegen.execute.rj_arith_float import FLOAT_NAMES, gen_float_fma64

    body = gen_float_fma64(
        context(arch, 'V_FMA_F64', ['d'], ['a', 'b', 'c'], is_vop3=True)
    )
    arch_suffix = FMA64_SUFFIX[arch]
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
            assert f'rj_arith_v_fma_f64_w{wave}' + FMA64_SUFFIX[arch] + ',' in body
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
    assert 'arith::execute_ternary32' in body
    assert 'wf.cu().arithmetic_flags(),' in body
    for flag in ('CLAMP', 'LO_A_HIGH', 'LO_B_HIGH', 'HI_A_LOW', 'HI_B_LOW'):
        assert 'RJ_ARITH_DOT_' + flag in body
    for operand in ('A', 'B'):
        assert ('RJ_ARITH_DOT_SIGNED_' + operand in body) == ('_I32_' in name)
    suffix = '_gfx1250' if arch == 'cdna5' and 'DOT4' in name else ''
    for wave in (32, 64):
        assert f'rj_arith_{name.lower()}_w{wave}{suffix},' in body
    assert 'ROCJITSU_TRY_SIMD' not in body
    assert 'amdgpu::execute_v_dot' not in body


@pytest.mark.parametrize(
    'arch,encoding',
    [
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


@pytest.mark.parametrize('arch', ['cdna2', 'cdna3', 'cdna4'])
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


@pytest.mark.parametrize('arch', ['cdna2', 'cdna3', 'cdna4', 'cdna5'])
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
    suffix = '_gfx950' if arch == 'cdna4' else '_gfx942'
    assert 'd, a, b, d' in body
    assert f'rj_arith_v_cvt_pkaccum_u8_f32_w64{suffix}' in body
    assert 'fp_round_mode_f32' in body
    assert 'RJ_ARITH_FP_FLUSH_INPUT_DENORMALS' in body
    assert 'RJ_ARITH_ALU_ABS_A' in body
    assert 'RJ_ARITH_ALU_OMOD_2' in body
    assert 'RJ_ARITH_ALU_CLAMP' in body


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
            suffix = '_gfx950' if arch == 'cdna4' else ''
            assert 'rj_arith_' + name.lower() + '_w64' + suffix in body
            assert ('RJ_ARITH_FP_FLUSH_OUTPUT_DENORMALS' in body) == (
                name == 'V_CVT_PK_F16_F32'
            )
            assert ('RJ_ARITH_FP_FLUSH_INPUT_DENORMALS' in body) == (
                arch == 'cdna4' and name in ('V_CVT_PK_BF16_F32', 'V_CVT_SR_BF16_F32')
            )
            assert ('float32_flags' in body) == (arch == 'cdna4' and widen)
            assert 'FP16_OVFL' in body
            assert ('execute_bf16_widen_sdwa' in body) == (
                widen and not vop3 and arch == 'cdna4'
            )
            if widen and not vop3 and arch == 'cdna4':
                assert 'flags, d, a' in body
            assert ('execute_ternary32' in body) == packed_sr
            assert ('? 0xC : 0x3' in body) == (widen or single_sr)
            assert ('inst_.abs & 1' in body) == (
                (widen and vop3 and arch == 'cdna4') or (not widen and not packed_sr)
            )
            assert ('inst_.abs & 2' in body) == (
                not widen and not packed_sr and not single_sr
            )


@pytest.mark.parametrize(
    'arch', ['cdna1', 'cdna2', 'rdna1', 'rdna2', 'cdna3', 'cdna4', 'cdna5']
)
def test_legacy_half_conversion_layout_and_sdwa(arch):
    from amdisa.codegen.execute.rj_arith_float import (
        CONVERSION16_NAMES,
        gen_conversion16,
    )

    for name in CONVERSION16_NAMES:
        output = name.split('_')[-2]
        for vop3 in (False, True):
            body = gen_conversion16(context(arch, name, ['dst'], ['src'], is_vop3=vop3))
            assert_routes_to(body, 'rj_arith_' + name.lower(), CONVERSION_SUFFIX[arch])
            assert ('execute_conversion16_sdwa' in body) == (
                arch != 'cdna5' and not vop3
            )
            if arch != 'cdna5' and not vop3 and output != 'F32':
                assert (
                    'rj_arith_'
                    + name.lower()
                    + '_w64'
                    + CONVERSION_SUFFIX[arch]
                    + ', flags'
                    in body
                )
            if arch not in ('cdna5', 'rdna1', 'rdna2') and output != 'F32':
                assert 'RJ_ARITH_ALU_HIGH_D) ? 0xC : 0xF' in body


@pytest.mark.parametrize(
    'arch', ['cdna1', 'cdna2', 'rdna1', 'rdna2', 'rdna3_5', 'cdna3', 'cdna4', 'cdna5']
)
def test_cdna_conversion32_legacy_names_and_sdwa(arch):
    from amdisa.codegen.execute.rj_arith_float import gen_conversion32

    for old, current in [('RPI', 'nearest'), ('FLR', 'floor')]:
        for vop3 in (False, True):
            body = gen_conversion32(
                context(arch, f'V_CVT_{old}_I32_F32', ['dst'], ['src'], is_vop3=vop3)
            )
            assert f'rj_arith_v_cvt_{current}_i32_f32_w64' in body
            assert ('execute_conversion32_sdwa' in body) == (
                arch not in ('cdna5', 'rdna3_5') and not vop3
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


@pytest.mark.parametrize('arch', ['cdna3', 'cdna4'])
def test_cdna_integer_dot_bindings(arch):
    from amdisa.codegen.execute.rj_arith_vector import (
        gen_integer_dot,
        CDNA_INTEGER_DOT_NAMES,
        ACCUMULATING_INTEGER_DOT_NAMES,
    )

    for name in sorted(CDNA_INTEGER_DOT_NAMES):
        body = gen_integer_dot(
            context(
                arch,
                name,
                ['d'],
                ['a', 'b', 'c'],
                enc_name='ENC_VOP3P',
                opsel_exprs=('low', 'high'),
            )
        )
        assert f'rj_arith_{name.lower()}_w64,' in body
        assert 'RJ_ARITH_DOT_SIGNED_' not in body
        assert 'RJ_ARITH_DOT_CLAMP' in body
    for name in sorted(ACCUMULATING_INTEGER_DOT_NAMES):
        for encoding in ('ENC_VOP2', 'ENC_VOP3'):
            body = gen_integer_dot(
                context(arch, name, ['d'], ['a', 'b'], enc_name=encoding)
            )
            api = name.lower().replace('c_i32', '_i32')
            assert f'rj_arith_{api}_w64,' in body
            assert '0, d, a, b, d);' in body
            assert 'inst_.' not in body


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
        suffix = (
            ('_gfx12' if name == 'V_LDEXP_F32' or fma else '_gfx1250')
            if arch == 'cdna5'
            else '_gfx942' if name == 'V_LDEXP_F32' else ''
        )
        for wave in (32, 64):
            assert api + '_w' + str(wave) + suffix + ',' in body
            assert api + '_w' + str(wave) + suffix in generated
        if name in {'V_ADD_F32', 'V_SUB_F32', 'V_SUBREV_F32', 'V_MUL_F32'}:
            for encoding in ('ENC_VOP2', 'ENC_VOP3'):
                routed = emitter(
                    context(
                        arch,
                        name,
                        ['d'],
                        sources,
                        enc_name=encoding,
                        is_vop3=encoding == 'ENC_VOP3',
                    )
                )
                guard = routed.index(
                    'RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL_NO_FP_STATUS'
                )
                assert 'RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL)' in routed
                assert (
                    guard
                    < routed.index('return;', guard)
                    < routed.index('amdgpu::arith::execute_')
                )
        assert 'arithmetic_flags()' in body
        if fma:
            assert ('d, a, b, d' if accumulate else 'd, a, b, c') in body
        else:
            assert 'compatibility_mul=*/' + str(name == 'V_MUL_F32').lower() in body
        if fma:
            assert body.endswith(str(arch != 'cdna5').lower() + ');')
        else:
            assert 'rdna3=*/' + str(arch != 'cdna5').lower() in body

    for name in ('V_FMAC_F32', 'V_FMAMK_F32', 'V_FMAAK_F32'):
        sources = ['a', 'b'] if name == 'V_FMAC_F32' else ['a', 'b', 'c']
        body = gen_float_fma32(context(arch, name, ['d'], sources))
        for wave in (32, 64):
            api = f'rj_arith_v_fma_f32_w{wave}' + ('_gfx12' if arch == 'cdna5' else '')
            assert api + ',' in body
        assert 'SEMANTICS_BIT_EXACT_EMPIRICAL_NO_FP_STATUS' in body
        source = (root / 'vop2_exec.cpp').read_text()
        method = (
            'void '
            + ''.join(word.title() for word in name.lower().split('_'))
            + 'Vop2::execute_impl'
        )
        method_body = source.split(method, 1)[1].split('\n}', 1)[0]
        for wave in (32, 64):
            suffix = '_gfx12' if arch == 'cdna5' else ''
            assert f'rj_arith_v_fma_f32_w{wave}{suffix},' in method_body


@pytest.mark.parametrize('arch', ['cdna3', 'cdna4', 'cdna5'])
def test_cdna_ldexp64_guest_policy(arch):
    from amdisa.codegen.execute.rj_arith_float import gen_float64_word

    body = gen_float64_word(
        context(arch, 'V_LDEXP_F64', ['d'], ['a', 'exponent'], is_vop3=True)
    )
    suffix = '_gfx12' if arch == 'cdna5' else '_gfx942'
    for wave in (32, 64):
        assert f'rj_arith_v_ldexp_f64_w{wave}{suffix},' in body
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


@pytest.mark.parametrize('arch', ['cdna3', 'cdna4'])
def test_cdna_nan_quieting_adapters(arch):
    from amdisa.codegen.execute.rj_arith_float import (
        gen_float_unary64,
        gen_float_binary32,
    )

    unary64_suffix = '_gfx950' if arch == 'cdna4' else '_gfx942'
    for name in ('TRUNC', 'CEIL', 'RNDNE', 'FLOOR', 'FRACT', 'FREXP_MANT'):
        body = gen_float_unary64(
            context(arch, f'V_{name}_F64', ['d'], ['a'], is_vop3=True)
        )
        for lanes in (32, 64):
            assert f'rj_arith_v_{name.lower()}_f64_w{lanes}{unary64_suffix},' in body
    body = gen_float_binary32(
        context(arch, 'V_LDEXP_F32', ['d'], ['a', 'b'], is_vop3=True)
    )
    for lanes in (32, 64):
        assert f'rj_arith_v_ldexp_f32_w{lanes}_gfx942,' in body


@pytest.mark.parametrize('vop3', [False, True])
def test_gfx1250_mul_encodings_use_target_status_policy(vop3):
    body = gen_float_binary32(
        context('cdna5', 'V_MUL_F32', ['d'], ['a', 'b'], is_vop3=vop3)
    )
    suffix = 'Vop3' if vop3 else 'Vop2'
    filename = 'vop3_exec_alu.cpp' if vop3 else 'vop2_exec.cpp'
    root = (
        Path(__file__).resolve().parents[3]
        / 'rocjitsu/src/rocjitsu/isa/arch/amdgpu/generated/cdna5'
    )
    source = (root / filename).read_text()
    bodies = [body]
    for method in ('execute_impl', 'execute_modifier_impl'):
        start = source.index(f'void VMulF32{suffix}::{method}(')
        bodies.append(source[start : source.index('\n}', start)])
    for body in bodies:
        for wave in (32, 64):
            assert f'rj_arith_v_mul_f32_w{wave}_gfx1250,' in body


@pytest.mark.parametrize('arch', ['cdna1', 'cdna2', 'cdna3', 'cdna4', 'cdna5'])
def test_cdna_binary_selection_policy_mapping(arch):
    from amdisa.codegen.execute.rj_arith_float import (
        SELECTION_F32,
        SELECTION_F64,
        gen_float_binary64,
    )

    root = (
        Path(__file__).resolve().parents[3]
        / 'rocjitsu/src/rocjitsu/isa/arch/amdgpu/generated'
        / arch
    )
    generated = '\n'.join(p.read_text() for p in root.glob('vop*_exec*.cpp'))
    for names, emitter in [
        (SELECTION_F32[arch], gen_float_binary32),
        (SELECTION_F64[arch], gen_float_binary64),
    ]:
        for name in names:
            body = emitter(context(arch, name, ['d'], ['a', 'b'], is_vop3=True))
            suffix = '_gfx942' if arch != 'cdna5' else ''
            for wave in (32, 64):
                api = 'rj_arith_' + name.lower() + '_w' + str(wave) + suffix
                assert api in body
                assert api in generated
            assert 'RJ_ARITH_ALU_OMOD_2' in body
            assert 'RJ_ARITH_ALU_CLAMP' in body


@pytest.mark.parametrize('arch', ['cdna3', 'cdna4', 'cdna5'])
def test_cdna_ternary_selection_mapping(arch):
    from amdisa.codegen.execute.rj_arith_float import (
        CDNA_TERNARY_SELECTION,
        TERNARY_SELECTION_RDNA4,
        gen_float_ternary_selection,
    )

    root = (
        Path(__file__).resolve().parents[3]
        / 'rocjitsu/src/rocjitsu/isa/arch/amdgpu/generated'
        / arch
    )
    generated = '\n'.join(p.read_text() for p in root.glob('vop3_exec*.cpp'))
    for name in (
        TERNARY_SELECTION_RDNA4 if arch == 'cdna5' else CDNA_TERNARY_SELECTION
    ):
        body = gen_float_ternary_selection(
            context(arch, name, ['d'], ['a', 'b', 'c'], is_vop3=True)
        )
        for wave in (32, 64):
            api = (
                'rj_arith_'
                + name.lower()
                + '_w'
                + str(wave)
                + (
                    '_gfx950'
                    if arch == 'cdna4'
                    else '_gfx942' if arch == 'cdna3' else ''
                )
            )
            assert api in body
            assert api in generated
        if name.endswith('_F16'):
            assert (', true, true);' in body) == (arch == 'cdna3')
        assert 'RJ_ARITH_ALU_NEG_C' in body


@pytest.mark.parametrize('arch', ['cdna3', 'cdna4'])
def test_cdna_half_selection_vop2_clears_upper_half(arch):
    from amdisa.codegen.execute.rj_arith_float import gen_float_binary16

    for name in ('V_MIN_F16', 'V_MAX_F16'):
        body = gen_float_binary16(context(arch, name, ['d'], ['a', 'b']))
        assert 'd, a, b, true, true);' in body


@pytest.mark.parametrize('arch', ['cdna3', 'cdna4', 'cdna5'])
def test_cdna_half_selection_policy_mapping(arch):
    from amdisa.codegen.execute.rj_arith_float import (
        gen_float_binary16,
        gen_packed_binary16,
    )

    root = (
        Path(__file__).resolve().parents[3]
        / 'rocjitsu/src/rocjitsu/isa/arch/amdgpu/generated'
        / arch
    )
    generated = '\n'.join(p.read_text() for p in root.glob('vop*_exec*.cpp'))
    for packed, emitter in [(False, gen_float_binary16), (True, gen_packed_binary16)]:
        for op in (
            ('MIN_NUM', 'MAX_NUM', 'MINIMUM', 'MAXIMUM')
            if arch == 'cdna5'
            else ('MIN', 'MAX')
        ):
            name = 'V_' + ('PK_' if packed else '') + op + '_F16'
            body = emitter(
                context(
                    arch,
                    name,
                    ['d'],
                    ['a', 'b'],
                    is_vop3=not packed,
                    enc_name='ENC_VOP3P' if packed else 'ENC_VOP3',
                    opsel_exprs=('low', 'high'),
                )
            )
            suffix = (
                '_gfx950' if arch == 'cdna4' else '_gfx942' if arch == 'cdna3' else ''
            )
            for wave in (32, 64):
                api = 'rj_arith_' + name.lower() + '_w' + str(wave) + suffix
                assert api in body
                assert api in generated
            assert 'CLAMP' in body


@pytest.mark.parametrize('arch', ['cdna4', 'cdna5'])
@pytest.mark.parametrize('op', ['MINIMUM3', 'MAXIMUM3', 'MIN3_NUM', 'MAX3_NUM'])
def test_cdna_packed_ternary_selection(arch, op):
    from amdisa.codegen.execute.rj_arith_float import gen_packed_fma16

    name = f'V_PK_{op}_F16'
    body = gen_packed_fma16(
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
    if arch == 'cdna4' and '_NUM' in op:
        assert body is None
        return
    suffix = '_gfx950' if arch == 'cdna4' else ''
    assert_routes_to(body, f'rj_arith_{name.lower()}', suffix)
    for operand in 'ABC':
        assert f'RJ_ARITH_PK_NEG_HI_{operand}' in body
        assert f'RJ_ARITH_PK_LO_{operand}_HIGH' in body
    assert 'high_c ? 0u : RJ_ARITH_PK_HI_C_LOW' in body
    assert 'narrow_sources, replicate_sources' in body


@pytest.mark.parametrize('name', ['V_MINIMUM3_F32', 'V_MAXIMUM3_F32'])
def test_cdna4_propagating_selection(name):
    from amdisa.codegen.execute.rj_arith_float import gen_float_ternary_selection

    body = gen_float_ternary_selection(
        context('cdna4', name, ['d'], ['a', 'b', 'c'], is_vop3=True)
    )
    for wave in (32, 64):
        assert f'rj_arith_{name.lower()}_w{wave}_gfx950' in body
    assert 'a, b, c, true);' in body
    assert (
        gen_float_ternary_selection(
            context('cdna3', name, ['d'], ['a', 'b', 'c'], is_vop3=True)
        )
        is None
    )


@pytest.mark.parametrize('arch', ['cdna1', 'cdna2', 'cdna3', 'cdna4', 'cdna5'])
def test_cdna_half_arithmetic_modes_and_destination(arch):
    from amdisa.codegen.execute.rj_arith_float import (
        gen_float_binary16,
        BINARY_ARITHMETIC_F16,
    )

    for name in sorted(BINARY_ARITHMETIC_F16):
        body = gen_float_binary16(context(arch, name, ['d'], ['a', 'b'], is_vop3=True))
        if arch == 'cdna5':
            suffix = '_gfx12' if name == 'V_LDEXP_F16' else '_gfx1250'
        else:
            suffix = (
                '_gfx942' if arch == 'cdna3' or name == 'V_LDEXP_F16' else '_gfx908'
            )
        if arch == 'cdna5' and name == 'V_MUL_F16':
            suffix = '_gfx1250'
        for lanes in (32, 64):
            assert f'rj_arith_{name.lower()}_w{lanes}{suffix},' in body
        assert (
            'd, a, b, false);' if arch == 'cdna5' else 'd, a, b, true, true);'
        ) in body
        assert ('RJ_ARITH_ALU_NEG_B' in body) == (name != 'V_LDEXP_F16')
        assert 'RJ_ARITH_ALU_HIGH_B' in body
        body = gen_float_binary16(context(arch, name, ['d'], ['a', 'b']))
        assert (', true, true);' in body) == (arch != 'cdna5')
        assert ('execute_float_binary16_sdwa' in body) == (arch != 'cdna5')
    selection = gen_float_binary16(context(arch, 'V_MIN_F16', ['d'], ['a', 'b']))
    assert (selection is not None) == (arch != 'cdna5')
    if selection:
        assert 'execute_sdwa_selection<16>' in selection
        assert 'execute_float_binary16_sdwa' not in selection


@pytest.mark.parametrize('arch', ['cdna3', 'cdna4', 'cdna5'])
@pytest.mark.parametrize('vop3', [False, True])
def test_cdna_half_exponent_bindings(arch, vop3):
    from amdisa.codegen.execute.rj_arith_float import gen_float_unary16

    body = gen_float_unary16(
        context(arch, 'V_FREXP_EXP_I16_F16', ['d'], ['a'], is_vop3=vop3)
    )
    suffix = '_gfx12' if arch == 'cdna5' else ''
    assert 'rj_arith_v_frexp_exp_i16_f16_w64' + suffix in body
    assert ('execute_half_exponent_sdwa' in body) == (arch != 'cdna5' and not vop3)
    operands = (
        'word_destination, word_source' if arch == 'cdna5' and not vop3 else 'd, a'
    )
    assert (
        f'{operands}, true, /*guest_rounding=*/false, /*zero_extend_low=*/true, amdgpu::arith::HalfUnaryWriter{{true}}'
        if arch != 'cdna5'
        else f'{operands}, false, /*guest_rounding=*/false, /*zero_extend_low=*/false'
    ) in body
    assert 'RJ_ARITH_ALU_OMOD' not in body


@pytest.mark.parametrize('arch', ['cdna3', 'cdna4', 'cdna5'])
def test_cdna_half_fma_policy_and_destination(arch):
    from amdisa.codegen.execute.rj_arith_float import gen_float_fma16

    body = gen_float_fma16(
        context(arch, 'V_FMA_F16', ['d'], ['a', 'b', 'c'], is_vop3=True)
    )
    suffix = '_gfx12' if arch == 'cdna5' else ''
    for lanes in (32, 64):
        assert f'rj_arith_v_fma_f16_w{lanes}{suffix},' in body
    assert (
        'd, a, b, c, false);' if arch == 'cdna5' else 'd, a, b, c, true, true);'
    ) in body
    assert 'RJ_ARITH_ALU_HIGH_C' in body
    # Legacy MAD has a different intermediate rounding contract.
    assert gen_float_fma16(context(arch, 'V_MAD_F16', ['d'], ['a', 'b', 'c'])) is None
    if arch == 'cdna5':
        body = gen_float_fma16(
            context(arch, 'V_FMAC_F16', ['d'], ['a', 'b'], is_vop3=True)
        )
        assert 'd, a, b, d, false);' in body
        assert 'RJ_ARITH_ALU_HIGH_C' in body


@pytest.mark.parametrize('arch', ['cdna3', 'cdna4'])
@pytest.mark.parametrize('name', ['V_FMA_LEGACY_F16', 'V_MAC_F16'])
def test_cdna_legacy_half_fma_reuses_architectural_model(arch, name):
    from amdisa.codegen.execute.rj_arith_float import gen_float_fma16

    inputs = ['a', 'b'] if name == 'V_MAC_F16' else ['a', 'b', 'c']
    body = gen_float_fma16(context(arch, name, ['d'], inputs, is_vop3=True))
    assert 'rj_arith_v_fma_f16_w32' in body
    assert 'rj_arith_v_fma_f16_w64' in body
    assert ('d, a, b, d' if name == 'V_MAC_F16' else 'd, a, b, c') in body
    assert 'true, true' in body
    if name == 'V_MAC_F16':
        body = gen_float_fma16(context(arch, name, ['d'], inputs, enc_name='ENC_VOP2'))
        assert 'd, a, b, d' in body


@pytest.mark.parametrize('arch', ['cdna1', 'cdna2', 'cdna3', 'cdna4'])
@pytest.mark.parametrize('name', ['V_MAD_F16', 'V_MAD_LEGACY_F16'])
def test_cdna_half_mad_preserves_its_distinct_intermediate(arch, name):
    from amdisa.codegen.execute.rj_arith_float import gen_float_fma16

    body = gen_float_fma16(context(arch, name, ['d'], ['a', 'b', 'c'], is_vop3=True))
    assert 'rj_arith_' + name.lower() + '_w64' in body
    assert 'rj_arith_v_fma_f16' not in body
    assert 'RJ_ARITH_ALU_HIGH_D' in body and 'RJ_ARITH_ALU_ABS_C' in body
    assert 'RJ_ARITH_ALU_OMOD_2' in body
    assert 'd, a, b, c, true, true' in body


@pytest.mark.parametrize('arch', ['cdna1', 'cdna2', 'cdna3', 'cdna4'])
@pytest.mark.parametrize('name', ['V_MADMK_F16', 'V_MADAK_F16'])
def test_cdna_literal_half_mad_reuses_mad_arithmetic(arch, name):
    from amdisa.codegen.execute.rj_arith_float import gen_float_fma16

    sources = ['a', 'literal', 'b'] if name == 'V_MADMK_F16' else ['a', 'b', 'literal']
    body = gen_float_fma16(context(arch, name, ['d'], sources))
    assert 'rj_arith_v_mad_f16_w64' in body
    assert 'arithmetic_flags(), 0,' in body
    assert 'd, ' + ', '.join(sources) + ', true, true' in body
    assert gen_float_fma16(context(arch, name, ['d'], sources, is_vop3=True)) is None


@pytest.mark.parametrize('arch', ['cdna3', 'cdna4', 'cdna5'])
@pytest.mark.parametrize('encoding', ['ENC_VOP2', 'ENC_VOP3'])
def test_cdna_packed_fmac_retains_modifiers_and_omod(arch, encoding):
    from amdisa.codegen.execute.rj_arith_float import gen_packed_fma16

    body = gen_packed_fma16(
        context(arch, 'V_PK_FMAC_F16', ['d'], ['a', 'b'], enc_name=encoding)
    )
    suffix = '_gfx12' if arch == 'cdna5' else '_gfx942'
    for wave in (32, 64):
        assert f'rj_arith_v_pk_fmac_f16_w{wave}{suffix},' in body
    assert 'execute_packed_fma16<true>' in body
    assert 'd, a, b, d' in body
    assert ('RJ_ARITH_ALU_OMOD_2' in body) == (arch != 'cdna5')
    assert ('RJ_ARITH_ALU_ABS_A' in body) == (encoding == 'ENC_VOP3')
    if encoding == 'ENC_VOP2':
        assert 'inst_.src1' not in body
        assert 'sdwa::output_modifier' not in body
        assert ('inst_.src0 == amdgpu::SRC_SDWA ? sdwa_omod_ : 0u' in body) == (
            arch != 'cdna5'
        )
    assert 'dot2_src_needs_half_replication(inst_.src0)' in body


@pytest.mark.parametrize('arch', ['cdna3', 'cdna4', 'cdna5'])
def test_cdna_packed_half_arithmetic(arch):
    from amdisa.codegen.execute.rj_arith_float import (
        gen_packed_binary16,
        gen_packed_fma16,
    )

    for name in ('V_PK_ADD_F16', 'V_PK_MUL_F16', 'V_PK_FMA_F16'):
        fma = name == 'V_PK_FMA_F16'
        emit = gen_packed_fma16 if fma else gen_packed_binary16
        body = emit(
            context(
                arch,
                name,
                ['d'],
                ['a', 'b', 'c'] if fma else ['a', 'b'],
                enc_name='ENC_VOP3P',
                opsel_exprs=('inst_.lo', 'inst_.hi'),
                op_sel_hi_2_expr='inst_.hi_c',
            )
        )
        suffix = (
            ('_gfx1250' if arch == 'cdna5' else '_gfx942')
            if name == 'V_PK_MUL_F16' or (name == 'V_PK_ADD_F16' and arch != 'cdna5')
            else ('_gfx12' if arch == 'cdna5' else '')
        )
        for lanes in (32, 64):
            assert f'rj_arith_{name.lower()}_w{lanes}{suffix},' in body
        assert 'pk16_src_needs_narrowing' in body
        assert 'RJ_ARITH_PK_HI_A_LOW' in body
        assert 'RJ_ARITH_PK_NEG_HI_B' in body
    # Promoted FMAC has a separate modifier policy from VOP3P FMA.
    assert (
        gen_packed_fma16(
            context(arch, 'V_PK_FMAC_F16', ['d'], ['a', 'b'], enc_name='ENC_VOP3')
        )
        is not None
    )


@pytest.mark.parametrize('arch', ['cdna3', 'cdna4', 'cdna5'])
def test_cdna_half_fma_qualification_precedes_execution(arch):
    from amdisa.codegen.execute.rj_arith_float import (
        gen_float_fma16,
        gen_packed_binary16,
        gen_packed_fma16,
    )

    for name, emit, sources in (
        ('V_FMA_F16', gen_float_fma16, ['a', 'b', 'c']),
        ('V_PK_FMA_F16', gen_packed_fma16, ['a', 'b', 'c']),
        ('V_PK_FMAC_F16', gen_packed_fma16, ['a', 'b']),
        ('V_PK_ADD_F16', gen_packed_binary16, ['a', 'b']),
    ):
        body = emit(context(arch, name, ['d'], sources, enc_name='ENC_VOP3P'))
        if name == 'V_PK_ADD_F16' and arch != 'cdna5':
            continue  # Its CDNA API already rejects strict requests.
        guard = body.index('RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL_NO_FP_STATUS')
        assert 'RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL)' in body
        assert guard < body.index('amdgpu::arith::execute_')


@pytest.mark.parametrize('arch', ['cdna3', 'cdna4', 'cdna5'])
@pytest.mark.parametrize('vop3', [False, True])
def test_cdna_half_exponent_qualification(arch, vop3):
    from amdisa.codegen.execute.rj_arith_float import (
        gen_float_binary16,
        gen_float_unary16,
    )

    cases = [('V_FREXP_EXP_I16_F16', gen_float_unary16, ['a'])]
    if arch == 'cdna5':
        cases.append(('V_LDEXP_F16', gen_float_binary16, ['a', 'b']))
    for name, emit, sources in cases:
        body = emit(context(arch, name, ['d'], sources, is_vop3=vop3))
        assert body.index(
            'RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL_NO_FP_STATUS'
        ) < body.index('amdgpu::arith::execute_')


def test_scaled_widen_gfx950_bindings_and_gates():
    from amdisa.codegen.execute.rj_arith_float import (
        SCALED_WIDEN_NAMES,
        gen_scaled_widen,
    )

    generated = (
        Path(__file__).resolve().parents[3]
        / 'rocjitsu/src/rocjitsu/isa/arch/amdgpu/generated/cdna4/vop3_exec.cpp'
    ).read_text()
    assert (
        sum(
            generated.count('rj_arith_' + name.lower() + '_w')
            for name in SCALED_WIDEN_NAMES
        )
        == 26
    )
    assert len(SCALED_WIDEN_NAMES) == 13
    for name in sorted(SCALED_WIDEN_NAMES):
        body = gen_scaled_widen(
            context('cdna4', name, ['dst'], ['a', 'b'], is_vop3=True)
        )
        assert 'rj_arith_' + name.lower() + '_w64_gfx950' in body
        assert 'rj_arith_' + name.lower() + '_w64_gfx950' in generated
        pair = '_PK_F32_' in name
        assert (
            f'execute_scaled_conversion<false, false, false, {str(pair).lower()}>'
            in body
        )
        assert 'RJ_ARITH_FP_FLUSH_OUTPUT_DENORMALS' in body
        expected_denorm = (
            'fp_denorm_mode_f16_f64' if '_F16_' in name else 'fp_denorm_mode_f32'
        )
        other_denorm = (
            'fp_denorm_mode_f32' if '_F16_' in name else 'fp_denorm_mode_f16_f64'
        )
        assert f'!(wf.{expected_denorm}() & 2)' in body
        assert other_denorm not in body
        assert (
            'uint64_t(inst_.op_sel & 3) << 16' in body
            if '_PK_' not in name or name.endswith('_FP4')
            else 'RJ_ARITH_ALU_HIGH_A' in body
        )
        if '_PK_' not in name and '_F16_' in name:
            assert 'RJ_ARITH_ALU_HIGH_D' in body
            assert '? 0xC : 0x3' in body
        for arch in ('rdna4', 'cdna3', 'cdna5'):
            assert (
                gen_scaled_widen(context(arch, name, ['dst'], ['a', 'b'], is_vop3=True))
                is None
            )
        assert (
            gen_scaled_widen(context('cdna4', name, ['dst'], ['a'], is_vop3=True))
            is None
        )
        assert gen_scaled_widen(context('cdna4', name, ['dst'], ['a', 'b'])) is None


def test_scaled_fp6_widen_bindings_and_gates():
    from amdisa.codegen.execute.rj_arith_float import (
        SCALED_FP6_WIDEN_NAMES,
        gen_wide_scaled_widen,
    )

    generated = (
        Path(__file__).resolve().parents[3]
        / 'rocjitsu/src/rocjitsu/isa/arch/amdgpu/generated/cdna4/vop3_exec.cpp'
    ).read_text()
    assert len(SCALED_FP6_WIDEN_NAMES) == 6
    for name in SCALED_FP6_WIDEN_NAMES:
        body = gen_wide_scaled_widen(
            context('cdna4', name, ['d'], ['a', 'b'], is_vop3=True)
        )
        assert 'rj_arith_' + name.lower() + '_w64_gfx950' in body
        assert 'rj_arith_' + name.lower() + '_w64_gfx950' in generated
        assert 'execute_wide_scaled_widen' in body
        expected_denorm = (
            'fp_denorm_mode_f16_f64' if '_F16_' in name else 'fp_denorm_mode_f32'
        )
        other_denorm = (
            'fp_denorm_mode_f32' if '_F16_' in name else 'fp_denorm_mode_f16_f64'
        )
        assert f'!(wf.{expected_denorm}() & 2)' in body
        assert other_denorm not in body
        assert 'source_base, b, ' + ('32' if '_F32_' in name else '16') in body
        assert 'inst_.op_sel' not in body and 'inst_.neg' not in body
        assert body.count('resolved_vgpr_offset') == 2
        for arch in ('cdna3', 'cdna5', 'rdna4'):
            assert (
                gen_wide_scaled_widen(
                    context(arch, name, ['d'], ['a', 'b'], is_vop3=True)
                )
                is None
            )
        assert (
            gen_wide_scaled_widen(context('cdna4', name, ['d'], ['a'], is_vop3=True))
            is None
        )
        assert gen_wide_scaled_widen(context('cdna4', name, ['d'], ['a', 'b'])) is None


def test_scaled_fp6_widen_gfx1250_bindings_and_gates():
    from amdisa.codegen.execute.rj_arith_float import (
        WIDE_SCALED_WIDEN_GFX1250_NAMES,
        gen_wide_scaled_widen,
    )

    generated = (
        Path(__file__).resolve().parents[3]
        / 'rocjitsu/src/rocjitsu/isa/arch/amdgpu/generated/cdna5/vop3_exec_cvt.cpp'
    ).read_text()
    assert len(WIDE_SCALED_WIDEN_GFX1250_NAMES) == 15
    for name in WIDE_SCALED_WIDEN_GFX1250_NAMES:
        body = gen_wide_scaled_widen(
            context('cdna5', name, ['d'], ['a', 'b'], is_vop3=True)
        )
        assert 'execute_wide_scaled_widen<true>' in body
        assert (
            'resolved_vgpr_offset(wf, a.opr_type_, a.encoding_value_, a.vgpr_msb_role())'
            in body
        )
        assert 'apply_gpr_idx' not in body
        assert 'CvtScaleSelection::' in body
        assert 'rj_arith_' + name.lower() + '_w32_gfx1250' in generated
        elements = 8 if '_PK8_' in name else 16
        assert (
            'source_base, b, ' + str(elements if '_F32_' in name else elements // 2)
            in body
        )
        for arch in ('cdna3', 'cdna4', 'rdna4'):
            assert (
                gen_wide_scaled_widen(
                    context(arch, name, ['d'], ['a', 'b'], is_vop3=True)
                )
                is None
            )


def test_scaled_narrow_gfx950_bindings_and_gates():
    from amdisa.codegen.execute.rj_arith_float import (
        SCALED_NARROW_NAMES,
        gen_scaled_narrow,
    )

    generated = (
        Path(__file__).resolve().parents[3]
        / 'rocjitsu/src/rocjitsu/isa/arch/amdgpu/generated/cdna4/vop3_exec.cpp'
    ).read_text()
    assert (
        sum(
            generated.count('rj_arith_' + name.lower() + '_w')
            for name in SCALED_NARROW_NAMES
        )
        == 36
    )
    assert len(SCALED_NARROW_NAMES) == 18
    for name in sorted(SCALED_NARROW_NAMES):
        fp4 = '_FP4_' in name
        stochastic = '_SR_' in name
        f32 = name.endswith('_F32')
        sources = ['a', 'b', 'c'] if stochastic or f32 else ['a', 'b']
        body = gen_scaled_narrow(context('cdna4', name, ['dst'], sources, is_vop3=True))
        assert 'rj_arith_' + name.lower() + '_w64_gfx950' in body
        assert 'rj_arith_' + name.lower() + '_w64_gfx950' in generated
        ternary = len(sources) == 3
        pair_a = fp4 and stochastic and f32
        special_policy = f', false, true, {str(stochastic).lower()}' if fp4 else ''
        assert (
            f'execute_scaled_conversion<true, {str(ternary).lower()}, {str(pair_a).lower()}{special_policy}>'
            in body
        )
        assert 'RJ_ARITH_FP_FLUSH_INPUT_DENORMALS' in body
        expected_denorm = (
            'fp_denorm_mode_f16_f64' if name.endswith('_F16') else 'fp_denorm_mode_f32'
        )
        other_denorm = (
            'fp_denorm_mode_f32' if name.endswith('_F16') else 'fp_denorm_mode_f16_f64'
        )
        assert f'!(wf.{expected_denorm}() & 1)' in body
        assert other_denorm not in body
        source_bytes = '0x3' if stochastic and not f32 and not fp4 else '0xF'
        assert f'{source_bytes}, destination_bytes' in body
        assert (
            '(inst_.op_sel >> 2) & 3' in body
            if stochastic or fp4
            else 'RJ_ARITH_ALU_HIGH_D' in body
        )
        for arch in ('rdna4', 'cdna3', 'cdna5'):
            assert (
                gen_scaled_narrow(context(arch, name, ['dst'], sources, is_vop3=True))
                is None
            )
        assert (
            gen_scaled_narrow(
                context('cdna4', name, ['dst'], sources[:-1], is_vop3=True)
            )
            is None
        )
        assert gen_scaled_narrow(context('cdna4', name, ['dst'], sources)) is None


def test_scaled_fp6_narrow_bindings_and_gates():
    from amdisa.codegen.execute.rj_arith_float import (
        SCALED_FP6_NARROW_NAMES,
        gen_wide_scaled_narrow,
    )

    generated = (
        Path(__file__).resolve().parents[3]
        / 'rocjitsu/src/rocjitsu/isa/arch/amdgpu/generated/cdna4/vop3_exec.cpp'
    ).read_text()
    assert len(SCALED_FP6_NARROW_NAMES) == 12
    for name in SCALED_FP6_NARROW_NAMES:
        ternary = '_SR_' in name or name.endswith('_F32')
        sources = ['a', 'b', 'c'] if ternary else ['a', 'b']
        body = gen_wide_scaled_narrow(
            context('cdna4', name, ['d'], sources, is_vop3=True)
        )
        assert f'execute_wide_scaled_narrow<{str(ternary).lower()}>' in body
        expected_denorm = (
            'fp_denorm_mode_f16_f64' if name.endswith('_F16') else 'fp_denorm_mode_f32'
        )
        other_denorm = (
            'fp_denorm_mode_f32' if name.endswith('_F16') else 'fp_denorm_mode_f16_f64'
        )
        assert f'!(wf.{expected_denorm}() & 1)' in body
        assert other_denorm not in body
        assert 'rj_arith_' + name.lower() + '_w64_gfx950' in generated
        assert 'resolved_vgpr_offset' in body
        for arch in ('rdna3', 'rdna4', 'cdna3', 'cdna5'):
            assert (
                gen_wide_scaled_narrow(
                    context(arch, name, ['d'], sources, is_vop3=True)
                )
                is None
            )
        assert gen_wide_scaled_narrow(context('cdna4', name, ['d'], sources)) is None
        assert (
            gen_wide_scaled_narrow(context('cdna4', name, ['d'], [], is_vop3=True))
            is None
        )


def test_gfx1250_scaled_fp6_narrow_bindings():
    from amdisa.codegen.execute.rj_arith_float import (
        WIDE_SCALED_NARROW_GFX1250_NAMES,
        gen_wide_scaled_narrow,
    )

    generated = (
        Path(__file__).resolve().parents[3]
        / 'rocjitsu/src/rocjitsu/isa/arch/amdgpu/generated/cdna5/vop3_exec_cvt.cpp'
    ).read_text()
    assert len(WIDE_SCALED_NARROW_GFX1250_NAMES) == 30
    for name in WIDE_SCALED_NARROW_GFX1250_NAMES:
        sr = '_SR_' in name
        sources = ['a', 'b', 'c'] if sr else ['a', 'b']
        body = gen_wide_scaled_narrow(
            context('cdna5', name, ['d'], sources, is_vop3=True)
        )
        policy = ', 1' if '_PK8_FP4_' in name else ', 2' if '_PK8_' in name else ''
        assert f'execute_wide_scaled_narrow<{str(sr).lower()}, true{policy}>' in body
        assert ('RJ_ARITH_FP16_OVFL' in body) == (
            '_PK8_' in name and '_FP4_' not in name
        )
        assert 'resolved_vgpr_offset(wf,' in body
        assert 'nullptr' in body
        assert 'rj_arith_' + name.lower() + '_w32_gfx1250' in generated
        for arch in ('rdna3', 'rdna4', 'cdna3', 'cdna4'):
            assert (
                gen_wide_scaled_narrow(
                    context(arch, name, ['d'], sources, is_vop3=True)
                )
                is None
            )
        assert gen_wide_scaled_narrow(context('cdna5', name, ['d'], sources)) is None
        assert (
            gen_wide_scaled_narrow(context('cdna5', name, ['d'], [], is_vop3=True))
            is None
        )


@pytest.mark.parametrize(
    'name', ['V_MIN_NUM_F32', 'V_MAX_NUM_F32', 'V_MINIMUM_F32', 'V_MAXIMUM_F32']
)
def test_gfx1250_selection_owns_its_numeric_only_compatibility_policy(name):
    body = gen_float_binary32(context('cdna5', name, ['d'], ['a', 'b'], is_vop3=True))
    for wave in (32, 64):
        assert f'rj_arith_{name.lower()}_w{wave}_gfx1250,' in body
    assert 'BinaryFpOperation' not in body


@pytest.mark.parametrize('arch', ['cdna1', 'cdna2', 'rdna1', 'rdna2', 'rdna3_5'])
@pytest.mark.parametrize('accumulate', [False, True])
def test_older_packed_fma_uses_inherited_policy(arch, accumulate):
    from amdisa.codegen.execute.rj_arith_float import gen_packed_fma16

    cdna = arch.startswith('cdna')
    name = 'V_PK_FMAC_F16' if accumulate else 'V_PK_FMA_F16'
    encodings = (
        ['ENC_VOP2', 'ENC_VOP3']
        if accumulate and cdna
        else ['ENC_VOP2'] if accumulate else ['ENC_VOP3P']
    )
    for enc in encodings:
        body = gen_packed_fma16(
            context(
                arch,
                name,
                ['d'],
                ['a', 'b'] if accumulate else ['a', 'b', 'c'],
                enc_name=enc,
                opsel_exprs=('lo', 'hi'),
                op_sel_hi_2_expr='hi_c',
            )
        )
        suffix = '_gfx942' if accumulate and cdna else ''
        for lanes in (32, 64):
            assert f'rj_arith_{name.lower()}_w{lanes}{suffix}' in body
        assert body.index('if (!wf.exec())') < body.index('UnsupportedOperandValue')
        assert 'pk16_src_needs_narrowing' in body
        assert ('execute_packed_fmac_sdwa' in body) == (
            accumulate and cdna and enc == 'ENC_VOP2'
        )
        assert ('RJ_ARITH_ALU_OMOD_2' in body) == (accumulate and cdna)
        assert 'sdwa::output_modifier<' not in body
        if accumulate and cdna and enc == 'ENC_VOP2':
            assert '(inst_.src0 == amdgpu::SRC_SDWA ? sdwa_omod_ : 0u)' in body


@pytest.mark.parametrize('arch', ['rdna1', 'rdna2', 'rdna3_5'])
@pytest.mark.parametrize(
    'name', ['V_FMA_MIX_F32', 'V_FMA_MIXLO_F16', 'V_FMA_MIXHI_F16']
)
def test_older_radeon_mixed_uses_preserved_half_policy(arch, name):
    from amdisa.codegen.execute.rj_arith_float import gen_mixed_fma

    body = gen_mixed_fma(
        context(
            arch,
            name,
            ['d'],
            ['a', 'b', 'c'],
            enc_name='ENC_VOP3P',
            opsel_exprs=('lo', 'hi'),
            op_sel_hi_2_expr='hi_c',
        )
    )
    for lanes in (32, 64):
        assert f'rj_arith_{name.lower()}_w{lanes}' in body
    assert body.index('if (!wf.exec())') < body.index('UnsupportedOperandValue')
    assert 'true, true);' in body
    assert 'RJ_ARITH_MIX_F16_C' in body
    assert 'is_inline_float_src(inst_.src2)' in body


@pytest.mark.parametrize(
    'name', ['V_MAD_MIX_F32', 'V_MAD_MIXLO_F16', 'V_MAD_MIXHI_F16']
)
def test_gfx908_mixed_fma_routes_independent_rounding_model(name):
    from amdisa.codegen.execute.rj_arith_float import gen_mixed_fma

    body = gen_mixed_fma(
        context(
            'cdna1',
            name,
            ['d'],
            ['a', 'b', 'c'],
            enc_name='ENC_VOP3P',
            opsel_exprs=('inst_.op_sel', 'inst_.op_sel_hi'),
            op_sel_hi_2_expr='inst_.op_sel_hi_2',
        )
    )
    assert f'rj_arith_{name.lower()}_w32_gfx908' in body
    assert f'rj_arith_{name.lower()}_w64_gfx908' in body
    assert 'true, false, true);' in body


def test_mixed_fma_has_no_legacy_dispatch_for_any_registered_profile():
    from amdisa.__main__ import _PROFILES
    from amdisa.codegen.execute import DISPATCH
    from amdisa.codegen.execute.rj_arith_float import (
        MIXED_FMA_NAMES,
        MIXED_BF16_NAMES,
        gen_mixed_fma,
    )

    assert not any(name.startswith('mad_mix') for name in DISPATCH)
    for profile in _PROFILES:
        if profile == 'cdna':
            continue
        arch = profile.replace('.', '_')
        names = MIXED_FMA_NAMES
        if arch in ('cdna1', 'cdna2', 'cdna3', 'cdna4'):
            names = {name.replace('FMA', 'MAD') for name in names}
        elif arch == 'cdna5':
            names = names | MIXED_BF16_NAMES
        for name in names:
            body = gen_mixed_fma(
                context(
                    arch,
                    name,
                    ['d'],
                    ['a', 'b', 'c'],
                    enc_name='ENC_VOP3P',
                    opsel_exprs=('lo', 'hi'),
                    op_sel_hi_2_expr='hi_c',
                )
            )
            assert body and 'arith::execute_mixed_' in body, (arch, name)


@pytest.mark.parametrize('arch', ['cdna1', 'cdna2', 'rdna1', 'rdna2', 'rdna3_5'])
def test_older_native_half_fma_routes_and_qualification(arch):
    from amdisa.codegen.execute.rj_arith_float import gen_float_fma16

    for name, inputs in [
        ('V_FMA_F16', ['a', 'b', 'c']),
        ('V_FMA_LEGACY_F16', ['a', 'b', 'c']),
        ('V_FMAC_F16', ['a', 'b']),
    ]:
        if ('LEGACY' in name) != arch.startswith('cdna') and name != 'V_FMA_F16':
            continue
        body = gen_float_fma16(context(arch, name, ['d'], inputs, is_vop3=True))
        assert 'execute_float_ternary16' in body
        for wave in (32, 64):
            assert f'rj_arith_v_fma_f16_w{wave}' in body
        assert body.index('if (!wf.exec())') < body.index('const uint64_t semantics')
        assert 'RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL_NO_FP_STATUS' in body
        assert 'RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL)' in body
        assert ('const uint32_t opsel = 0;' in body) == (
            arch in ('rdna1', 'rdna2') and name == 'V_FMAC_F16'
        )


@pytest.mark.parametrize('arch', ['cdna1', 'cdna2', 'rdna1', 'rdna2', 'rdna3_5'])
def test_generated_older_native_half_fma_has_no_legacy_bypass(arch):
    root = (
        Path(__file__).resolve().parents[3]
        / 'rocjitsu/src/rocjitsu/isa/arch/amdgpu/generated'
        / arch
    )
    names = {'VFmaF16Vop3'}
    if arch.startswith('cdna'):
        names.add('VFmaLegacyF16Vop3')
        if arch == 'cdna1':
            names |= {'VMacF16Vop2', 'VMacF16Vop3'}
    else:
        names |= {'VFmacF16Vop2', 'VFmacF16Vop3', 'VFmamkF16Vop2', 'VFmaakF16Vop2'}
    found = set()
    for encoding in ('vop2', 'vop3'):
        source = (root / f'{encoding}_exec.cpp').read_text()
        for match in re.finditer(
            r'void (\w+)::execute(?:_modifier)?_impl\([^)]*\)\s*\{', source
        ):
            if match[1] not in names:
                continue
            end, depth = match.end(), 1
            while depth:
                depth += (source[end] == '{') - (source[end] == '}')
                end += 1
            body = source[match.end() : end - 1]
            assert 'arith::execute_float_ternary16' in body
            assert 'fp_mode::fma_f16' not in body
            assert 'ROCJITSU_TRY_SIMD' not in body
            assert 'execute_v_' not in body
            found.add(match[1])
    assert found == names


def test_generated_gfx908_half_mad_has_no_legacy_bypass():
    root = (
        Path(__file__).resolve().parents[3]
        / 'rocjitsu/src/rocjitsu/isa/arch/amdgpu/generated/cdna1'
    )
    names = {'VMadF16Vop3', 'VMadLegacyF16Vop3', 'VMadmkF16Vop2', 'VMadakF16Vop2'}
    found = set()
    for encoding in ('vop2', 'vop3'):
        source = (root / f'{encoding}_exec.cpp').read_text()
        for match in re.finditer(
            r'void (\w+)::execute(?:_modifier)?_impl\([^)]*\)\s*\{', source
        ):
            if match[1] not in names:
                continue
            end, depth = match.end(), 1
            while depth:
                depth += (source[end] == '{') - (source[end] == '}')
                end += 1
            body = source[match.end() : end - 1]
            assert 'arith::execute_float_ternary16' in body
            assert 'rj_arith_v_mad_' in body
            assert 'rj_arith_v_fma_' not in body
            assert 'ROCJITSU_TRY_SIMD' not in body
            found.add(match[1])
    assert found == names


@pytest.mark.parametrize('encoding', ['ENC_VOP2', 'ENC_VOP3'])
def test_cdna2_half_mac_keeps_its_separate_rounding(encoding):
    from amdisa.codegen.execute.rj_arith_float import gen_float_fma16

    body = gen_float_fma16(
        context(
            'cdna2',
            'V_MAC_F16',
            ['d'],
            ['a', 'b'],
            enc_name=encoding,
            is_vop3=encoding == 'ENC_VOP3',
        )
    )
    assert 'rj_arith_v_mad_f16_w64_gfx90a' in body
    assert 'd, a, b, d, true, true' in body


def test_cdna2_half_mad_has_no_retired_evaluator():
    generated = (
        Path(__file__).resolve().parents[3] / 'rocjitsu/src/rocjitsu/isa/arch/amdgpu'
    )
    # The production generated bindings, not just an emitter result, use the new API.
    for encoding in ('vop2', 'vop3'):
        text = (generated / f'generated/cdna2/{encoding}_exec.cpp').read_text()
        assert 'rj_arith_v_mad_f16_w64_gfx90a' in text
        assert 'execute_cdna2_mad_f16' not in text
        assert 'MadVop2::Accumulate, true' not in text
    assert 'cdna2_mad_f16' not in (generated / 'shared/simd_glue.h').read_text()


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
        assert 'execute_cdna_float_unary(*this, wf' in body
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
    assert ('execute_cdna_float_unary' in body) == (not vop3)
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


def test_cdna_half_dispatch_does_not_mutate_shared_names():
    from amdisa.codegen.execute.rj_arith_float import gen_float_unary16

    original = frozenset(CDNA_MATH_F16)
    assert gen_float_unary16(context('cdna3', 'V_RCP_F16', ['d'], ['a'], is_vop3=True))
    assert frozenset(CDNA_MATH_F16) == original


@pytest.mark.parametrize(
    'name',
    [
        'V_ADD_F32',
        'V_SUB_F32',
        'V_SUBREV_F32',
        'V_MUL_F32',
        'V_FMA_F32',
        'V_FMAC_F32',
        'V_FMAMK_F32',
        'V_FMAAK_F32',
    ],
)
def test_gfx1150_native_fp32_method_routes(name):
    root = (
        Path(__file__).resolve().parents[3]
        / 'rocjitsu/src/rocjitsu/isa/arch/amdgpu/generated/rdna3_5'
    )
    fused = name.startswith(('V_FMA', 'V_FMAC'))
    literal = name in ('V_FMAMK_F32', 'V_FMAAK_F32')
    encodings = (
        ('vop2',) if literal else ('vop3',) if name == 'V_FMA_F32' else ('vop2', 'vop3')
    )
    for encoding in encodings:
        source = (root / (encoding + '_exec.cpp')).read_text()
        cls = ''.join(word.title() for word in (name + '_' + encoding).split('_'))
        methods = re.findall(
            r'void ' + cls + r'::execute(?:_modifier)?_impl\([^)]*\)\s*\{(.*?)\n}',
            source,
            re.S,
        )
        assert len(methods) == (1 if literal else 2)
        for body in methods:
            api = 'v_fma_f32' if fused else name.lower()
            suffix = (
                '_gfx1150'
                if name
                in ('V_ADD_F32', 'V_SUB_F32', 'V_SUBREV_F32', 'V_MUL_F32', 'V_FMA_F32')
                else ''
            )
            for width in (32, 64):
                assert f'rj_arith_{api}_w{width}{suffix},' in body
            assert 'wf.cu().arithmetic_flags()' in body
            assert 'amdgpu::execute_v_' not in body
            assert 'try_execute_' not in body
            assert 'arithmetic_flags<' not in body
            if name == 'V_MUL_F32':
                assert 'compatibility_mul=*/false' in body


@pytest.mark.parametrize(
    'name', ['V_ADD_F32', 'V_SUB_F32', 'V_SUBREV_F32', 'V_MUL_F32']
)
def test_gfx908_binary32_methods_forward_policy_and_scaling(name):
    root = (
        Path(__file__).resolve().parents[3]
        / 'rocjitsu/src/rocjitsu/isa/arch/amdgpu/generated/cdna1'
    )
    for encoding in ('vop2', 'vop3'):
        source = (root / (encoding + '_exec.cpp')).read_text()
        cls = ''.join(word.title() for word in (name + '_' + encoding).split('_'))
        methods = re.findall(
            r'void ' + cls + r'::execute(?:_modifier)?_impl\([^)]*\)\s*\{(.*?)\n}',
            source,
            re.S,
        )
        assert len(methods) == (2 if encoding == 'vop2' else 1)
        for body in methods:
            suffix = '_gfx908' if name in ('V_ADD_F32', 'V_MUL_F32') else ''
            for wave in (32, 64):
                assert f'rj_arith_{name.lower()}_w{wave}{suffix},' in body
            assert 'wf.cu().arithmetic_flags()' in body
            assert 'compatibility_mul=*/false' in body
            assert 'classify_binary_f32' not in body
            assert 'amdgpu::execute_v_' not in body
            assert 'try_execute_' not in body


@pytest.mark.parametrize('arch', ['cdna2', 'rdna1', 'rdna2'])
def test_older_binary_fp32_routes(arch):
    root = (
        Path(__file__).resolve().parents[3]
        / 'rocjitsu/src/rocjitsu/isa/arch/amdgpu/generated'
        / arch
    )
    expected = {
        op + encoding
        for op in ('VAddF32', 'VSubF32', 'VSubrevF32', 'VMulF32')
        for encoding in ('Vop2', 'Vop3')
    }
    found = set()
    for encoding in ('vop2', 'vop3'):
        text = (root / (encoding + '_exec.cpp')).read_text()
        for match in re.finditer(
            r'void (\w+)::execute(?:_modifier)?_impl\([^)]*\)\s*\{', text
        ):
            if match[1] not in expected:
                continue
            end, depth = match.end(), 1
            while depth:
                depth += (text[end] == '{') - (text[end] == '}')
                end += 1
            body = ' '.join(text[match.end() : end - 1].split())
            found.add(match[1])
            op = re.sub(r'Vop[23]$', '', match[1])[1:-3].lower()
            for wave in (32, 64):
                assert f'rj_arith_v_{op}_f32_w{wave},' in body
            assert 'arithmetic_flags()' in body
            assert 'amdgpu::execute_v_' not in body
            assert '/*rdna3=*/true' in body
            assert ('/*compatibility_mul=*/true' in body) == (op == 'mul')
            assert ('execute_sdwa_binary32(*this, wf' in body) == (encoding == 'vop2')
            if encoding == 'vop3':
                for operand in 'AB':
                    assert 'RJ_ARITH_ALU_ABS_' + operand in body
                    assert 'RJ_ARITH_ALU_NEG_' + operand in body
                assert 'RJ_ARITH_ALU_CLAMP' in body
                assert 'RJ_ARITH_ALU_OMOD_2' in body
    assert found == expected
    assert (
        gen_float_binary32(
            context(arch, 'V_LDEXP_F32', ['d'], ['a', 'exponent'], is_vop3=True)
        )
        is None
    )


def test_rdna2_ordinary_fma_uses_arithmetic_library():
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
            is not None
        )


@pytest.mark.parametrize('arch,count', [('cdna1', 6), ('cdna2', 6), ('rdna1', 8)])
def test_older_mad_bindings_keep_each_form_and_wave(arch, count):
    root = (
        Path(__file__).resolve().parents[3]
        / 'rocjitsu/src/rocjitsu/isa/arch/amdgpu/generated'
        / arch
    )
    checked = 0
    for encoding in ('vop2', 'vop3'):
        source = (root / (encoding + '_exec.cpp')).read_text()
        for m in re.finditer(
            r'void (V(?:Mac(?:Legacy)?|Mad(?:Legacy|mk|ak)?)F32Vop[23])::execute_impl\([^)]*\)\s*\{',
            source,
        ):
            end, depth = m.end(), 1
            while depth:
                depth += (source[end] == '{') - (source[end] == '}')
                end += 1
            body = source[m.end() : end - 1]
            body = re.sub(r'\s+', ' ', body)
            api = 'rj_arith_v_mad' + ('_legacy' if 'Legacy' in m[1] else '') + '_f32'
            for wave in (32, 64):
                assert f'{api}_w{wave},' in body, m[1]
            assert 'execute_float_fma32<false>(' in body
            assert 'try_execute_' not in body
            if 'Madmk' in m[1]:
                assert 'vdst, src0, simm32, vsrc1' in body
            elif 'Madak' in m[1]:
                assert 'vdst, src0, vsrc1, simm32' in body
            elif 'Mac' in m[1]:
                assert ('src1, vdst' if encoding == 'vop3' else 'vsrc1, vdst') in body
                assert 'RJ_ARITH_ALU_NEG_C' not in body
            checked += 1
    assert checked == count


@pytest.mark.parametrize('arch', ['cdna1', 'cdna2', 'rdna1', 'rdna2'])
def test_older_native_fma_bindings(arch):
    root = (
        Path(__file__).resolve().parents[3]
        / 'rocjitsu/src/rocjitsu/isa/arch/amdgpu/generated'
        / arch
    )
    expected = {'VFmaF32Vop3', 'VFmacF32Vop2', 'VFmacF32Vop3'}
    if arch.startswith('rdna'):
        expected |= {'VFmamkF32Vop2', 'VFmaakF32Vop2'}
    found = set()
    for encoding in ('vop2', 'vop3'):
        source = (root / (encoding + '_exec.cpp')).read_text()
        for match in re.finditer(
            r'void (\w+)::execute(?:_modifier)?_impl\([^)]*\)\s*\{', source
        ):
            if match[1] not in expected:
                continue
            found.add(match[1])
            end, depth = match.end(), 1
            while depth:
                depth += (source[end] == '{') - (source[end] == '}')
                end += 1
            body = ' '.join(source[match.end() : end - 1].split())
            for wave in (32, 64):
                assert f'rj_arith_v_fma_f32_w{wave},' in body
            assert 'arith::execute_' in body
            assert 'amdgpu::execute_v_' not in body
            assert 'ScopedEnvironment' not in body
            if 'Fmac' in match[1]:
                assert 'RJ_ARITH_ALU_NEG_C' not in body
                assert (
                    'vdst, src0, '
                    + ('vsrc1' if encoding == 'vop2' else 'src1')
                    + ', vdst,'
                    in body
                )
            if match[1] == 'VFmaF32Vop3':
                for operand in 'ABC':
                    assert 'RJ_ARITH_ALU_NEG_' + operand in body
                    assert 'RJ_ARITH_ALU_ABS_' + operand in body
                assert 'RJ_ARITH_ALU_OMOD_2' in body
                assert 'RJ_ARITH_ALU_CLAMP' in body
            if 'Fmamk' in match[1]:
                assert 'vdst, src0, simm32, vsrc1,' in body
            if 'Fmaak' in match[1]:
                assert 'vdst, src0, vsrc1, simm32,' in body
    assert found == expected
    # The combined migrations also enable ordinary binary arithmetic.
    for name in ('V_ADD_F32', 'V_MUL_F32'):
        assert (
            gen_float_binary32(context(arch, name, ['d'], ['a', 'b'], is_vop3=True))
            is not None
        )


@pytest.mark.parametrize('arch', ['cdna1', 'cdna2', 'rdna1', 'rdna2', 'rdna3_5'])
def test_older_legacy_multiply_bindings(arch):
    name = 'V_MUL_DX9_ZERO_F32' if arch == 'rdna3_5' else 'V_MUL_LEGACY_F32'
    for vop3 in ([True] if arch == 'cdna2' else [False, True]):
        body = gen_float_binary32(
            context(arch, name, ['dst'], ['a', 'b'], is_vop3=vop3)
        )
        assert 'execute_legacy_multiply32' in body
        for wave in (32, 64):
            assert f'rj_arith_v_mul_legacy_f32_w{wave},' in body


@pytest.mark.parametrize('arch', ['cdna1', 'cdna2', 'rdna1', 'rdna2', 'rdna3_5'])
def test_older_ldexp32_binding(arch):
    from amdisa.codegen.execute.rj_arith_float import gen_older_ldexp32

    body = gen_older_ldexp32(
        context(arch, 'V_LDEXP_F32', ['dst'], ['a', 'b'], is_vop3=True)
    )
    for wave in (32, 64):
        assert f'rj_arith_v_ldexp_f32_w{wave}_gfx942,' in body
    assert 'RJ_ARITH_ALU_ABS_B' not in body
    assert 'RJ_ARITH_ALU_NEG_B' not in body
    assert '/*rdna3=*/true' in body
    assert gen_older_ldexp32(context(arch, 'V_LDEXP_F32', ['dst'], ['a', 'b'])) is None


@pytest.mark.parametrize('arch', ['cdna1', 'cdna2'])
def test_gfx9_native_fp64_routes_each_encoding(arch):
    from amdisa.codegen.execute.rj_arith_float import (
        gen_float_binary64,
        gen_float64_word,
    )

    root = (
        Path(__file__).resolve().parents[3]
        / 'rocjitsu/src/rocjitsu/isa/arch/amdgpu/generated'
        / arch
    )
    expected = {'VAddF64', 'VMulF64', 'VFmaF64', 'VLdexpF64'}
    if arch == 'cdna2':
        expected.add('VFmacF64')
    found = set()
    for source in root.glob('vop*_exec.cpp'):
        text = source.read_text()
        for match in re.finditer(
            r'void (\w+)::execute(?:_modifier)?_impl\([^)]*\)\s*\{', text
        ):
            operation = re.sub(r'Vop[23]$', '', match[1])
            if operation not in expected:
                continue
            end, depth = match.end(), 1
            while depth:
                depth += (text[end] == '{') - (text[end] == '}')
                end += 1
            body = text[match.end() : end - 1]
            found.add(operation)
            assert 'amdgpu::execute_v_' not in body, match[1]
            if operation == 'VFmacF64':
                assert 'execute_fmac64<true>' in body
                assert 'vdst, src0,' in body
                assert 'RJ_ARITH_ALU_NEG_C' not in body
            else:
                for wave in (32, 64):
                    suffix = '_gfx942' if operation == 'VLdexpF64' else '_gfx950'
                    api = (
                        'rj_arith_v_'
                        + operation[1:-3].lower()
                        + f'_f64_w{wave}{suffix},'
                    )
                    assert api in body, (match[1], api)
    assert found == expected
    for name in ('V_ADD_F32', 'V_MUL_F32'):
        assert (
            gen_float_binary32(context(arch, name, ['d'], ['a', 'b'], is_vop3=True))
            is not None
        )
    for name in ('V_ADD_F64', 'V_MUL_F64'):
        assert gen_float_binary64(context(arch, name, ['d'], ['a', 'b'], is_vop3=True))
    body = gen_float64_word(
        context(arch, 'V_LDEXP_F64', ['d'], ['a', 'exponent'], is_vop3=True)
    )
    assert 'RJ_ARITH_ALU_ABS_B' not in body
    assert 'RJ_ARITH_ALU_NEG_B' not in body
    assert '/*ldexp=*/true' in body


@pytest.mark.parametrize('arch', ['rdna1', 'rdna2', 'rdna3_5'])
def test_older_radeon_fp64_routes_and_policy(arch):
    from amdisa.codegen.execute.rj_arith_float import gen_float64_word

    root = (
        Path(__file__).resolve().parents[3]
        / 'rocjitsu/src/rocjitsu/isa/arch/amdgpu/generated'
        / arch
    )
    source = (root / 'vop3_exec.cpp').read_text()
    expected = {
        'VAddF64': 'add',
        'VMulF64': 'mul',
        'VFmaF64': 'fma',
        'VLdexpF64': 'ldexp',
    }
    found = set()
    for match in re.finditer(
        r'void (\w+)Vop3::execute(?:_modifier)?_impl\([^)]*\)\s*\{', source
    ):
        if match[1] not in expected:
            continue
        found.add(match[1])
        end, depth = match.end(), 1
        while depth:
            depth += (source[end] == '{') - (source[end] == '}')
            end += 1
        body = source[match.end() : end - 1]
        suffix = (
            '_gfx942'
            if match[1] == 'VLdexpF64'
            else '_gfx10' if arch == 'rdna2' and match[1] == 'VFmaF64' else '_gfx950'
        )
        for wave in (32, 64):
            assert f'rj_arith_v_{expected[match[1]]}_f64_w{wave}{suffix},' in body
        assert 'arithmetic_flags()' in body
        assert 'amdgpu::execute_v_' not in body
    assert found == expected.keys()
    for name in ('V_ADD_F32', 'V_MUL_F32'):
        assert (
            gen_float_binary32(context(arch, name, ['d'], ['a', 'b'], is_vop3=True))
            is not None
        )
    for name in ('V_LDEXP_F64', 'V_TRIG_PREOP_F64'):
        body = gen_float64_word(
            context(arch, name, ['d'], ['a', 'exponent'], is_vop3=True)
        )
        assert 'RJ_ARITH_ALU_ABS_B' not in body
        assert 'RJ_ARITH_ALU_NEG_B' not in body
        assert ('/*ldexp=*/true' in body) == (name == 'V_LDEXP_F64')
        if name == 'V_TRIG_PREOP_F64':
            assert '/*preop_rounding=*/true' in body
            assert 'rj_arith_v_trig_preop_f64_w64_gfx942,' in body


@pytest.mark.parametrize('arch', ['cdna1', 'cdna2', 'rdna1', 'rdna2', 'rdna3_5'])
def test_older_selection_routes_each_method_and_neighboring_arithmetic(arch):
    from amdisa.codegen.execute.rj_arith_float import (
        OLDER_SELECTION_NAMES,
        gen_float_binary64,
        gen_float_ternary_selection,
    )

    root = (
        Path(__file__).resolve().parents[3]
        / 'rocjitsu/src/rocjitsu/isa/arch/amdgpu/generated'
        / arch
    )
    suffix = '_gfx942' if arch.startswith('cdna') else ''
    found = set()
    for source in root.glob('vop*_exec.cpp'):
        for match in re.finditer(
            r'void (\w+)::execute(?:_modifier)?_impl\([^)]*\)\s*\{', source.read_text()
        ):
            text = source.read_text()
            end, depth = match.end(), 1
            while depth:
                depth += (text[end] == '{') - (text[end] == '}')
                end += 1
            body = text[match.end() : end - 1]
            for name in OLDER_SELECTION_NAMES:
                instruction = ''.join(
                    part.capitalize() for part in name.lower().split('_')
                )
                if not match[1].startswith(instruction + 'Vop'):
                    continue
                found.add(name)
                for wave in (32, 64):
                    assert f'rj_arith_{name.lower()}_w{wave}{suffix},' in body, match[1]
                assert 'amdgpu::execute_v_' not in body, match[1]
                assert 'arithmetic_flags()' in body
    assert found == OLDER_SELECTION_NAMES
    for name in ('V_ADD_F32', 'V_MUL_F32', 'V_ADD_F64', 'V_MUL_F64'):
        emitter = gen_float_binary64 if name.endswith('F64') else gen_float_binary32
        body = emitter(context(arch, name, ['d'], ['a', 'b'], is_vop3=True))
        assert 'rj_arith_' + name.lower() in body
    for name in ('V_MIN3_F32', 'V_MAX3_F32', 'V_MED3_F32'):
        body = gen_float_ternary_selection(
            context(arch, name, ['d'], ['a', 'b', 'c'], is_vop3=True)
        )
        assert 'RJ_ARITH_ALU_NEG_C' in body


@pytest.mark.parametrize('arch', ['cdna1', 'cdna2', 'rdna1', 'rdna2', 'rdna3_5'])
def test_older_half_selection_model_and_production_routes(arch):
    from amdisa.codegen.execute.rj_arith_float import (
        gen_float_binary16,
        gen_float_ternary_selection,
    )

    root = (
        Path(__file__).resolve().parents[3]
        / 'rocjitsu/src/rocjitsu/isa/arch/amdgpu/generated'
        / arch
    )
    for name in (
        'MIN',
        'MAX',
        'MIN3',
        'MAX3',
        'MED3',
        *(['MINMAX', 'MAXMIN'] if arch == 'rdna3_5' else []),
    ):
        binary = name in ('MIN', 'MAX')
        mnemonic = f'V_{name}_F16'
        emitter = gen_float_binary16 if binary else gen_float_ternary_selection
        for encoding in (('vop2', 'vop3') if binary else ('vop3',)):
            body = emitter(
                context(
                    arch,
                    mnemonic,
                    ['d'],
                    ['a', 'b'] if binary else ['a', 'b', 'c'],
                    is_vop3=encoding == 'vop3',
                )
            )
            api = f'rj_arith_v_{name.lower()}_f16_w64' + (
                '_gfx942' if arch.startswith('cdna') else ''
            )
            assert api in body
            assert 'SEMANTICS_BIT_EXACT_EMPIRICAL_NO_FP_STATUS' in body
            assert 'UnsupportedOperandValue' in body
            source = (root / f'{encoding}_exec.cpp').read_text()
            class_name = 'V' + name.title() + 'F16' + encoding.title()
            production = source.split(f'void {class_name}::execute_impl', 1)[1].split(
                'void ', 1
            )[0]
            assert api in production
            assert 'ROCJITSU_TRY_SIMD_' not in production
            if binary and arch in ('rdna1', 'rdna2') and encoding == 'vop3':
                assert 'const uint32_t opsel = 0;' in body
    # Shared evaluators lose their final production callers on this migration.
    shared = (root.parent / 'shared/execute_shared.h').read_text()
    for name in ('min', 'max', 'min3', 'max3', 'med3'):
        assert f'execute_v_{name}_f16_' not in shared


@pytest.mark.parametrize('arch', ['cdna1', 'cdna2', 'rdna2', 'cdna3', 'cdna4'])
def test_host_model_float_dot_bindings(arch):
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
                if arch != 'cdna4' and fmt == 'BF16':
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


@pytest.mark.parametrize('arch', ['cdna1', 'cdna2', 'rdna2'])
def test_older_float_dot_generated_routes(arch):
    root = (
        Path(__file__).resolve().parents[3]
        / 'rocjitsu/src/rocjitsu/isa/arch/amdgpu/generated'
        / arch
    )
    checked = 0
    for source in root.glob('*_exec.cpp'):
        text = source.read_text()
        for match in re.finditer(
            r'void (VDot2(?:c)?F32F16\w+)::execute(?:_modifier)?_impl\([^)]*\)\s*\{',
            text,
        ):
            end, depth = match.end(), 1
            while depth:
                depth += (text[end] == '{') - (text[end] == '}')
                end += 1
            body = text[match.end() : end - 1]
            assert 'rj_arith_v_dot2_f32_f16_w64_gfx942' in body, match[1]
            assert 'arith::execute_dot2_f32' in body, match[1]
            assert 'ROCJITSU_TRY_SIMD' not in body, match[1]
            assert 'amdgpu::execute_v_dot' not in body, match[1]
            assert 'shared/rj_arith_float_exec.h' in text, source
            checked += 1
    assert checked == (3 if arch == 'rdna2' else 4)


def test_gfx1150_bf16_inline_uses_upper_float_half():
    from amdisa.codegen.execute.rj_arith_float import gen_float_dot32

    body = gen_float_dot32(
        context(
            'rdna3_5',
            'V_DOT2_F32_BF16',
            ['d'],
            ['a', 'b', 'c'],
            enc_name='ENC_VOP3P',
            opsel_exprs=('low', 'high'),
        )
    )
    assert 'rj_arith_v_dot2_f32_bf16_w32,' in body
    assert 'dot2_src_needs_half_replication(inst_.src0)' in body
    assert 'is_inline_float_src' not in body
    assert 'RJ_ARITH_DOT_ABS_C' in body and 'RJ_ARITH_DOT_NEG_C' in body


@pytest.mark.parametrize(
    'name,half,accumulate',
    [
        ('V_DOT2_F32_F16', False, False),
        ('V_DOT2_F32_BF16', False, False),
        ('V_DOT2_F16_F16', True, False),
        ('V_DOT2_BF16_BF16', True, False),
        ('V_DOT2ACC_F32_F16', False, True),
    ],
)
def test_gfx1150_float_dot_rejects_unqualified_strict_replay(name, half, accumulate):
    from amdisa.codegen.execute.rj_arith_float import gen_float_dot32, gen_half_dot

    body = (gen_half_dot if half else gen_float_dot32)(
        context(
            'rdna3_5',
            name,
            ['d'],
            ['a', 'b'] if accumulate else ['a', 'b', 'c'],
            is_vop3=half,
            enc_name='ENC_VOP3' if half else 'ENC_VOP2' if accumulate else 'ENC_VOP3P',
            opsel_exprs=('low', 'high'),
        )
    )
    assert (
        body.index('if (!wf.exec())')
        < body.index('UnsupportedOperandValue')
        < body.index('execute_dot2_')
    )
    assert 'RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL_NO_FP_STATUS' in body


@pytest.mark.parametrize(
    'arch', ['cdna1', 'cdna2', 'cdna3', 'cdna4', 'rdna1', 'rdna2', 'rdna3_5']
)
@pytest.mark.parametrize('op', ['TRUNC', 'CEIL', 'RNDNE', 'FLOOR'])
def test_older_half_rounding_keeps_policy_and_layout(arch, op):
    from amdisa.codegen.execute.rj_arith_float import gen_float_unary16

    body = gen_float_unary16(context(arch, f'V_{op}_F16', ['d'], ['a'], is_vop3=True))
    if arch in ('cdna3', 'cdna4'):
        assert body is None  # The earlier CDNA integral binding owns this form.
        return
    assert f'rj_arith_v_{op.lower()}_f16_w32_gfx908' in body
    assert ('RJ_ARITH_ALU_HIGH_D' in body) == (arch not in ('rdna1', 'rdna2'))
    assert body.index('if (!wf.exec())') < body.index('arithmetic_flags()')
    assert 'RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL_NO_FP_STATUS' in body
    assert 'execute_float_unary16' in body


@pytest.mark.parametrize(
    'arch', ['cdna1', 'cdna2', 'rdna1', 'rdna2', 'rdna3_5', 'cdna3', 'cdna4']
)
@pytest.mark.parametrize('name', ['V_FREXP_MANT_F16', 'V_FREXP_EXP_I16_F16'])
def test_older_half_frexp_preserves_operand_and_output_policy(arch, name):
    from amdisa.codegen.execute.rj_arith_float import gen_float_unary16

    body = gen_float_unary16(context(arch, name, ['d'], ['a'], is_vop3=True))
    if arch in ('cdna3', 'cdna4') and name == 'V_FREXP_MANT_F16':
        assert body is None  # The earlier CDNA integral binding owns mantissas.
        return
    assert 'execute_float_unary16' in body
    assert ('RJ_ARITH_ALU_HIGH_A' in body) == (arch not in ('rdna1', 'rdna2'))
    assert ('_gfx908' in body) == (name == 'V_FREXP_MANT_F16')
    assert ('inst_.clamp' in body) == (name == 'V_FREXP_MANT_F16')
    body = gen_float_unary16(context(arch, name, ['d'], ['a'], is_vop3=False))
    expected = (
        'execute_half_exponent_sdwa'
        if name == 'V_FREXP_EXP_I16_F16'
        else 'execute_half_unary_sdwa'
    )
    assert expected in body


@pytest.mark.parametrize(
    'arch', ['cdna1', 'cdna2', 'rdna1', 'rdna2', 'rdna3_5', 'cdna3', 'cdna4']
)
def test_older_half_fract_uses_rounded_result_adapter(arch):
    from amdisa.codegen.execute.rj_arith_float import gen_float_unary16

    for vop3 in (False, True):
        body = gen_float_unary16(
            context(arch, 'V_FRACT_F16', ['d'], ['a'], is_vop3=vop3)
        )
        if arch in ('cdna3', 'cdna4'):
            assert body is None  # The earlier CDNA integral binding owns this form.
            continue
        assert 'rj_arith_v_fract_f16_w32_gfx908' in body
        assert 'RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL_NO_FP_STATUS' in body
        assert ('execute_half_unary_sdwa' in body) == (not vop3)


@pytest.mark.parametrize('arch', ['cdna3', 'cdna4', 'cdna5'])
def test_cdna_half_unary_uses_the_decoded_route(arch):
    from amdisa.codegen.execute.rj_arith_float import gen_float_unary16

    root = (
        Path(__file__).resolve().parents[3]
        / 'rocjitsu/src/rocjitsu/isa/arch/amdgpu/generated'
        / arch
    )
    for name in ('FLOOR', 'FREXP_MANT', 'FRACT'):
        assert gen_float_unary16(context(arch, f'V_{name}_F16', ['d'], ['a'])) is None
        api = f'rj_arith_v_{name.lower()}_f16_w' + (
            '32_gfx12' if arch == 'cdna5' else '64_gfx942'
        )
        class_name = 'V' + ''.join(word.title() for word in name.split('_')) + 'F16'
        for encoding in ('vop1', 'vop3'):
            text = '\n'.join(p.read_text() for p in root.glob(f'{encoding}_exec*.cpp'))
            start = text.index(
                'void ' + class_name + encoding.title() + '::execute_impl('
            )
            end = text.index('\n}', start)
            assert api in text[start:end]


@pytest.mark.parametrize('arch', ['cdna1', 'cdna2', 'rdna1', 'rdna2', 'rdna3_5'])
def test_older_bit_packing_reuses_guest_policy(arch):
    from amdisa.codegen.execute.rj_arith_float import gen_bit_pack

    half = gen_bit_pack(
        context(arch, 'V_PACK_B32_F16', ['dst'], ['a', 'b'], is_vop3=True)
    )
    assert 'rj_arith_v_pack_b32_f16_w64' in half
    assert 'RJ_ARITH_FP_IEEE_MODE' in half
    assert 'wf.fp_denorm_mode_f16_f64()' in half
    assert 'RJ_ARITH_ALU_HIGH_B' in half
    byte = gen_bit_pack(
        context(arch, 'V_CVT_PK_U8_F32', ['dst'], ['a', 'b', 'c'], is_vop3=True)
    )
    assert 'wf.fp_round_mode_f32()' in byte
    assert 'RJ_ARITH_ALU_ABS_A' in byte
    if arch.startswith('cdna'):
        accum = gen_bit_pack(
            context(arch, 'V_CVT_PKACCUM_U8_F32', ['dst'], ['a', 'b'], is_vop3=True)
        )
        assert 'rj_arith_v_cvt_pkaccum_u8_f32_w64' in accum
        assert 'dst, a, b, dst' in accum
        assert 'fp_round_mode' not in accum


@pytest.mark.parametrize('arch', ['cdna1', 'cdna2', 'rdna1', 'rdna2', 'rdna3_5'])
def test_older_normalized_layout_and_sdwa_policy(arch):
    from amdisa.codegen.execute.rj_arith_float import gen_normalized

    cdna = arch in ('cdna1', 'cdna2')
    legacy_e32 = cdna or arch in ('rdna1', 'rdna2')
    for vop3 in (False, True):
        body = gen_normalized(
            context(arch, 'V_CVT_NORM_I16_F16', ['dst'], ['a'], is_vop3=vop3)
        )
        assert 'RJ_ARITH_FP_FLUSH_INPUT_DENORMALS' in body
        assert ('execute_normalized_sdwa' in body) == (legacy_e32 and not vop3)
        assert ('auto word_source' in body) == (not legacy_e32 and not vop3)
        assert_routes_to(body, 'rj_arith_v_cvt_norm_i16_f16', CONVERSION_SUFFIX[arch])
        assert '_gfx942' not in body
        for half in (False, True):
            name = 'V_CVT_PKNORM_U16_' + ('F16' if half else 'F32')
            packed = gen_normalized(
                context(arch, name, ['dst'], ['a', 'b'], is_vop3=vop3)
            )
            assert 'rj_arith_v_cvt_pk_norm_u16_' in packed
            assert ('execute_float_pack' in packed) == half


@pytest.mark.parametrize('arch', ['cdna1', 'cdna2', 'rdna1', 'rdna2', 'rdna3_5'])
@pytest.mark.parametrize(
    'name', ['V_PK_ADD_F16', 'V_PK_MUL_F16', 'V_PK_MIN_F16', 'V_PK_MAX_F16']
)
def test_older_packed_half_routes_inherited_model(arch, name):
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
    arithmetic = name in ('V_PK_ADD_F16', 'V_PK_MUL_F16')
    suffix = '_gfx942' if arithmetic or arch.startswith('cdna') else ''
    for lanes in (32, 64):
        assert f'rj_arith_{name.lower()}_w{lanes}{suffix}' in body
    assert 'd, a, b, true, narrow_sources' in body
    assert body.index('if (!wf.exec())') < body.index('UnsupportedOperandValue')
    assert 'UnsupportedOperandValue' in body
    assert 'RJ_ARITH_PK_HI_A_LOW' in body
    assert 'pk16_src_needs_narrowing(inst_.src1, b.size_bits())' in body


@pytest.mark.parametrize('arch', ['cdna1', 'cdna2'])
def test_older_cdna_half_binary_handlers_have_no_legacy_route(arch):
    from amdisa.codegen.execute.rj_arith_float import BINARY_ARITHMETIC_F16

    root = (
        Path(__file__).resolve().parents[3]
        / 'rocjitsu/src/rocjitsu/isa/arch/amdgpu/generated'
        / arch
    )
    for encoding in ('vop2', 'vop3'):
        source = (root / (encoding + '_exec.cpp')).read_text()
        assert 'shared/rj_arith_float_exec.h"' in source
        for mnemonic in BINARY_ARITHMETIC_F16:
            suffix = 'gfx942' if mnemonic == 'V_LDEXP_F16' else 'gfx908'
            assert f'rj_arith_{mnemonic.lower()}_w64_{suffix}' in source
            assert f'amdgpu::execute_{mnemonic.lower()}_{encoding}(' not in source
        if encoding == 'vop2':
            assert 'execute_float_binary16_sdwa' in source


def test_gfx1150_half_arithmetic_preserves_inherited_policy_and_halves():
    from amdisa.codegen.execute.rj_arith_float import (
        gen_float_binary16,
        BINARY_ARITHMETIC_F16,
    )

    for name in sorted(BINARY_ARITHMETIC_F16):
        for vop3 in (False, True):
            body = gen_float_binary16(
                context('rdna3_5', name, ['d'], ['a', 'b'], is_vop3=vop3)
            )
            suffix = 'gfx942' if name == 'V_LDEXP_F16' else 'gfx908'
            assert f'rj_arith_{name.lower()}_w32_{suffix}' in body
            assert f'rj_arith_{name.lower()}_w64_{suffix}' in body
            assert 'd, a, b, true);' in body
            assert 'execute_float_binary16_sdwa' not in body
            assert ('RJ_ARITH_ALU_HIGH_D' in body) == vop3
    root = (
        Path(__file__).resolve().parents[3]
        / 'rocjitsu/src/rocjitsu/isa/arch/amdgpu/generated/rdna3_5'
    )
    checked = 0
    for source in root.glob('vop[23]_exec.cpp'):
        text = source.read_text()
        for match in re.finditer(
            r'void (V(?:Add|Sub|Subrev|Mul|Ldexp)F16Vop[23])::execute(?:_modifier)?_impl\([^)]*\)\s*\{',
            text,
        ):
            end, depth = match.end(), 1
            while depth:
                depth += (text[end] == '{') - (text[end] == '}')
                end += 1
            body = text[match.end() : end - 1]
            assert 'arith::execute_float_binary16' in body, match[1]
            assert (
                'ROCJITSU_TRY_SIMD' not in body and 'amdgpu::execute_v_' not in body
            ), match[1]
            assert 'shared/rj_arith_float_exec.h' in text, source
            checked += 1
    assert checked == 20


@pytest.mark.parametrize('arch,suffix', [('rdna1', 'gfx1010'), ('rdna2', 'gfx1030')])
@pytest.mark.parametrize('vop3', [False, True])
def test_gfx10_half_arithmetic_policy_and_sdwa(arch, suffix, vop3):
    from amdisa.codegen.execute.rj_arith_float import (
        BINARY_ARITHMETIC_F16,
        gen_float_binary16,
    )

    for name in sorted(BINARY_ARITHMETIC_F16):
        body = gen_float_binary16(context(arch, name, ['d'], ['a', 'b'], is_vop3=vop3))
        model = 'gfx942' if name == 'V_LDEXP_F16' else suffix
        for wave in (32, 64):
            assert f'rj_arith_{name.lower()}_w{wave}_{model}' in body
        assert ('execute_float_binary16_sdwa' in body) == (not vop3)
        assert 'execute_sdwa_float_binary16' not in body
        assert 'HalfBinaryOperation' not in body
        assert 'd, a, b, true);' in body
        if vop3:
            assert 'const uint32_t opsel = 0;' in body
            assert 'vop3_opsel(inst_)' not in body


@pytest.mark.parametrize('arch', ['rdna1', 'rdna2'])
def test_generated_gfx10_half_has_no_legacy_arithmetic_or_status(arch):
    root = (
        Path(__file__).resolve().parents[3]
        / 'rocjitsu/src/rocjitsu/isa/arch/amdgpu/generated'
        / arch
    )
    checked = 0
    for encoding in ('vop2', 'vop3'):
        source = (root / f'{encoding}_exec.cpp').read_text()
        assert (
            '#include "rocjitsu/isa/arch/amdgpu/shared/rj_arith_float_exec.h"' in source
        )
        for match in re.finditer(
            r'void (V(?:Add|Sub|Subrev|Mul|Ldexp)F16Vop[23])::execute(?:_modifier)?_impl\([^)]*\)\s*\{',
            source,
        ):
            end, depth = match.end(), 1
            while depth:
                depth += (source[end] == '{') - (source[end] == '}')
                end += 1
            body = source[match.end() : end - 1]
            assert 'arith::execute_float_binary16' in body, match[1]
            assert 'record_gfx10_add' not in body
            assert 'RJ_SIMD' not in body
            assert 'HalfBinaryOperation' not in body
            checked += 1
    assert checked == 15


@pytest.mark.parametrize('arch', ['cdna1', 'cdna2', 'rdna1', 'rdna2', 'rdna3_5'])
def test_older_packed_rtz_routes_existing_kernels_and_sdwa(arch):
    from amdisa.codegen.execute.rj_arith_float import gen_packed_conversion

    for vop3 in (False, True):
        body = gen_packed_conversion(
            context(arch, 'V_CVT_PKRTZ_F16_F32', ['dst'], ['a', 'b'], is_vop3=vop3)
        )
        assert 'rj_arith_v_cvt_pk_rtz_f16_f32_w64' in body
        assert 'RJ_ARITH_FP16_OVFL' in body
        assert 'RJ_ARITH_FP_FLUSH_INPUT_DENORMALS' in body
        assert 'RJ_ARITH_FP_FLUSH_OUTPUT_DENORMALS' in body
        assert ('execute_packed_rtz_sdwa' in body) == (
            arch in ('rdna1', 'rdna2') and not vop3
        )
        assert ('RJ_ARITH_ALU_ABS_B' in body) == vop3


@pytest.mark.parametrize('arch', ['cdna1', 'cdna2', 'rdna1', 'rdna2', 'rdna3_5'])
def test_remaining_integral_f32_uses_legacy_nan_policy(arch):
    for op in ('TRUNC', 'CEIL', 'RNDNE', 'FLOOR'):
        for vop3 in (False, True):
            body = gen_float_unary32(
                context(arch, f'V_{op}_F32', ['dst'], ['src'], is_vop3=vop3)
            )
            suffix = 'gfx12' if arch == 'cdna5' else 'gfx908'
            assert f'rj_arith_v_{op.lower()}_f32_w32_{suffix}' in body
            assert '/*guest_rounding=*/true' in body
            assert ('execute_float_round_sdwa' in body) != vop3
            assert ('inst_.omod' in body) == vop3


@pytest.mark.parametrize(
    'arch', ['cdna1', 'cdna2', 'cdna3', 'cdna4', 'cdna5', 'rdna1', 'rdna2', 'rdna3_5']
)
def test_remaining_integral_f64_keeps_legacy_nan_and_architecture_policy(arch):
    from amdisa.codegen.execute.rj_arith_float import gen_float_unary64

    for op in ('TRUNC', 'CEIL', 'RNDNE', 'FLOOR'):
        for vop3 in (False, True):
            body = gen_float_unary64(
                context(arch, f'V_{op}_F64', ['dst'], ['src'], is_vop3=vop3)
            )
            suffix = (
                'gfx12'
                if arch == 'cdna5'
                else (
                    'gfx942'
                    if arch == 'cdna3'
                    else 'gfx950' if arch == 'cdna4' else 'gfx908'
                )
            )
            assert f'rj_arith_v_{op.lower()}_f64_w32_{suffix}' in body
            assert ('inst_.omod' in body) == vop3
            assert 'wf.cu().arithmetic_flags()' in body
    # Reciprocal/root and mantissa policy migrations remain independent.
    assert gen_float_unary64(context(arch, 'V_SQRT_F64', ['dst'], ['src'])) is not None


@pytest.mark.parametrize(
    'arch', ['cdna1', 'cdna2', 'cdna3', 'cdna4', 'cdna5', 'rdna1', 'rdna2', 'rdna3_5']
)
def test_inherited_exponents_preserve_integer_output_and_qualification(arch):
    from amdisa.codegen.execute.rj_arith_float import (
        gen_float_unary32,
        gen_float_unary64,
    )

    for bits, generate in [(32, gen_float_unary32), (64, gen_float_unary64)]:
        for vop3 in (False, True):
            body = generate(
                context(
                    arch, f'V_FREXP_EXP_I32_F{bits}', ['dst'], ['src'], is_vop3=vop3
                )
            )
            assert f'rj_arith_v_frexp_exp_i32_f{bits}_w32' in body
            assert 'RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL_NO_FP_STATUS' in body
            assert 'UnsupportedOperandValue' in body
            assert 'inst_.omod' not in body and 'inst_.clamp' not in body
            assert ('execute_exponent_sdwa' in body) == (bits == 32 and not vop3)
            if bits == 64:
                assert '/*integer_result=*/true' in body


@pytest.mark.parametrize(
    'arch', ['cdna1', 'cdna2', 'cdna3', 'cdna4', 'cdna5', 'rdna1', 'rdna2', 'rdna3_5']
)
def test_inherited_mantissas_use_legacy_nan_and_guest_output_policy(arch):
    from amdisa.codegen.execute.rj_arith_float import (
        gen_float_unary32,
        gen_float_unary64,
    )

    for bits, generate in [(32, gen_float_unary32), (64, gen_float_unary64)]:
        # CDNA FP32 production dispatch selects gen_cdna_integral_unary first.
        if bits == 32 and arch in ('cdna3', 'cdna4', 'cdna5'):
            continue
        for vop3 in (False, True):
            body = generate(
                context(arch, f'V_FREXP_MANT_F{bits}', ['dst'], ['src'], is_vop3=vop3)
            )
            suffix = (
                '_gfx12'
                if arch == 'cdna5'
                else (
                    CONVERSION_SUFFIX[arch]
                    if bits == 64 and arch in ('cdna3', 'cdna4')
                    else '_gfx908'
                )
            )
            assert f'rj_arith_v_frexp_mant_f{bits}_w32{suffix}' in body
            assert ('execute_float_unary_sdwa' in body) == (bits == 32 and not vop3)
            if arch == 'cdna5':
                assert 'RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL_NO_FP_STATUS' in body
            if vop3:
                assert 'inst_.omod' in body and 'inst_.clamp' in body


@pytest.mark.parametrize(
    'arch', ['cdna1', 'cdna2', 'cdna3', 'cdna4', 'cdna5', 'rdna1', 'rdna2', 'rdna3_5']
)
def test_inherited_fract_preserves_guest_rounding_and_legacy_output_policy(arch):
    from amdisa.codegen.execute.rj_arith_float import (
        gen_float_unary32,
        gen_float_unary64,
    )

    for bits, generate in [(32, gen_float_unary32), (64, gen_float_unary64)]:
        # CDNA FP32 production dispatch selects gen_cdna_integral_unary first.
        if bits == 32 and arch in ('cdna3', 'cdna4', 'cdna5'):
            continue
        for vop3 in (False, True):
            body = generate(
                context(arch, f'V_FRACT_F{bits}', ['dst'], ['src'], is_vop3=vop3)
            )
            suffix = (
                '_gfx12'
                if arch == 'cdna5'
                else (
                    CONVERSION_SUFFIX[arch]
                    if bits == 64 and arch in ('cdna3', 'cdna4')
                    else '_gfx908'
                )
            )
            assert f'rj_arith_v_fract_f{bits}_w32{suffix}' in body
            assert ('execute_float_unary_sdwa' in body) == (bits == 32 and not vop3)
            if bits == 32:
                assert '/*guest_rounding=*/true' in body
            if arch == 'cdna5':
                assert 'RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL_NO_FP_STATUS' in body
            if vop3:
                assert 'inst_.omod' in body and 'inst_.clamp' in body


@pytest.mark.parametrize('arch', ['cdna1', 'cdna2', 'rdna1', 'rdna2', 'rdna3_5'])
def test_older_native_f64_reuses_raw_host_model(arch):
    from amdisa.codegen.execute.rj_arith_float import gen_float_unary64

    for operation in ('RCP', 'RSQ', 'SQRT'):
        for vop3 in (False, True):
            body = gen_float_unary64(
                context(arch, f'V_{operation}_F64', ['dst'], ['src'], is_vop3=vop3)
            )
            assert f'rj_arith_v_{operation.lower()}_f64_w32_gfx908' in body
            assert 'execute_float_unary64' in body
            if vop3:
                assert 'inst_.omod' in body and 'inst_.clamp' in body
    # Older integral rounding belongs to independent migrations.
    assert gen_float_unary64(context(arch, 'V_TRUNC_F64', ['dst'], ['src'])) is not None


@pytest.mark.parametrize('arch', ['cdna1', 'cdna2', 'rdna1', 'rdna2', 'rdna3_5'])
def test_older_trig_preop_reuses_legacy_table_without_claiming_ldexp(arch):
    from amdisa.codegen.execute.rj_arith_float import gen_float64_word

    body = gen_float64_word(
        context(arch, 'V_TRIG_PREOP_F64', ['dst'], ['src', 'selector'], is_vop3=True)
    )
    assert 'rj_arith_v_trig_preop_f64_w32_gfx942' in body
    assert 'rj_arith_v_trig_preop_f64_w64_gfx942' in body
    assert '/*preop_rounding=*/true' in body
    assert 'true, /*ldexp=*/false' in body
    # The combined graph also contains the independent FP64 arithmetic owner.
    ldexp = gen_float64_word(
        context(arch, 'V_LDEXP_F64', ['dst'], ['src', 'exponent'], is_vop3=True)
    )
    assert 'rj_arith_v_ldexp_f64_w32_gfx942' in ldexp
    assert '/*ldexp=*/true' in ldexp


@pytest.mark.parametrize('arch', ['cdna1', 'cdna2', 'rdna1', 'rdna2', 'rdna3_5'])
@pytest.mark.parametrize(
    'name', ['V_RCP_F32', 'V_RCP_IFLAG_F32', 'V_RSQ_F32', 'V_SQRT_F32']
)
@pytest.mark.parametrize('vop3', [False, True])
def test_older_roots_reuse_the_inherited_cdna_model(arch, name, vop3):
    body = gen_float_unary32(context(arch, name, ['dst'], ['src'], is_vop3=vop3))
    assert f'rj_arith_{name.lower()}_w32_gfx942,' in body
    assert f'rj_arith_{name.lower()}_w64_gfx942,' in body
    assert 'guest_rounding=*/false' in body
    assert ('compatibility_all_exec=*/true' in body) == (
        name in {'V_RCP_IFLAG_F32', 'V_SQRT_F32'}
    )
    if not vop3 and arch != 'rdna3_5':
        assert 'execute_cdna_float_unary' in body
    else:
        assert 'execute_cdna_float_unary' not in body


@pytest.mark.parametrize('arch', ['cdna1', 'cdna2', 'rdna1', 'rdna2', 'rdna3_5'])
@pytest.mark.parametrize('name', ['V_EXP_F32', 'V_LOG_F32', 'V_SIN_F32', 'V_COS_F32'])
@pytest.mark.parametrize('vop3', [False, True])
def test_older_special_math_reuses_existing_policy(arch, name, vop3):
    body = gen_float_unary32(context(arch, name, ['dst'], ['src'], is_vop3=vop3))
    assert f'rj_arith_{name.lower()}_w32_gfx942,' in body
    assert f'rj_arith_{name.lower()}_w64_gfx942,' in body
    assert 'guest_rounding=*/false' in body
    assert 'compatibility_all_exec' not in body
    if not vop3 and arch != 'rdna3_5':
        assert 'execute_cdna_float_unary' in body
    else:
        assert 'execute_cdna_float_unary' not in body


@pytest.mark.parametrize('arch', ['cdna1', 'cdna2'])
@pytest.mark.parametrize('name', ['V_EXP_LEGACY_F32', 'V_LOG_LEGACY_F32'])
@pytest.mark.parametrize('vop3', [False, True])
def test_older_legacy_exp_log_reuses_the_ordinary_math_model(arch, name, vop3):
    body = gen_float_unary32(context(arch, name, ['dst'], ['src'], is_vop3=vop3))
    for wave in (32, 64):
        assert f'rj_arith_{name.lower()}_w{wave}_gfx942,' in body
    assert ('execute_cdna_float_unary' in body) == (not vop3)


@pytest.mark.parametrize(
    'arch', ['cdna1', 'cdna2', 'rdna1', 'rdna2', 'rdna3_5', 'cdna3', 'cdna4']
)
@pytest.mark.parametrize('op', ['RCP', 'RSQ', 'SQRT', 'EXP', 'LOG'])
def test_older_half_roots_use_fixed_rounding_and_staged_sdwa(arch, op):
    from amdisa.codegen.execute.rj_arith_float import gen_float_unary16

    for vop3 in (False, True):
        body = gen_float_unary16(
            context(arch, f'V_{op}_F16', ['d'], ['a'], is_vop3=vop3)
        )
        suffix = 'gfx942' if arch in ('cdna3', 'cdna4') else 'gfx908'
        assert f'rj_arith_v_{op.lower()}_f16_w32_{suffix}' in body
        assert '/*guest_rounding=*/false' in body
        if arch in ('cdna3', 'cdna4'):
            assert (
                'HalfUnaryWriter' in body if vop3 else 'sdwa_half_math_writer' in body
            )
        else:
            assert ('execute_half_unary_sdwa' in body) == (not vop3)
        if arch not in ('cdna3', 'cdna4'):
            assert 'RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL_NO_FP_STATUS' in body


@pytest.mark.parametrize('vop3', [False, True])
@pytest.mark.parametrize('op', ['RCP', 'RSQ', 'SQRT', 'EXP', 'LOG'])
def test_gfx1250_half_roots_keep_qualification_and_true16_layout(vop3, op):
    from amdisa.codegen.execute.rj_arith_float import gen_float_unary16

    body = gen_float_unary16(
        context('cdna5', f'V_{op}_F16', ['d'], ['a'], is_vop3=vop3)
    )
    assert f'rj_arith_v_{op.lower()}_f16_w32_gfx1250' in body
    assert 'nullptr' in body
    assert body.index('if (!wf.exec())') < body.index('arithmetic_flags()')
    assert 'guest_rounding=*/false' in body
    # e32 uses the decoder's half-width operand views; VOP3 carries OP_SEL.
    assert ('RJ_ARITH_ALU_HIGH_A' in body) == vop3
    assert ('RJ_ARITH_ALU_HIGH_D' in body) == vop3


@pytest.mark.parametrize(
    'arch', ['cdna1', 'cdna2', 'rdna1', 'rdna2', 'rdna3_5', 'cdna3', 'cdna4']
)
@pytest.mark.parametrize('op', ['SIN', 'COS'])
def test_older_half_trig_inherits_policy_and_staged_sdwa(arch, op):
    from amdisa.codegen.execute.rj_arith_float import gen_float_unary16

    for vop3 in (False, True):
        body = gen_float_unary16(
            context(arch, f'V_{op}_F16', ['d'], ['a'], is_vop3=vop3)
        )
        suffix = 'gfx942' if arch in ('cdna3', 'cdna4') else 'gfx908'
        assert f'rj_arith_v_{op.lower()}_f16_w32_{suffix}' in body
        assert '/*guest_rounding=*/false' in body
        if arch in ('cdna3', 'cdna4'):
            assert (
                'HalfUnaryWriter' in body if vop3 else 'sdwa_half_math_writer' in body
            )
        else:
            assert ('execute_half_unary_sdwa' in body) == (not vop3)
        if arch not in ('cdna3', 'cdna4'):
            assert 'RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL_NO_FP_STATUS' in body


@pytest.mark.parametrize('vop3', [False, True])
@pytest.mark.parametrize('op', ['SIN', 'COS'])
def test_gfx1250_half_trig_keeps_qualification_and_true16_layout(vop3, op):
    from amdisa.codegen.execute.rj_arith_float import gen_float_unary16

    body = gen_float_unary16(
        context('cdna5', f'V_{op}_F16', ['d'], ['a'], is_vop3=vop3)
    )
    assert f'rj_arith_v_{op.lower()}_f16_w32_gfx1250' in body
    assert 'nullptr' in body
    assert body.index('if (!wf.exec())') < body.index('arithmetic_flags()')
    assert 'guest_rounding=*/false' in body
    # e32 uses the decoder's half-width operand views; VOP3 carries OP_SEL.
    assert ('RJ_ARITH_ALU_HIGH_A' in body) == vop3
    assert ('RJ_ARITH_ALU_HIGH_D' in body) == vop3


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
            is not None
        )


@pytest.mark.parametrize('arch', ['cdna1', 'cdna2', 'rdna1', 'rdna2', 'rdna3_5'])
def test_older_selection_rejects_unqualified_strict_requests(arch):
    from amdisa.codegen.execute.rj_arith_float import (
        gen_float_binary64,
        gen_float_ternary_selection,
    )

    for name, emitter, sources in (
        ('V_MIN_F32', gen_float_binary32, ['a', 'b']),
        ('V_MIN_F64', gen_float_binary64, ['a', 'b']),
        ('V_MIN3_F32', gen_float_ternary_selection, ['a', 'b', 'c']),
    ):
        for vop3 in (True, False) if name == 'V_MIN_F32' else (True,):
            body = emitter(context(arch, name, ['d'], sources, is_vop3=vop3))
            assert body.index('!wf.exec()') < body.index('UnsupportedOperandValue')
            assert 'SEMANTICS_BIT_EXACT_EMPIRICAL_NO_FP_STATUS' in body
            assert body.index('UnsupportedOperandValue') < body.index(
                'amdgpu::arith::execute_'
            )


@pytest.mark.parametrize('arch', ['cdna1', 'cdna2'])
def test_old_cdna_conversion_half_selectors(arch):
    from amdisa.codegen.execute.rj_arith_float import (
        CONVERSION16_NAMES,
        gen_conversion16,
    )

    for name in CONVERSION16_NAMES:
        body = gen_conversion16(context(arch, name, ['dst'], ['src'], is_vop3=True))
        assert 'opsel = amdgpu::vop3_opsel(inst_)' in body
        if not name.startswith('V_CVT_F32_'):
            assert 'RJ_ARITH_ALU_HIGH_D) ? 0xC : 0xF' in body


@pytest.mark.parametrize('arch', ['cdna1', 'cdna2', 'rdna1', 'rdna2'])
def test_older_fp32_fma_and_mad_reject_active_unqualified_strict_modes(arch):
    from amdisa.codegen.execute.rj_arith_float import gen_float_mad32

    cases = [(gen_float_fma32, 'V_FMA_F32')]
    if arch != 'rdna2':
        cases.append((gen_float_mad32, 'V_MAD_F32'))
    for generate, name in cases:
        body = generate(
            context(arch, name, ['vdst'], ['src0', 'src1', 'src2'], is_vop3=True)
        )
        assert body.index('if (!wf.exec())') < body.index('const uint64_t semantics')
        assert 'RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL_NO_FP_STATUS' in body
        assert 'RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL)' in body
        assert 'UnsupportedOperandValue' in body
