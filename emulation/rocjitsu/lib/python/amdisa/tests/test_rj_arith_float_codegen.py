# Copyright (c) 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

from pathlib import Path
import re

import pytest

from amdisa.codegen.execute.rj_arith_float import (
    FMA_F32,
    UNARY_F32,
    BINARY_F32,
    gen_float_binary32,
    gen_float_fma32,
    gen_float_unary32,
)


@pytest.mark.parametrize('arch', ['rdna3', 'rdna4'])
@pytest.mark.parametrize('name', sorted(UNARY_F32))
def test_unary_float_bindings_and_policy(arch, name):
    body = gen_float_unary32(context(arch, name, ['dst'], ['src'], is_vop3=True))
    arch_suffix = '_gfx12' if arch == 'rdna4' else ''
    api = 'rj_arith_' + name.lower()
    header = public_headers()
    for suffix, mask in (('_w32', 'uint32_t'), ('_w64', 'uint64_t')):
        function = api + suffix + arch_suffix
        assert function in body
        declaration = re.search(r'\bint ' + function + r'\(([^;]+)\);', header)
        assert declaration is not None
        parameters = declaration[1].split(',')
        assert len(parameters) == 6
        assert parameters[1].strip().startswith(mask + ' ')
        assert 'uint32_t *' in parameters[-1]
    assert 'RJ_ARITH_ALU_ABS_A' in body
    assert 'RJ_ARITH_ALU_NEG_A' in body
    # Exponent results are integers and ignore floating-point output modifiers.
    assert ('RJ_ARITH_ALU_OMOD_2' in body) == (not name.startswith('V_FREXP_EXP_'))
    assert ('RJ_ARITH_ALU_CLAMP' in body) == (not name.startswith('V_FREXP_EXP_'))
    assert 'arithmetic_flags()' in body
    assert 'arithmetic_reports_fp_status' not in body
    assert 'inst_.clamp' not in gen_float_unary32(
        context(arch, name, ['dst'], ['src'], is_vop3=False)
    )
    assert (
        gen_float_unary32(context(arch, name, ['dst'], ['src', 'extra'], is_vop3=False))
        is None
    )


def test_other_architectures_keep_existing_path():
    for arch in ('rdna1', 'rdna2', 'rdna3_5', 'cdna4', 'cdna5'):
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


@pytest.mark.parametrize('arch', ['rdna3', 'rdna4'])
@pytest.mark.parametrize('name', sorted(BINARY_F32))
def test_binary_float_bindings_match_architecture_and_policy(arch, name):
    body = gen_float_binary32(context(arch, name, ['dst'], ['a', 'b'], is_vop3=True))
    rdna3_only = name in ('V_MIN_F32', 'V_MAX_F32')
    rdna4_only = name in (
        'V_MIN_NUM_F32',
        'V_MAX_NUM_F32',
        'V_MINIMUM_F32',
        'V_MAXIMUM_F32',
    )
    if (rdna3_only and arch != 'rdna3') or (rdna4_only and arch != 'rdna4'):
        assert body is None
        return
    arch_suffix = '_gfx12' if arch == 'rdna4' and not rdna4_only else ''
    api = 'rj_arith_' + name.lower()
    header = public_headers()
    for suffix, mask in (('_w32', 'uint32_t'), ('_w64', 'uint64_t')):
        function = api + suffix + arch_suffix
        assert function in body
        declaration = re.search(r'\bint ' + function + r'\(([^;]+)\);', header)
        assert declaration is not None
        parameters = declaration[1].split(',')
        assert len(parameters) == 7
        assert parameters[1].strip().startswith(mask + ' ')
    for modifier in ('ABS_A', 'ABS_B', 'NEG_A', 'NEG_B', 'OMOD_2', 'CLAMP'):
        assert ('RJ_ARITH_ALU_' + modifier in body) == (
            name != 'V_LDEXP_F32' or modifier not in ('ABS_B', 'NEG_B')
        )
    assert 'arithmetic_flags()' in body
    assert 'arithmetic_reports_fp_status' not in body
    assert ('/*compatibility_mul=*/true' in body) == (name == 'V_MUL_F32')
    vop2 = gen_float_binary32(context(arch, name, ['dst'], ['a', 'b'], is_vop3=False))
    assert ('/*compatibility_mul=*/true' in vop2) == (name == 'V_MUL_F32')
    assert 'inst_.clamp' not in gen_float_binary32(
        context(arch, name, ['dst'], ['a', 'b'], is_vop3=False)
    )
    assert (
        gen_float_binary32(context(arch, name, ['dst'], ['a'], is_vop3=False)) is None
    )
    assert (
        gen_float_binary32(context('cdna4', name, ['dst'], ['a', 'b'], is_vop3=False))
        is None
    )


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


from types import SimpleNamespace
from amdisa.codegen.execute import ExecuteContext
from rj_arith_headers import public_headers


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
    assert 'arithmetic_flags()' in body
    assert (
        gen_float_fma32(context('cdna4', name, ['dst'], sources, is_vop3=not literal))
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


@pytest.mark.parametrize('arch', ['rdna3', 'rdna4'])
def test_dot2_operand_transformations_and_accumulator(arch):
    from amdisa.codegen.execute.rj_arith_float import gen_float_dot32

    for bf16 in (False, True):
        name = 'V_DOT2_F32_' + ('BF16' if bf16 else 'F16')
        body = gen_float_dot32(
            context(
                arch,
                name,
                ['d'],
                ['a', 'b', 'c'],
                enc_name='ENC_VOP3P',
                opsel_exprs=('low', 'high'),
            )
        )
        arch_suffix = '_gfx12' if arch == 'rdna4' else ''
        api = 'rj_arith_' + name.lower()
        assert api + '_w32' + arch_suffix + ', ' + api + '_w64' + arch_suffix in body
        assert 'arithmetic_flags()' in body
        assert 'RJ_ARITH_DOT_NEG_C' in body and 'RJ_ARITH_DOT_NEG_HI_A' in body
        assert 'RJ_ARITH_DOT_LO_B_HIGH' in body and 'RJ_ARITH_DOT_HI_B_LOW' in body
        if bf16:
            assert 'dot2_src_needs_half_replication' in body
            if arch == 'rdna4':
                assert 'is_inline_float_src' in body
        else:
            assert 'pk16_src_needs_narrowing' in body
            assert 'replicate_sources = 0' in body
    body = gen_float_dot32(
        context(
            arch,
            'V_DOT2ACC_F32_F16',
            ['d'],
            ['a', 'b'],
            enc_name='ENC_VOP2',
            opsel_exprs=('low', 'high'),
        )
    )
    if arch == 'rdna4':
        assert body is None
    else:
        assert 'd, a, b, d' in body
        assert 'inst_.src1' not in body
        assert 'modifiers = 0' in body


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


# RDNA4 now emits separate ordinary and modifier executors for these six instructions.
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
def test_half_unary_modes_architecture_and_all_generated_encodings(arch):
    from amdisa.codegen.execute.rj_arith_float import UNARY_F16, gen_float_unary16

    root = (
        Path(__file__).resolve().parents[3]
        / 'rocjitsu/src/rocjitsu/isa/arch/amdgpu/generated'
        / arch
    )
    classes = {
        'V' + ''.join(part.title() for part in name.split('_')[1:])
        for name in UNARY_F16
    }
    for name in UNARY_F16:
        body = gen_float_unary16(context(arch, name, ['d'], ['a'], is_vop3=True))
        arch_suffix = '_gfx12' if arch == 'rdna4' else ''
        api = 'rj_arith_' + name.lower()
        assert api + '_w32' + arch_suffix + ', ' + api + '_w64' + arch_suffix in body
        for flag in ('HIGH_A', 'HIGH_D', 'ABS_A', 'NEG_A'):
            assert 'RJ_ARITH_ALU_' + flag in body
        for flag in ('OMOD_2', 'CLAMP'):
            assert ('RJ_ARITH_ALU_' + flag in body) == (
                not name.startswith('V_FREXP_EXP_')
            )
        assert 'arithmetic_flags()' in body
        assert 'arithmetic_reports_fp_status' not in body
        assert (
            gen_float_unary16(context('cdna4', name, ['d'], ['a'], is_vop3=True))
            is None
        )
    count = 0
    for encoding in ('vop1', 'vop3'):
        source = (root / (encoding + '_exec.cpp')).read_text()
        for match in re.finditer(
            r'void (V\w+F16)Vop[13]::execute(?:_modifier)?_impl\([^)]*\)\s*\{', source
        ):
            if match[1] not in classes:
                continue
            end, depth = match.end(), 1
            while depth:
                depth += (source[end] == '{') - (source[end] == '}')
                end += 1
            body = source[match.end() : end - 1]
            assert 'arith::execute_float_unary16' in body
            assert 'ROCJITSU_TRY_SIMD' not in body
            assert 'try_execute_words_simd' not in body
            count += 1
    assert count == 56


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
def test_fp64_unary_generated_paths(arch):
    from amdisa.codegen.execute.rj_arith_float import UNARY_F64, gen_float_unary64

    for name in sorted(UNARY_F64):
        body = gen_float_unary64(context(arch, name, ['dst'], ['src'], is_vop3=True))
        arch_suffix = '_gfx12' if arch == 'rdna4' else ''
        api = 'rj_arith_' + name.lower()
        assert api + '_w64' + arch_suffix in body
        assert 'execute_float_unary64' in body
        assert ('/*integer_result=*/true' in body) == (name == 'V_FREXP_EXP_I32_F64')
        assert 'arithmetic_flags()' in body
        assert 'arithmetic_reports_fp_status' not in body
        for modifier in ('ABS_A', 'NEG_A'):
            assert 'RJ_ARITH_ALU_' + modifier in body
        for modifier in ('OMOD_2', 'CLAMP'):
            assert ('RJ_ARITH_ALU_' + modifier in body) == (
                not name.startswith('V_FREXP_EXP_')
            )
        assert (
            gen_float_unary64(context('cdna4', name, ['dst'], ['src'], is_vop3=True))
            is None
        )
        assert (
            gen_float_unary64(
                context(arch, name, ['dst'], ['src', 'extra'], is_vop3=True)
            )
            is None
        )
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
            if 'arith::execute_float_unary64' not in body:
                continue
            assert 'RJ_SIMD' not in body
            assert 'try_execute_' not in body
            assert 'amdgpu::execute_v_' not in body
            checked += 1
    assert checked == 20


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
        gen_float64_word(context('cdna4', name, ['dst'], ['a', 'b'], is_vop3=True))
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
                context('cdna4', name, ['dst'], ['a', 'b'], is_vop3=True)
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


@pytest.mark.parametrize('arch', ['rdna3', 'rdna4'])
def test_fp64_binary_arithmetic_adapter(arch):
    from amdisa.codegen.execute.rj_arith_float import gen_float_binary64

    for name in ('V_ADD_F64', 'V_MUL_F64'):
        for vop3 in (False, True):
            body = gen_float_binary64(
                context(arch, name, ['dst'], ['a', 'b'], is_vop3=vop3)
            )
            arch_suffix = '_gfx12' if arch == 'rdna4' else ''
            api = 'rj_arith_' + name.lower()
            assert api + '_w64' + arch_suffix in body
            assert 'execute_float_binary64' in body
            assert 'arithmetic_flags()' in body


@pytest.mark.parametrize('arch', ['rdna3', 'rdna4'])
def test_cube_reuses_ternary_adapter_and_guest_policies(arch):
    from amdisa.codegen.execute.rj_arith_float import CUBE_NAMES, gen_cube

    for name in CUBE_NAMES:
        body = gen_cube(context(arch, name, ['dst'], ['x', 'y', 'z'], is_vop3=True))
        arch_suffix = '_gfx12' if arch == 'rdna4' else ''
        api = 'rj_arith_' + name.lower()
        assert api + '_w64' + arch_suffix in body
        assert 'execute_ternary32' in body
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


@pytest.mark.parametrize('arch', ['rdna3', 'rdna4'])
def test_byte_conversion_policy_and_selected_read_byte(arch):
    from amdisa.codegen.execute.rj_arith_float import gen_byte_nibble_conversion

    for index in range(5):
        byte = index % 4
        name = f'V_CVT_F32_UBYTE{index}' if index < 4 else 'V_CVT_OFF_F32_I4'
        for vop3 in (False, True):
            body = gen_byte_nibble_conversion(
                context(arch, name, ['dst'], ['src'], is_vop3=vop3)
            )
            arch_suffix = '_gfx12' if arch == 'rdna4' else ''
            api = 'rj_arith_' + name.lower()
            assert api + '_w64' + arch_suffix in body
            assert 'execute_unary32' in body
            assert f'dst, src, {1 << byte}' in body
            assert 'arithmetic_flags()' in body
            assert 'RJ_ARITH_FP_IEEE_MODE' in body
            assert 'RJ_ARITH_FP_FLUSH_OUTPUT_DENORMALS' in body
            assert ('inst_.omod' in body) == vop3
            assert 'inst_.abs' not in body and 'inst_.neg' not in body
        assert (
            gen_byte_nibble_conversion(
                context('cdna4', name, ['dst'], ['src'], is_vop3=True)
            )
            is None
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
            assert ('inst_.opsel' in body) == vop3
            assert 'source_bytes' in body
            assert 'arithmetic_reports_fp_status' not in body
            assert 'inst_.clamp' not in body
        assert (
            gen_fp8_widen(context('rdna3', name, ['dst'], ['src'], is_vop3=True))
            is None
        )
        assert (
            gen_fp8_widen(context('cdna5', name, ['dst'], ['src'], is_vop3=True))
            is None
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
        assert (
            gen_fp8_narrow(context('cdna5', name, ['dst'], ['a', 'b'], is_vop3=True))
            is None
        )
        assert (
            gen_fp8_narrow(context('rdna4', name, ['dst'], ['a', 'b'], is_vop3=False))
            is None
        )


@pytest.mark.parametrize('arch', ['rdna3', 'rdna4'])
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
            arch_suffix = '_gfx12' if arch == 'rdna4' else ''
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
        assert (
            gen_conversion64(context('cdna4', name, ['dst'], ['src'], is_vop3=True))
            is None
        )
        assert gen_conversion64(context(arch, name, ['dst'], [], is_vop3=True)) is None


@pytest.mark.parametrize('arch', ['rdna3', 'rdna4'])
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
            arch_suffix = '_gfx12' if arch == 'rdna4' and not fixed_round else ''
            assert api + '_w64' + arch_suffix in body
            assert 'float32_flags' in body
            assert 'arithmetic_flags()' in body
            assert 'arithmetic_reports_fp_status' not in body
            assert ('inst_.omod' in body) == (vop3 and not fixed_round)
            assert ('RJ_ARITH_ALU_ABS_A' in body) == (vop3 and not to_float)
            assert ('RJ_ARITH_ALU_NEG_A' in body) == (vop3 and not to_float)
        assert (
            gen_conversion32(context('cdna4', name, ['dst'], ['src'], is_vop3=True))
            is None
        )
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
        assert (
            gen_conversion16(context('cdna4', name, ['dst'], ['src'], is_vop3=True))
            is None
        )
        assert gen_conversion16(context(arch, name, ['dst'], [], is_vop3=True)) is None


@pytest.mark.parametrize('arch', ['rdna3', 'rdna4'])
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
            arch_suffix = '_gfx12' if arch == 'rdna4' else ''
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
        assert (
            gen_packed_conversion(
                context('cdna4', name, ['dst'], ['a', 'b'], is_vop3=True)
            )
            is None
        )
        assert (
            gen_packed_conversion(context(arch, name, ['dst'], ['a'], is_vop3=True))
            is None
        )


@pytest.mark.parametrize('arch', ['rdna3', 'rdna4'])
def test_normalized_conversion_layouts_and_state_contract(arch):
    from amdisa.codegen.execute.rj_arith_float import NORMALIZED_NAMES, gen_normalized

    for name in NORMALIZED_NAMES:
        packed = '_PK_' in name
        half = name.endswith('_F16')
        sources = ['a', 'b'] if packed else ['a']
        for vop3 in (False, True):
            body = gen_normalized(context(arch, name, ['d'], sources, is_vop3=vop3))
            api = 'rj_arith_' + name.lower()
            arch_suffix = '_gfx12' if packed and half and arch == 'rdna4' else ''
            assert api + '_w64' + arch_suffix in body
            assert 'arithmetic_flags()' in body
            assert 'arithmetic_reports_fp_status' not in body
            assert ('RJ_ARITH_ALU_HIGH_A' in body) == (half and (vop3 or not packed))
            assert ('RJ_ARITH_ALU_HIGH_B' in body) == (packed and half and vop3)
            assert ('RJ_ARITH_ALU_HIGH_D' in body) == (not packed)
            assert 'RJ_ARITH_ALU_OMOD_2' not in body
            assert ('RJ_ARITH_ALU_CLAMP' in body) == (packed and vop3)
            assert 'fp_round_mode' not in body
        assert (
            gen_normalized(context('cdna4', name, ['d'], sources, is_vop3=True)) is None
        )
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


@pytest.mark.parametrize('arch', ['rdna3', 'rdna4'])
def test_bit_pack_input_layouts_and_state_policy(arch):
    from amdisa.codegen.execute.rj_arith_float import BIT_PACK_NAMES, gen_bit_pack

    for name in BIT_PACK_NAMES:
        half = name == 'V_PACK_B32_F16'
        sources = ['a', 'b'] if half else ['a', 'b', 'c']
        body = gen_bit_pack(context(arch, name, ['d'], sources, is_vop3=True))
        arch_suffix = '_gfx12' if arch == 'rdna4' else ''
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
        assert (
            gen_bit_pack(context('cdna4', name, ['d'], sources, is_vop3=True)) is None
        )
        assert gen_bit_pack(context(arch, name, ['d'], sources, is_vop3=False)) is None
        assert gen_bit_pack(context(arch, name, ['d'], [], is_vop3=True)) is None


@pytest.mark.parametrize('arch', ['rdna3', 'rdna4'])
def test_fma64_pair_layout_modifiers_and_policy(arch):
    from amdisa.codegen.execute.rj_arith_float import FLOAT_NAMES, gen_float_fma64

    body = gen_float_fma64(
        context(arch, 'V_FMA_F64', ['d'], ['a', 'b', 'c'], is_vop3=True)
    )
    arch_suffix = '_gfx12' if arch == 'rdna4' else ''
    api = 'rj_arith_v_fma_f64'
    assert api + '_w64' + arch_suffix in body
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
            context('cdna4', 'V_FMA_F64', ['d'], ['a', 'b', 'c'], is_vop3=True)
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
    source = (root / arch / 'vop3_exec.cpp').read_text()
    match = re.search(r'void VFmaF64Vop3::execute_impl\([^)]*\)\s*\{([^}]+)\}', source)
    assert match
    assert 'execute_float_ternary64' in match[1]
    assert 'try_execute_' not in match[1]
    assert 'amdgpu::execute_v_' not in match[1]


@pytest.mark.parametrize('arch', ['rdna3', 'rdna4'])
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
        functions.update(
            re.findall(r'\brj_arith_v_(?:fma|pk_fma)[a-z0-9_]+', path.read_text())
        )
    assert functions
    for function in functions:
        assert re.search(r'\bint ' + function + r'\(', declarations), function


@pytest.mark.parametrize('arch', ['rdna3', 'rdna4', 'cdna5'])
def test_pseudo_scalar_routing_and_semantics(arch):
    from amdisa.codegen.execute.rj_arith_float import (
        PSEUDO_SCALAR_NAMES,
        gen_pseudo_scalar,
    )

    for name in PSEUDO_SCALAR_NAMES:
        body = gen_pseudo_scalar(context(arch, name, ['d'], ['a'], is_vop3=True))
        if arch != 'rdna4':
            assert body is None
            continue
        assert 'execute_pseudo_scalar' in body
        assert 'rj_arith_' + name.lower() in body
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
