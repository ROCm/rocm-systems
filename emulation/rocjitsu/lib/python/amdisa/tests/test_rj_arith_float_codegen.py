from pathlib import Path


import re


import pytest


from amdisa.codegen.execute.rj_arith_float import (
    UNARY_F32,
    BINARY_F32,
    FMA_F32,
    gen_float_fma32,
    gen_float_unary32,
    gen_float_binary32,
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


@pytest.mark.parametrize('arch', ['rdna3', 'rdna4'])
@pytest.mark.parametrize('name', sorted(FMA_F32))
def test_fma_operands_modifiers_and_optional_state(arch, name):
    accumulate = name in ('V_FMAC_F32', 'V_FMAC_DX9_ZERO_F32')
    literal = name in ('V_FMAAK_F32', 'V_FMAMK_F32')
    sources = ['a', 'b'] if accumulate else ['a', 'b', 'c']
    body = gen_float_fma32(arch, name, ['dst'], sources, not literal)
    if arch == 'rdna4' and name == 'V_FMAC_DX9_ZERO_F32':
        assert body is None
        return
    assert 'dst, a, b, dst' in body if accumulate else 'dst, a, b, c' in body
    assert ('RJ_ARITH_ALU_NEG_C' in body) == (not accumulate and not literal)
    assert ('RJ_ARITH_ALU_ABS_C' in body) == (not accumulate and not literal)
    assert 'arithmetic_flags()' in body
    assert ('arithmetic_reports_fp_status' in body) == ('DX9_ZERO' not in name)
    assert gen_float_fma32('cdna4', name, ['dst'], sources, not literal) is None
    assert (
        gen_float_fma32(arch, name, ['dst'], sources + ['extra'], not literal) is None
    )


def test_literal_fma_sources_keep_their_arithmetic_order():
    assert 'dst, a, literal, b' in gen_float_fma32(
        'rdna4', 'V_FMAMK_F32', ['dst'], ['a', 'literal', 'b'], False
    )
    assert 'dst, a, b, literal' in gen_float_fma32(
        'rdna3', 'V_FMAAK_F32', ['dst'], ['a', 'b', 'literal'], False
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
        arch,
        'V_PK_FMA_F16',
        ['d'],
        ['a', 'b', 'c'],
        'ENC_VOP3P',
        ('low', 'high'),
        'high_c',
    )
    assert 'execute_packed_fma16<false>' in body
    assert 'RJ_ARITH_PK_NEG_LO_C' in body and 'RJ_ARITH_PK_NEG_HI_C' in body
    assert 'pk16_src_needs_narrowing(inst_.src2, c.size_bits())' in body
    assert 'replicate_sources = 0' in body
    body = gen_packed_fma16(
        arch, 'V_PK_FMAC_F16', ['d'], ['a', 'b'], 'ENC_VOP2', ('low', 'high'), 'high_c'
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
        arch, name, ['d'], ['a', 'b'], 'ENC_VOP3P', ('low', 'high')
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
    assert 'arithmetic_reports_fp_status()' in body
    assert (
        gen_packed_binary16(
            'cdna4', name, ['d'], ['a', 'b'], 'ENC_VOP3P', ('low', 'high')
        )
        is None
    )
    assert (
        gen_packed_binary16(arch, name, ['d'], ['a', 'b'], 'ENC_VOP3', ('low', 'high'))
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
        assert 'arithmetic_reports_fp_status' in body, match[1]
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
        arch, name, ['d'], ['a', 'b'], 'ENC_VOP3P', ('low', 'high')
    )
    legacy = name in ('V_PK_MIN_F16', 'V_PK_MAX_F16')
    if legacy != (arch == 'rdna3'):
        assert body is None
        return
    api = 'rj_arith_' + name.lower()
    assert api + '_w32' + ', ' + api + '_w64' in body
    assert 'narrow_sources' in body
    assert 'arithmetic_flags()' in body
    assert 'arithmetic_reports_fp_status()' in body


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
        'rdna4', name, ['d'], ['a', 'b', 'c'], 'ENC_VOP3P', ('low', 'high')
    )
    assert 'rj_arith_' + name.lower() in body
    assert 'inst_.neg & 4' in body and 'RJ_ARITH_DOT_NEG_C' in body
    assert 'inst_.neg_hi & 4' in body and 'RJ_ARITH_DOT_ABS_C' in body
    assert 'arithmetic_flags()' in body
    assert (
        gen_float_dot32(
            'rdna3', name, ['d'], ['a', 'b', 'c'], 'ENC_VOP3P', ('low', 'high')
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

    body = gen_half_dot(arch, name, ['d'], ['a', 'b', 'c'], True)
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
    assert gen_half_dot('cdna4', name, ['d'], ['a', 'b', 'c'], True) is None


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

    body = gen_float_binary16(arch, name, ['d'], ['a', 'b'], True)
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
    assert 'arithmetic_reports_fp_status()' in body
    vop2 = gen_float_binary16(arch, name, ['d'], ['a', 'b'], False)
    assert 'inst_.op' not in vop2 and 'RJ_ARITH_ALU_HIGH' not in vop2
    assert gen_float_binary16('rdna3_5', name, ['d'], ['a', 'b'], True) is None


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
        body = gen_float_binary16(arch, name, ['d'], ['a', 'b'], True)
        api = 'rj_arith_' + name.lower()
        assert api + '_w32' + ', ' + api + '_w64' in body
        assert 'arithmetic_flags()' in body
        assert 'arithmetic_reports_fp_status()' in body
        assert (
            gen_float_binary16(
                'rdna4' if arch == 'rdna3' else 'rdna3', name, ['d'], ['a', 'b'], True
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
        arch, name, ['d'], inputs, name in ('V_FMA_F16', 'V_FMAC_F16')
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
    assert gen_float_fma16('rdna3_5', name, ['d'], inputs, False) is None


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
        body = gen_float_ternary_selection(arch, name, ['d'], ['a', 'b', 'c'], True)
        assert 'rj_arith_' + name.lower() + '_w64' in body
        assert 'arithmetic_flags()' in body
        assert 'arithmetic_reports_fp_status()' in body
        for letter in 'ABC':
            for flag in ('ABS_', 'NEG_'):
                assert 'RJ_ARITH_ALU_' + flag + letter in body
        if name.endswith('_F16'):
            for letter in 'ABCD':
                assert 'RJ_ARITH_ALU_HIGH_' + letter in body
        else:
            assert 'RJ_ARITH_ALU_HIGH' not in body
        assert (
            gen_float_ternary_selection('cdna4', name, ['d'], ['a', 'b', 'c'], True)
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
def test_fp64_selection_pair_adapter(arch):
    from amdisa.codegen.execute.rj_arith_float import SELECTION_F64, gen_float_binary64

    for name in SELECTION_F64[arch]:
        body = gen_float_binary64(arch, name, ['dst'], ['a', 'b'], True)
        assert 'execute_float_binary64' in body
        assert 'rj_arith_' + name.lower() + '_w64' in body
        assert 'arithmetic_flags()' in body
        assert 'arithmetic_reports_fp_status()' in body
        for modifier in ('ABS_A', 'ABS_B', 'NEG_A', 'NEG_B', 'OMOD_2', 'CLAMP'):
            assert 'RJ_ARITH_ALU_' + modifier in body
        assert gen_float_binary64('cdna4', name, ['dst'], ['a', 'b'], True) is None
        assert gen_float_binary64(arch, name, ['dst'], ['a'], True) is None
        assert 'modifiers = 0' in gen_float_binary64(
            arch, name, ['dst'], ['a', 'b'], False
        )


@pytest.mark.parametrize('arch', ['rdna3', 'rdna4'])
def test_fp64_binary_arithmetic_adapter(arch):
    from amdisa.codegen.execute.rj_arith_float import gen_float_binary64

    for name in ('V_ADD_F64', 'V_MUL_F64'):
        for vop3 in (False, True):
            body = gen_float_binary64(arch, name, ['dst'], ['a', 'b'], vop3)
            arch_suffix = '_gfx12' if arch == 'rdna4' else ''
            api = 'rj_arith_' + name.lower()
            assert api + '_w64' + arch_suffix in body
            assert 'execute_float_binary64' in body
            assert 'arithmetic_flags()' in body
            assert 'arithmetic_reports_fp_status()' in body


@pytest.mark.parametrize('arch', ['rdna3', 'rdna4'])
def test_fma64_pair_layout_modifiers_and_policy(arch):
    from amdisa.codegen.execute.rj_arith_float import FLOAT_NAMES, gen_float_fma64

    body = gen_float_fma64(arch, 'V_FMA_F64', ['d'], ['a', 'b', 'c'], True)
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
    assert 'arithmetic_reports_fp_status()' in body
    assert 'V_FMA_F64' in FLOAT_NAMES
    assert gen_float_fma64('cdna4', 'V_FMA_F64', ['d'], ['a', 'b', 'c'], True) is None
    assert gen_float_fma64(arch, 'V_FMA_F64', ['d'], ['a', 'b'], True) is None
    assert gen_float_fma64(arch, 'V_FMA_F64', ['d'], ['a', 'b', 'c'], False) is None

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
            arch, name, ['d'], ['a', 'b', 'c'], 'ENC_VOP3P', ('low', 'high'), 'high_c'
        )
        assert 'execute_mixed_fma' in body
        assert 'arithmetic_flags()' in body
        assert 'arithmetic_reports_fp_status()' in body
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
                'cdna4',
                name,
                ['d'],
                ['a', 'b', 'c'],
                'ENC_VOP3P',
                ('low', 'high'),
                'high_c',
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
