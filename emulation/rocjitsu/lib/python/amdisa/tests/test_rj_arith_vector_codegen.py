from pathlib import Path


import re


import pytest


from amdisa.codegen.execute.rj_arith_vector import (
    BINARY32,
    CARRY32_NAMES,
    gen_carry32,
    TERNARY32,
    UNARY32,
    UNMODIFIED_MULTIPLIES,
    gen_vector32,
)


def test_unmigrated_architectures_and_operand_shapes_keep_existing_path():
    for arch in ('rdna1', 'rdna2', 'rdna3_5', 'cdna4', 'cdna5'):
        assert gen_vector32(arch, 'V_AND_B32', ['dst'], ['a', 'b'], False) is None
    assert gen_vector32('rdna4', 'V_AND_B32', ['dst', 'scc'], ['a', 'b'], False) is None
    assert gen_vector32('rdna4', 'V_ADD_F32', ['dst'], ['a', 'b'], False) is None


@pytest.mark.parametrize('arch', ['rdna3', 'rdna4'])
def test_generated_rj_arith_bodies_cannot_be_bypassed_by_legacy_simd(arch):
    root = (
        Path(__file__).resolve().parents[3]
        / 'rocjitsu/src/rocjitsu/isa/arch/amdgpu/generated'
        / arch
    )
    checked = {'unary': 0, 'binary': 0, 'ternary': 0, 'carry': 0}
    for path in root.glob('*_exec.cpp'):
        source = path.read_text()
        for match in re.finditer(
            r'void (\w+)::execute(?:_modifier)?_impl\([^)]*\)\s*\{', source
        ):
            end, depth = match.end(), 1
            while depth:
                depth += (source[end] == '{') - (source[end] == '}')
                end += 1
            body = source[match.end() : end - 1]
            kind = re.search(r'arith::execute_(unary|binary|ternary|carry)32', body)
            if kind is None:
                continue
            assert 'RJ_SIMD' not in body, match[1]
            assert 'try_execute_' not in body, match[1]
            assert 'amdgpu::execute_v_' not in body, match[1]
            checked[kind[1]] += 1
    # Other instruction families add users of these adapters as they migrate.
    assert all(count > 0 for count in checked.values())


@pytest.mark.parametrize('arch', ['rdna3', 'rdna4'])
@pytest.mark.parametrize('name', sorted(CARRY32_NAMES))
@pytest.mark.parametrize('vop3', [False, True])
def test_carry_bindings_keep_decoder_commit_and_snapshot_input(arch, name, vop3):
    with_carry = '_CI_' in name
    dst = ['vdst', 'sdst'] if vop3 else ['vdst']
    src = ['a', 'b', 'cin'] if vop3 and with_carry else ['a', 'b']
    body = gen_carry32(arch, name, dst, src, vop3, 'commit_result')
    assert f'execute_carry32<{str(with_carry).lower()}>' in body
    assert 'rj_arith_' + name.lower() + '_w64' in body
    assert 'input_carry, commit_result' in body
    assert ('inst_.clamp' in body) == vop3
    if with_carry:
        assert ('read_wave_mask_scalar(cin, wf)' if vop3 else 'wf.vcc()') in body
    else:
        assert 'input_carry = 0u' in body
    assert gen_carry32('cdna4', name, dst, src, vop3) is None


@pytest.mark.parametrize('arch', ['rdna3', 'rdna4'])
def test_wide_integer_bindings_and_generated_routing(arch):
    from amdisa.codegen.execute.rj_arith_vector import (
        SHIFT64_NAMES,
        MAD64_NAMES,
        gen_wide_integer,
    )

    for name in SHIFT64_NAMES:
        body = gen_wide_integer(arch, name, ['d'], ['a', 'b'], True)
        assert 'execute_shift64(wf, rj_arith_' + name.lower() in body
        assert 'rj_arith_' + name.lower() + '_w64' in body
        assert 'inst_.clamp' in body
        assert gen_wide_integer(arch, name, ['d'], ['a', 'b'], True, 'commit') is None
    for name in MAD64_NAMES:
        body = gen_wide_integer(
            arch, name, ['d', 'sdst'], ['a', 'b', 'c'], True, 'commit'
        )
        assert 'execute_mad64(wf, rj_arith_' + name.lower() in body
        assert 'rj_arith_' + name.lower() + '_w64' in body
        assert 'd, a, b, c, commit' in body
        assert (
            gen_wide_integer('cdna4', name, ['d', 'sdst'], ['a', 'b', 'c'], True)
            is None
        )
    root = (
        Path(__file__).resolve().parents[3]
        / 'rocjitsu/src/rocjitsu/isa/arch/amdgpu/generated'
        / arch
    )
    checked = {'shift': 0, 'mad': 0}
    for path in root.glob('*_exec.cpp'):
        source = path.read_text()
        for match in re.finditer(
            r'void (\w+)::execute(?:_modifier)?_impl\([^)]*\)\s*\{', source
        ):
            end, depth = match.end(), 1
            while depth:
                depth += (source[end] == '{') - (source[end] == '}')
                end += 1
            body = source[match.end() : end - 1]
            kind = re.search(r'arith::execute_(shift|mad)64', body)
            if kind is None:
                continue
            assert 'RJ_SIMD' not in body, match[1]
            assert 'try_execute_' not in body, match[1]
            assert 'amdgpu::execute_v_' not in body, match[1]
            checked[kind[1]] += 1
    assert checked == {'shift': 3 if arch == 'rdna3' else 4, 'mad': 2}


@pytest.mark.parametrize('arch', ['rdna3', 'rdna4'])
def test_integer16_bindings_own_arithmetic_but_not_dpp_routing(arch):
    from amdisa.codegen.execute.rj_arith_vector import (
        UNARY16,
        BINARY16,
        TERNARY16,
        gen_integer16,
    )

    for names, count in ((UNARY16, 1), (BINARY16, 2), (TERNARY16, 3)):
        for name in names:
            sources = ['a', 'b', 'c'][:count]
            body = gen_integer16(arch, name, ['d'], sources, True)
            assert (
                f'execute_integer16<{count}>(wf, rj_arith_{name.lower()}_w32,' in body
            )
            assert f'rj_arith_{name.lower()}_w64' in body
            assert 'RJ_ARITH_ALU_CLAMP' in body and 'RJ_ARITH_ALU_HIGH_D' in body
            for index, selector in enumerate('ABC'):
                assert (f'RJ_ARITH_ALU_HIGH_{selector}' in body) == (index < count)
            assert 'RJ_ARITH_DPP' not in body
            assert gen_integer16('cdna4', name, ['d'], sources, True) is None
            assert gen_integer16(arch, name, ['d'], sources + ['extra'], True) is None
    body = gen_integer16(arch, 'V_NOT_B16', ['d'], ['a'], False)
    assert 'modifiers = 0' in body
    assert 'inst_.clamp' not in body and 'opsel' not in body
    root = (
        Path(__file__).resolve().parents[3]
        / 'rocjitsu/src/rocjitsu/isa/arch/amdgpu/generated'
        / arch
    )
    counts = {1: 0, 2: 0, 3: 0}
    for path in root.glob('*_exec.cpp'):
        source = path.read_text()
        for match in re.finditer(
            r'void (\w+)::execute(?:_modifier)?_impl\([^)]*\)\s*\{', source
        ):
            end, depth = match.end(), 1
            while depth:
                depth += (source[end] == '{') - (source[end] == '}')
                end += 1
            body = source[match.end() : end - 1]
            count = re.search(r'arith::execute_integer16<([123])>', body)
            if count is None:
                continue
            assert 'RJ_SIMD' not in body, match[1]
            assert 'try_execute_' not in body, match[1]
            assert 'amdgpu::execute_v_' not in body, match[1]
            counts[int(count[1])] += 1
    assert counts == {1: 4 * len(UNARY16), 2: 2 * len(BINARY16), 3: 2 * len(TERNARY16)}


@pytest.mark.parametrize('arch', ['rdna3', 'rdna4'])
def test_widening_integer_keeps_full_word_accumulator_and_destination(arch):
    from amdisa.codegen.execute.rj_arith_vector import (
        gen_widening_integer,
        WIDENING_INTEGER_NAMES,
    )

    for name in WIDENING_INTEGER_NAMES:
        mad = 'MAD' in name
        sources = ['a', 'b', 'c'] if mad else ['a', 'b']
        body = gen_widening_integer(arch, name, ['d'], sources, True)
        assert 'rj_arith_' + name.lower() + '_w64' in body
        assert 'RJ_ARITH_ALU_HIGH_C' not in body and 'RJ_ARITH_ALU_HIGH_D' not in body
        assert ('RJ_ARITH_ALU_HIGH_A' in body) == mad
        assert ('RJ_ARITH_ALU_CLAMP' in body) == mad
        assert 'execute_integer16' not in body
        assert gen_widening_integer('cdna4', name, ['d'], sources, True) is None
        assert gen_widening_integer(arch, name, ['d'], sources, False) is None


@pytest.mark.parametrize('arch', ['rdna3', 'rdna4'])
def test_packed_integer_selector_mapping_and_compatibility(arch):
    from amdisa.codegen.execute.rj_arith_vector import (
        PACKED_INTEGER_NAMES,
        PACKED_INTEGER_TERNARY,
        gen_packed_integer,
    )

    for name in PACKED_INTEGER_NAMES:
        count = 3 if name in PACKED_INTEGER_TERNARY else 2
        body = gen_packed_integer(
            arch,
            name,
            ['d'],
            ['a', 'b', 'c'][:count],
            'ENC_VOP3P',
            ('low', 'high'),
            'high_c',
        )
        assert 'rj_arith_' + name.lower() + '_w64' in body
        assert '(high & 1) ? 0u : RJ_ARITH_PK_HI_A_LOW' in body
        assert ('high_c ? 0u : RJ_ARITH_PK_HI_C_LOW' in body) == (count == 3)
        assert 'RJ_ARITH_PK_NEG' not in body
        assert 'arithmetic_flags()' in body


@pytest.mark.parametrize('arch', ['rdna3', 'rdna4'])
def test_integer_dot_signedness_bits_are_not_negation_or_half_selectors(arch):
    from amdisa.codegen.execute.rj_arith_vector import (
        INTEGER_DOT_NAMES,
        gen_integer_dot,
    )

    for name in INTEGER_DOT_NAMES:
        body = gen_integer_dot(
            arch, name, ['d'], ['a', 'b', 'c'], 'ENC_VOP3P', ('low', 'high')
        )
        assert 'rj_arith_' + name.lower() + '_w64' in body
        assert 'RJ_ARITH_DOT_CLAMP' in body
        assert ('RJ_ARITH_DOT_SIGNED_A' in body) == ('_I32_' in name)
        assert ('RJ_ARITH_DOT_SIGNED_B' in body) == ('_I32_' in name)
        assert 'RJ_ARITH_PK_' not in body and 'opsel' not in body
        assert (
            gen_integer_dot(
                'cdna4', name, ['d'], ['a', 'b', 'c'], 'ENC_VOP3P', ('low', 'high')
            )
            is None
        )


@pytest.mark.parametrize('arch', ['rdna3', 'rdna4'])
@pytest.mark.parametrize('name', ['V_CVT_I32_I16', 'V_CVT_U32_U16'])
def test_integer_widening_half_selection_and_ignored_modifiers(arch, name):
    for vop3 in (False, True):
        body = gen_vector32(arch, name, ['dst'], ['src'], vop3)
        assert 'rj_arith_' + name.lower() + '_w64' in body
        assert 'execute_unary32' in body
        assert ('vop3_opsel' in body) == vop3
        assert 'RJ_ARITH_ALU_HIGH_A ? 0xC : 0x3' in body
        assert 'arithmetic_flags()' in body
        for ignored in ('inst_.abs', 'inst_.neg', 'inst_.omod', 'inst_.clamp'):
            assert ignored not in body


@pytest.mark.parametrize('arch', ['rdna3', 'rdna4'])
def test_packed_saturation_reads_word_and_writes_selected_half(arch):
    for vop3 in (False, True):
        body = gen_vector32(arch, 'V_SAT_PK_U8_I16', ['dst'], ['src'], vop3)
        assert 'rj_arith_v_sat_pk_u8_i16_w64' in body
        assert 'execute_unary32' in body
        assert ('vop3_opsel(inst_) & 8' in body) == vop3
        assert '0xF, modifiers & RJ_ARITH_ALU_HIGH_D ? 0xC : 0x3' in body
        assert 'inst_.abs' not in body and 'inst_.neg' not in body


@pytest.mark.parametrize('arch', ['rdna3', 'rdna4'])
def test_qsad_register_layouts_and_no_legacy_bypass(arch):
    from amdisa.codegen.execute.rj_arith_vector import QSAD_NAMES, gen_wide_integer

    root = (
        Path(__file__).resolve().parents[3]
        / 'rocjitsu/src/rocjitsu/isa/arch/amdgpu/generated'
    )
    source = (root / arch / 'vop3_exec.cpp').read_text()
    for name in QSAD_NAMES:
        body = gen_wide_integer(arch, name, ['d'], ['a', 'b', 'c'], True)
        assert 'execute_qsad' in body
        assert 'rj_arith_' + name.lower() + '_w64' in body
        assert 'RJ_ARITH_ALU_CLAMP' in body
        assert 'RJ_ARITH_ALU_ABS' not in body and 'RJ_ARITH_ALU_NEG' not in body
        assert ', ' + str(name != 'V_MQSAD_U32_U8').lower() + ');' in body
        assert 'arithmetic_flags()' in body
        assert gen_wide_integer('cdna4', name, ['d'], ['a', 'b', 'c'], True) is None
        assert gen_wide_integer(arch, name, ['d'], ['a', 'b', 'c'], False) is None
        assert gen_wide_integer(arch, name, ['d'], ['a', 'b'], True) is None
        assert (
            gen_wide_integer(arch, name, ['d'], ['a', 'b', 'c'], True, 'commit') is None
        )
    bodies = re.findall(
        r'void V(?:QsadPkU16U8|MqsadPkU16U8|MqsadU32U8)Vop3::execute_impl\([^)]*\)\s*\{([^}]+)\}',
        source,
    )
    assert len(bodies) == 3
    for body in bodies:
        assert 'execute_qsad' in body
        assert 'try_execute_' not in body and 'amdgpu::execute_v_' not in body
