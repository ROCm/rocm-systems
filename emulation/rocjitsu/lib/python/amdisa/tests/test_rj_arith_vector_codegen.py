# Copyright (c) 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

from rj_arith_headers import public_headers

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


@pytest.mark.parametrize('arch', ['rdna3', 'rdna4'])
@pytest.mark.parametrize('name', sorted(BINARY32))
def test_binary_bindings_have_both_wave_apis_and_clamp(arch, name):
    body = gen_vector32(arch, name, ['dst'], ['a', 'b'], True)
    api = 'rj_arith_' + name.lower()
    header = public_headers()
    for function in (api + '_w32', api + '_w64'):
        assert function in body
        assert re.search(r'\bint ' + function + r'\(', header)
    assert ('inst_.clamp ? RJ_ARITH_ALU_CLAMP : 0u' in body) == (
        name not in UNMODIFIED_MULTIPLIES and name != 'V_MOV_B32'
    )
    assert 'dst, a, b' in body
    assert 'arithmetic_flags()' in body
    assert 'inst_.clamp' not in gen_vector32(arch, name, ['dst'], ['a', 'b'], False)


def test_unmigrated_architectures_and_operand_shapes_keep_existing_path():
    for arch in ('rdna1', 'rdna2', 'rdna3_5', 'cdna4', 'cdna5'):
        assert gen_vector32(arch, 'V_AND_B32', ['dst'], ['a', 'b'], False) is None
    assert gen_vector32('rdna4', 'V_AND_B32', ['dst', 'scc'], ['a', 'b'], False) is None
    assert gen_vector32('rdna4', 'V_ADD_F32', ['dst'], ['a', 'b'], False) is None


@pytest.mark.parametrize(
    'name,count,kind',
    [(name, 1, 'unary') for name in sorted(UNARY32)]
    + [(name, 2, 'binary') for name in sorted(BINARY32)]
    + [(name, 3, 'ternary') for name in sorted(TERNARY32)],
)
def test_single_vgpr_integer_bindings_match_public_signatures(name, count, kind):
    header = public_headers()
    for suffix, mask in (('_w32', 'uint32_t'), ('_w64', 'uint64_t')):
        api = 'rj_arith_' + name.lower() + suffix
        declaration = re.search(r'\bint ' + api + r'\(([^;]+)\);', header)
        assert declaration is not None
        parameters = declaration[1].split(',')
        assert len(parameters) == count + 4
        assert parameters[1].strip().startswith(mask + ' ')
        assert 'uint32_t *const *' in parameters[3]
        for source in parameters[4:]:
            assert 'const uint32_t *const *' in source
    sources = [f'src{i}' for i in range(count)]
    for arch in ('rdna3', 'rdna4'):
        body = gen_vector32(arch, name, ['dst'], sources, True)
        assert f'arith::execute_{kind}32' in body
        assert ('inst_.clamp ? RJ_ARITH_ALU_CLAMP : 0u' in body) == (
            name not in UNMODIFIED_MULTIPLIES and name != 'V_MOV_B32'
        )
        assert ', '.join(['dst'] + sources) in body
        assert gen_vector32(arch, name, ['dst'], sources + ['extra'], True) is None


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
        body = gen_integer_dot(arch, name, ['d'], ['a', 'b', 'c'], 'ENC_VOP3P')
        assert 'rj_arith_' + name.lower() + '_w64' in body
        assert 'RJ_ARITH_DOT_CLAMP' in body
        assert ('RJ_ARITH_DOT_SIGNED_A' in body) == ('_I32_' in name)
        assert ('RJ_ARITH_DOT_SIGNED_B' in body) == ('_I32_' in name)
        assert 'RJ_ARITH_PK_' not in body and 'opsel' not in body
        assert (
            gen_integer_dot('cdna4', name, ['d'], ['a', 'b', 'c'], 'ENC_VOP3P') is None
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
