# Copyright (c) 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

from pathlib import Path
from types import SimpleNamespace
from amdisa.codegen.execute import ExecuteContext
from amdisa.__main__ import _PROFILES
import re

from rj_arith_headers import public_headers

import pytest

from amdisa.codegen.execute.rj_arith_vector import (
    ACCUMULATOR_MOVES,
    BINARY32,
    CDNA5_COMBINED32,
    UNARY32,
    TERNARY32,
    UNMODIFIED_MULTIPLIES,
    CARRY32_NAMES,
    gen_carry32,
    gen_vector32,
)


def context(
    arch,
    name,
    dst,
    src,
    is_vop3,
    result_writer=None,
    *,
    enc_name='',
    opsel_exprs=('', ''),
    op_sel_hi_2_expr='',
):
    return ExecuteContext(
        inst=SimpleNamespace(name=name),
        sem=None,
        profile=_PROFILES[arch.replace("rdna3_5", "rdna3.5")](),
        arch_name=arch,
        dst_ops=dst,
        src_ops=src,
        enc_name=enc_name,
        opsel_exprs=opsel_exprs,
        op_sel_hi_2_expr=op_sel_hi_2_expr,
        is_vop3=is_vop3,
        has_abs=is_vop3,
        result_writer=result_writer,
    )


@pytest.mark.parametrize('arch', ['rdna3', 'rdna4'])
@pytest.mark.parametrize('name', sorted(BINARY32))
def test_binary_bindings_have_both_wave_apis_and_clamp(arch, name):
    body = gen_vector32(context(arch, name, ['dst'], ['a', 'b'], True))
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
    assert 'inst_.clamp' not in gen_vector32(
        context(arch, name, ['dst'], ['a', 'b'], False)
    )


def test_unmigrated_architectures_and_operand_shapes_keep_existing_path():
    for arch in ('rdna1', 'rdna2', 'rdna3_5', 'cdna1', 'cdna2'):
        assert (
            gen_vector32(context(arch, 'V_AND_B32', ['dst'], ['a', 'b'], False)) is None
        )
    assert (
        gen_vector32(context('rdna4', 'V_AND_B32', ['dst', 'scc'], ['a', 'b'], False))
        is None
    )
    assert (
        gen_vector32(context('rdna4', 'V_ADD_F32', ['dst'], ['a', 'b'], False)) is None
    )


@pytest.mark.parametrize(
    'name,count,kind',
    [
        (name, 1, 'unary')
        for name in sorted(UNARY32 - ACCUMULATOR_MOVES - {'V_SCREEN_PARTITION_4SE_B32'})
    ]
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
    architectures = (
        ('cdna5',)
        if name in CDNA5_COMBINED32
        else ('cdna4', 'cdna5') if name == 'V_PRNG_B32' else ('rdna3', 'rdna4')
    )
    for arch in architectures:
        body = gen_vector32(context(arch, name, ['dst'], sources, True))
        assert f'arith::execute_{kind}32' in body
        assert ('inst_.clamp ? RJ_ARITH_ALU_CLAMP : 0u' in body) == (
            name not in UNMODIFIED_MULTIPLIES and name != 'V_MOV_B32'
        )
        assert ', '.join(['dst'] + sources) in body
        assert (
            gen_vector32(context(arch, name, ['dst'], sources + ['extra'], True))
            is None
        )


@pytest.mark.parametrize('arch', ['rdna3', 'rdna4', 'cdna3', 'cdna4', 'cdna5'])
def test_generated_rj_arith_bodies_cannot_be_bypassed_by_legacy_simd(arch):
    root = (
        Path(__file__).resolve().parents[3]
        / 'rocjitsu/src/rocjitsu/isa/arch/amdgpu/generated'
        / arch
    )
    checked = {'unary': 0, 'binary': 0, 'ternary': 0, 'carry': 0}
    for path in root.glob('*exec*.cpp'):
        source = path.read_text()
        for match in re.finditer(
            r'void (\w+)::execute(?:_modifier)?_impl\([^)]*\)\s*\{', source
        ):
            end, depth = match.end(), 1
            while depth:
                depth += (source[end] == '{') - (source[end] == '}')
                end += 1
            body = source[match.end() : end - 1]
            kind = re.search(
                r'arith::execute_(unary|binary|ternary|carry|conditional)(?:32|<)', body
            )
            # Identify the instruction independently of the dispatch in its body:
            # restoring a legacy implementation must not remove it from this test.
            from amdisa.codegen.execute.rj_arith_vector import VECTOR32_NAMES

            expected = {
                ''.join(part.title() for part in name.split('_')): name
                for name in VECTOR32_NAMES | CARRY32_NAMES
                # Routing and conversions migrate on separate CDNA branches.
                if not (
                    arch.startswith('cdna')
                    and name
                    in {
                        'V_MOV_B32',
                        'V_CNDMASK_B32',
                        'V_CNDMASK_B16',
                        'V_CVT_I32_I16',
                        'V_CVT_U32_U16',
                        'V_SAT_PK_U8_I16',
                    }
                )
            }
            instruction = re.sub(r'Vop[123](?:SdstEnc)?$', '', match[1])
            if instruction not in expected:
                continue
            if expected[instruction] == 'V_ACCVGPR_MOV_B32' or (
                expected[instruction] == 'V_MOV_B32' and match[1].endswith('Vop1')
            ):
                # Moves use a bit-preserving adapter rather than numeric source acquisition.
                assert 'arith::execute_move32(' in body
                assert 'amdgpu::execute_v_' not in body
                assert 'try_execute_' not in body
                continue
            assert kind is not None, (path.name, match[1])
            mnemonic = expected[instruction]
            api = 'rj_arith_' + mnemonic.lower()
            for wave in ('w32', 'w64'):
                assert api + '_' + wave in body, (path.name, match[1], wave)
            assert 'RJ_SIMD' not in body, match[1]
            # SDWA placement still invokes rj-arith; reject legacy SIMD probes.
            assert 'try_execute_' not in body.replace(
                'arith::try_execute_sdwa_integer32', ''
            ), match[1]
            assert 'amdgpu::execute_v_' not in body, match[1]
            if kind[1] in checked:
                checked[kind[1]] += 1
    # Other instruction families add users of these adapters as they migrate.
    assert all(count > 0 for count in checked.values())


@pytest.mark.parametrize('arch', ['rdna3', 'rdna4', 'cdna3', 'cdna4', 'cdna5'])
@pytest.mark.parametrize('name', sorted(CARRY32_NAMES))
@pytest.mark.parametrize('vop3', [False, True])
def test_carry_bindings_keep_decoder_commit_and_snapshot_input(arch, name, vop3):
    with_carry = '_CI_' in name
    dst = ['vdst', 'sdst'] if vop3 else ['vdst']
    src = ['a', 'b', 'cin'] if vop3 and with_carry else ['a', 'b']
    body = gen_carry32(context(arch, name, dst, src, vop3, 'commit_result'))
    carry_policy = ', false' if arch.startswith('cdna') else ''
    assert f'execute_carry32<{str(with_carry).lower()}{carry_policy}>' in body
    assert 'rj_arith_' + name.lower() + '_w64' in body
    assert 'input_carry, commit_result' in body
    assert ('inst_.clamp' in body) == vop3
    if with_carry:
        assert ('read_wave_mask_scalar(cin, wf)' if vop3 else 'wf.vcc()') in body
    else:
        assert 'input_carry = 0u' in body
    assert gen_carry32(context('cdna2', name, dst, src, vop3)) is None


@pytest.mark.parametrize('arch', ['rdna3', 'rdna4', 'cdna3', 'cdna4', 'cdna5'])
def test_wide_integer_bindings_and_generated_routing(arch):
    from amdisa.codegen.execute.rj_arith_vector import (
        SHIFT64_NAMES,
        MAD64_NAMES,
        gen_wide_integer,
    )

    for name in SHIFT64_NAMES:
        body = gen_wide_integer(context(arch, name, ['d'], ['a', 'b'], True))
        assert 'execute_shift64(wf, rj_arith_' + name.lower() in body
        assert 'rj_arith_' + name.lower() + '_w64' in body
        assert 'inst_.clamp' in body
        assert (
            gen_wide_integer(context(arch, name, ['d'], ['a', 'b'], True, 'commit'))
            is None
        )
    for name in MAD64_NAMES:
        body = gen_wide_integer(
            context(arch, name, ['d', 'sdst'], ['a', 'b', 'c'], True, 'commit')
        )
        assert 'execute_mad64(wf, rj_arith_' + name.lower() in body
        assert 'rj_arith_' + name.lower() + '_w64' in body
        assert 'd, a, b, c, commit' in body
        assert (
            gen_wide_integer(
                context('cdna2', name, ['d', 'sdst'], ['a', 'b', 'c'], True)
            )
            is None
        )
    root = (
        Path(__file__).resolve().parents[3]
        / 'rocjitsu/src/rocjitsu/isa/arch/amdgpu/generated'
        / arch
    )
    checked = {'shift': 0, 'mad': 0}
    for path in root.glob('*exec*.cpp'):
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
            # Identify the instruction independently of the dispatch in its body:
            # restoring a legacy implementation must not remove it from this test.
            from amdisa.codegen.execute.rj_arith_vector import VECTOR32_NAMES

            expected = {
                ''.join(part.title() for part in name.split('_')): name
                for name in SHIFT64_NAMES | MAD64_NAMES
            }
            instruction = re.sub(r'Vop[123](?:SdstEnc)?$', '', match[1])
            if instruction not in expected:
                continue
            assert kind is not None, (path.name, match[1])
            mnemonic = expected[instruction]
            api = 'rj_arith_' + mnemonic.lower()
            for wave in ('w32', 'w64'):
                assert api + '_' + wave in body, (path.name, match[1], wave)
            assert 'RJ_SIMD' not in body, match[1]
            # SDWA placement still invokes rj-arith; reject legacy SIMD probes.
            assert 'try_execute_' not in body.replace(
                'arith::try_execute_sdwa_integer32', ''
            ), match[1]
            assert 'amdgpu::execute_v_' not in body, match[1]
            if kind[1] in checked:
                checked[kind[1]] += 1
    assert checked == {
        'shift': 8 if arch == 'cdna5' else 4 if arch == 'rdna4' else 3,
        'mad': 4 if arch == 'cdna5' else 2,
    }


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
            body = gen_integer16(context(arch, name, ['d'], sources, True))
            assert (
                f'execute_integer16<{count}>(wf, rj_arith_{name.lower()}_w32,' in body
            )
            assert f'rj_arith_{name.lower()}_w64' in body
            assert 'RJ_ARITH_ALU_CLAMP' in body and 'RJ_ARITH_ALU_HIGH_D' in body
            for index, selector in enumerate('ABC'):
                assert (f'RJ_ARITH_ALU_HIGH_{selector}' in body) == (index < count)
            assert 'RJ_ARITH_DPP' not in body
            assert gen_integer16(context('cdna2', name, ['d'], sources, True)) is None
            assert (
                gen_integer16(context(arch, name, ['d'], sources + ['extra'], True))
                is None
            )
    body = gen_integer16(context(arch, 'V_NOT_B16', ['d'], ['a'], False))
    assert 'modifiers = 0' in body
    assert 'inst_.clamp' not in body and 'opsel' not in body
    root = (
        Path(__file__).resolve().parents[3]
        / 'rocjitsu/src/rocjitsu/isa/arch/amdgpu/generated'
        / arch
    )
    counts = {1: 0, 2: 0, 3: 0}
    for path in root.glob('*exec*.cpp'):
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
            # SDWA placement still invokes rj-arith; reject legacy SIMD probes.
            assert 'try_execute_' not in body.replace(
                'arith::try_execute_sdwa_integer32', ''
            ), match[1]
            assert 'amdgpu::execute_v_' not in body, match[1]
            counts[int(count[1])] += 1
    assert counts == {1: 8, 2: 2 * len(BINARY16), 3: 2 * len(TERNARY16)}


@pytest.mark.parametrize('arch', ['rdna3', 'rdna4'])
def test_widening_integer_keeps_full_word_accumulator_and_destination(arch):
    from amdisa.codegen.execute.rj_arith_vector import (
        gen_widening_integer,
        WIDENING_INTEGER_NAMES,
    )

    for name in WIDENING_INTEGER_NAMES:
        mad = 'MAD' in name
        sources = ['a', 'b', 'c'] if mad else ['a', 'b']
        body = gen_widening_integer(context(arch, name, ['d'], sources, True))
        assert 'rj_arith_' + name.lower() + '_w64' in body
        assert 'RJ_ARITH_ALU_HIGH_C' not in body and 'RJ_ARITH_ALU_HIGH_D' not in body
        assert ('RJ_ARITH_ALU_HIGH_A' in body) == mad
        assert ('RJ_ARITH_ALU_CLAMP' in body) == mad
        assert 'execute_integer16' not in body
        body = gen_widening_integer(context('cdna4', name, ['d'], sources, True))
        assert body is not None
        assert (
            gen_widening_integer(context('cdna2', name, ['d'], sources, True)) is None
        )
        assert gen_widening_integer(context(arch, name, ['d'], sources, False)) is None


@pytest.mark.parametrize('arch', ['rdna3', 'rdna4'])
def test_packed_integer_selector_mapping_and_compatibility(arch):
    from amdisa.codegen.execute.rj_arith_vector import (
        PACKED_INTEGER_NAMES,
        CDNA5_PACKED_TERNARY,
        PACKED_INTEGER_TERNARY,
        gen_packed_integer,
    )

    for name in PACKED_INTEGER_NAMES:
        count = 3 if name in PACKED_INTEGER_TERNARY else 2
        body = gen_packed_integer(
            context(
                arch,
                name,
                ['d'],
                ['a', 'b', 'c'][:count],
                False,
                enc_name='ENC_VOP3P',
                opsel_exprs=('low', 'high'),
                op_sel_hi_2_expr='high_c',
            )
        )
        if name in CDNA5_PACKED_TERNARY:
            assert body is None
            continue
        assert 'rj_arith_' + name.lower() + '_w64' in body
        assert '(high & 1) ? 0u : RJ_ARITH_PK_HI_A_LOW' in body
        assert ('high_c ? 0u : RJ_ARITH_PK_HI_C_LOW' in body) == (count == 3)
        assert 'packed_integer_negations(wf.cu().arch(), inst_)' in body
        assert 'RJ_ARITH_PK_NEG' not in body
        assert 'arithmetic_flags()' in body


@pytest.mark.parametrize('arch', ['rdna3', 'rdna4'])
def test_integer_dot_signedness_bits_are_not_negation_or_half_selectors(arch):
    from amdisa.codegen.execute.rj_arith_vector import (
        MODERN_INTEGER_DOT_NAMES,
        gen_integer_dot,
    )

    for name in MODERN_INTEGER_DOT_NAMES:
        body = gen_integer_dot(
            context(
                arch,
                name,
                ['d'],
                ['a', 'b', 'c'],
                False,
                enc_name='ENC_VOP3P',
                opsel_exprs=('low', 'high'),
            )
        )
        assert 'rj_arith_' + name.lower() + '_w64' in body
        assert 'RJ_ARITH_DOT_CLAMP' in body
        assert ('RJ_ARITH_DOT_SIGNED_A' in body) == ('_I32_' in name)
        assert ('RJ_ARITH_DOT_SIGNED_B' in body) == ('_I32_' in name)
        assert 'RJ_ARITH_PK_' not in body and 'opsel' not in body
        assert (
            gen_integer_dot(
                context(
                    'cdna2',
                    name,
                    ['d'],
                    ['a', 'b', 'c'],
                    False,
                    enc_name='ENC_VOP3P',
                    opsel_exprs=('low', 'high'),
                )
            )
            is None
        )


@pytest.mark.parametrize('arch', ['rdna3', 'rdna4'])
@pytest.mark.parametrize('name', ['V_CVT_I32_I16', 'V_CVT_U32_U16'])
def test_integer_widening_half_selection_and_ignored_modifiers(arch, name):
    for vop3 in (False, True):
        body = gen_vector32(context(arch, name, ['dst'], ['src'], vop3))
        assert 'rj_arith_' + name.lower() + '_w64' in body
        assert 'execute_unary32' in body
        assert ('vop3_opsel' in body) == vop3
        assert 'RJ_ARITH_ALU_HIGH_A ? 0xC : 0x3' in body
        assert 'arithmetic_flags()' in body
        for ignored in ('inst_.abs', 'inst_.neg', 'inst_.omod', 'inst_.clamp'):
            assert ignored not in body


@pytest.mark.parametrize('arch', ['rdna3', 'rdna4', 'cdna5'])
def test_packed_saturation_reads_word_and_writes_selected_half(arch):
    for vop3 in (False, True):
        body = gen_vector32(context(arch, 'V_SAT_PK_U8_I16', ['dst'], ['src'], vop3))
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
        body = gen_wide_integer(context(arch, name, ['d'], ['a', 'b', 'c'], True))
        assert 'execute_qsad' in body
        assert 'rj_arith_' + name.lower() + '_w64' in body
        assert 'RJ_ARITH_ALU_CLAMP' in body
        assert 'RJ_ARITH_ALU_ABS' not in body and 'RJ_ARITH_ALU_NEG' not in body
        assert ', ' + str(name != 'V_MQSAD_U32_U8').lower() + ');' in body
        assert 'arithmetic_flags()' in body
        assert (
            gen_wide_integer(context('cdna2', name, ['d'], ['a', 'b', 'c'], True))
            is None
        )
        assert (
            gen_wide_integer(context(arch, name, ['d'], ['a', 'b', 'c'], False)) is None
        )
        assert gen_wide_integer(context(arch, name, ['d'], ['a', 'b'], True)) is None
        assert (
            gen_wide_integer(
                context(arch, name, ['d'], ['a', 'b', 'c'], True, 'commit')
            )
            is None
        )
    bodies = re.findall(
        r'void V(?:QsadPkU16U8|MqsadPkU16U8|MqsadU32U8)Vop3::execute_impl\([^)]*\)\s*\{([^}]+)\}',
        source,
    )
    assert len(bodies) == 3
    for body in bodies:
        assert 'execute_qsad' in body
        assert 'try_execute_' not in body and 'amdgpu::execute_v_' not in body


@pytest.mark.parametrize('arch', ['cdna3', 'cdna4'])
def test_cdna_saturation_writes_full_word_and_retains_sdwa(arch):
    for vop3 in (False, True):
        body = gen_vector32(context(arch, 'V_SAT_PK_U8_I16', ['dst'], ['src'], vop3))
        assert 'rj_arith_v_sat_pk_u8_i16_w64_gfx942' in body
        assert ('vop3_opsel(inst_) & 8' in body) == vop3
        assert ('execute_saturating_pack_sdwa' in body) == (not vop3)
        assert 'arithmetic_flags(), modifiers, dst, src' in body
        for ignored in ('inst_.abs', 'inst_.neg', 'inst_.omod', 'inst_.clamp'):
            assert ignored not in body


@pytest.mark.parametrize(
    'name', ['V_CVT_I32_I16', 'V_CVT_U32_U16', 'V_CVT_PK_I16_I32', 'V_CVT_PK_U16_U32']
)
def test_gfx1250_integer_conversion_bindings(name):
    from amdisa.codegen.execute.rj_arith_vector import gen_widening_integer

    packed = '_PK_' in name
    generator = gen_widening_integer if packed else gen_vector32
    src = ['a', 'b'] if packed else ['a']
    body = generator(context('cdna5', name, ['d'], src, True))
    assert 'rj_arith_' + name.lower() + '_w32' in body
    assert ('vop3_opsel' in body) == (not packed)
    for ignored in ('inst_.abs', 'inst_.neg', 'inst_.omod', 'inst_.clamp'):
        assert ignored not in body
    generated = (
        Path(__file__).resolve().parents[3]
        / 'rocjitsu/src/rocjitsu/isa/arch/amdgpu/generated/cdna5/vop3_exec_cvt.cpp'
    ).read_text()
    assert 'rj_arith_' + name.lower() + '_w32' in generated
    for arch in ('cdna3', 'cdna4'):
        assert (generator(context(arch, name, ['d'], src, True)) is not None) == packed
    assert (generator(context('cdna5', name, ['d'], src, False)) is None) == packed
    assert generator(context('cdna5', name, [], src, True)) is None


@pytest.mark.parametrize('name', ['V_SAT_PK4_I4_I8', 'V_SAT_PK4_U4_U8'])
@pytest.mark.parametrize('vop3', [False, True])
def test_gfx1250_nibble_pack_placement(name, vop3):
    body = gen_vector32(context('cdna5', name, ['dst'], ['src'], vop3))
    assert 'rj_arith_' + name.lower() + '_w32' in body
    assert '0xF, modifiers & RJ_ARITH_ALU_HIGH_D ? 0xC : 0x3' in body
    if vop3:
        assert '(amdgpu::vop3_opsel(inst_) & 8) ? RJ_ARITH_ALU_HIGH_D : 0u' in body
    else:
        assert 'e32_word_operand(dst)' in body
        assert 'dst.encoding_value() & 0x80' in body
    for arch in ('rdna3', 'rdna4', 'cdna3', 'cdna4'):
        assert gen_vector32(context(arch, name, ['dst'], ['src'], vop3)) is None


@pytest.mark.parametrize('arch', ['cdna3', 'cdna4', 'cdna5'])
@pytest.mark.parametrize('name', ['V_CVT_I32_I16', 'V_CVT_U32_U16', 'V_SAT_PK_U8_I16'])
def test_routing_migration_preserves_integer_conversions(arch, name):
    for vop3 in (False, True):
        body = gen_vector32(context(arch, name, ['dst'], ['src'], vop3))
        if name == 'V_SAT_PK_U8_I16' or arch == 'cdna5':
            assert 'rj_arith_' + name.lower() + '_w32' in body
        else:
            assert body is None


@pytest.mark.parametrize('arch', ['rdna3', 'rdna4', 'cdna3', 'cdna4', 'cdna5'])
def test_vop3_move_source_modifiers_follow_architecture(arch):
    body = gen_vector32(context(arch, 'V_MOV_B32', ['dst'], ['a'], True))
    for modifier in ('RJ_ARITH_ALU_ABS_A', 'RJ_ARITH_ALU_NEG_A'):
        assert (modifier in body) == (arch.startswith('rdna') or arch == 'cdna5')


@pytest.mark.parametrize('arch', ['cdna3', 'cdna4', 'cdna5'])
def test_cdna_integer_aliases_preserve_arithmetic_and_carry_snapshot(arch):
    from amdisa.codegen.execute.rj_arith_vector import CDNA_INTEGER_ALIASES

    for name, api in CDNA_INTEGER_ALIASES.items():
        if '_CO_' in name:
            body = gen_carry32(
                context(arch, name, ['d', 'sdst'], ['a', 'b', 'cin'], True, 'commit')
            )
            assert 'read_wave_mask_scalar(cin, wf)' in body
            assert 'input_carry, commit' in body
        elif name.endswith(('U16', 'I16')):
            from amdisa.codegen.execute.rj_arith_vector import gen_integer16

            src = ['a', 'b', 'c'] if name.startswith('V_MAD_') else ['a', 'b']
            body = gen_integer16(context(arch, name, ['d'], src, True))
            if name.startswith('V_MAD_LEGACY_'):
                assert 'RJ_ARITH_ALU_CLAMP' not in body
                assert 'RJ_ARITH_ALU_HIGH_C' in body
            if name == 'V_SUBREV_U16' and arch != 'cdna5':
                assert 'modifiers, d, b, &a' in body
        else:
            src = ['a'] if name.startswith('V_FFB') else ['a', 'b']
            body = gen_vector32(context(arch, name, ['d'], src, True))
        assert 'rj_arith_' + api.lower() + '_w64' in body
        assert 'wf.cu().arithmetic_flags()' in body


@pytest.mark.parametrize('arch', ['cdna3', 'cdna4', 'cdna5'])
def test_cdna_halfword_destination_policy_and_packed_dispatch(arch):
    from amdisa.codegen.execute.rj_arith_vector import (
        gen_integer16,
        gen_packed_integer,
        gen_widening_integer,
    )

    body = gen_integer16(context(arch, 'V_MIN_U16', ['d'], ['a', 'b'], True))
    shape = '2' if arch == 'cdna5' else '2, true'
    assert f'execute_integer16<{shape}>' in body
    assert 'RJ_ARITH_ALU_HIGH_D' in body
    assert 'RJ_ARITH_ALU_HIGH_A' in body and 'RJ_ARITH_ALU_HIGH_B' in body
    body = gen_packed_integer(
        context(
            arch,
            'V_PK_ADD_U16',
            ['d'],
            ['a', 'b'],
            False,
            enc_name='ENC_VOP3P',
            opsel_exprs=('inst_.op_sel', 'inst_.op_sel_hi'),
        )
    )
    assert 'rj_arith_v_pk_add_u16_w64' in body
    body = gen_widening_integer(
        context(arch, 'V_MAD_U32_U16', ['d'], ['a', 'b', 'c'], True)
    )
    assert 'rj_arith_v_mad_u32_u16_w64' in body


@pytest.mark.parametrize('arch', ['cdna4', 'cdna5'])
def test_byte_shift_reads_full_words_and_selects_only_destination(arch):
    from amdisa.codegen.execute.rj_arith_vector import gen_integer16

    for name in ('V_ASHR_PK_I8_I32', 'V_ASHR_PK_U8_I32'):
        body = gen_integer16(context(arch, name, ['d'], ['a', 'b', 'c'], True))
        zero_high = str(arch == 'cdna4').lower()
        assert f'execute_integer16<3, {zero_high}, true>' in body
        assert 'RJ_ARITH_ALU_HIGH_D' in body
        assert 'RJ_ARITH_ALU_HIGH_A' not in body
        assert 'RJ_ARITH_ALU_CLAMP' not in body


def test_noncarry_mad64_has_no_scalar_result_callback():
    from amdisa.codegen.execute.rj_arith_vector import gen_wide_integer

    for name in ('V_MAD_NC_U64_U32', 'V_MAD_NC_I64_I32'):
        body = gen_wide_integer(context('cdna5', name, ['d'], ['a', 'b', 'c'], True))
        assert 'execute_mad64_nc' in body
        assert f'rj_arith_{name.lower()}_w32' in body
        assert 'RJ_ARITH_ALU_CLAMP' in body
        assert 'read_wave_mask_scalar' not in body
        assert 'write_wave_mask_scalar' not in body
        assert (
            gen_wide_integer(
                context('cdna5', name, ['d'], ['a', 'b', 'c'], True, 'commit')
            )
            is None
        )


@pytest.mark.parametrize('arch', ['cdna3', 'cdna4', 'cdna5'])
def test_wide_binary_and_shift_add_bindings(arch):
    from amdisa.codegen.execute.rj_arith_vector import gen_wide_integer, BINARY64_NAMES

    for name in BINARY64_NAMES | {'V_LSHL_ADD_U64'}:
        shift_add = name == 'V_LSHL_ADD_U64'
        sources = ['a', 'b', 'c'] if shift_add else ['a', 'b']
        body = gen_wide_integer(context(arch, name, ['d'], sources, True))
        assert ('execute_shift_add64' if shift_add else 'execute_binary64') in body
        assert f'rj_arith_{name.lower()}_w64' in body
        assert 'RJ_ARITH_ALU_CLAMP' in body
        assert 'write_wave_mask_scalar' not in body


@pytest.mark.parametrize('arch', ['cdna4', 'cdna5'])
def test_cdna_prng_reuses_unary_bridge(arch):
    body = gen_vector32(context(arch, 'V_PRNG_B32', ['dst'], ['src'], True))
    assert 'execute_unary32' in body
    assert 'rj_arith_v_prng_b32_w32' in body
    assert 'rj_arith_v_prng_b32_w64' in body
    assert gen_vector32(context('cdna3', 'V_PRNG_B32', ['dst'], ['src'], True)) is None


@pytest.mark.parametrize('arch', ['cdna4', 'cdna5'])
@pytest.mark.parametrize('bits', [16, 32])
def test_bitop3_truth_table_and_destination_contract(arch, bits):
    from amdisa.codegen.execute.rj_arith_vector import gen_bitop3

    body = gen_bitop3(
        context(arch, f'V_BITOP3_B{bits}', ['dst'], ['a', 'b', 'c'], True)
    )
    assert '(inst_.omod << 6) | (inst_.abs << 3) | inst_.neg' in body
    assert f'rj_arith_v_bitop3_b{bits}_w32' in body
    assert f'rj_arith_v_bitop3_b{bits}_w64' in body
    assert ('RJ_ARITH_ALU_HIGH_A' in body) == (arch == 'cdna5' and bits == 16)
    if bits == 16:
        assert f'execute_integer16<3, {str(arch == "cdna4").lower()}>' in body
    else:
        assert 'execute_ternary32' in body


@pytest.mark.parametrize(
    'name',
    ['V_MAD_U32', 'V_ADD_MIN_I32', 'V_ADD_MIN_U32', 'V_ADD_MAX_I32', 'V_ADD_MAX_U32'],
)
def test_cdna5_combined_integer_bindings(name):
    body = gen_vector32(context('cdna5', name, ['dst'], ['a', 'b', 'c'], True))
    assert 'execute_ternary32' in body
    assert f'rj_arith_{name.lower()}_w32' in body
    assert f'rj_arith_{name.lower()}_w64' in body
    for arch in ('cdna3', 'cdna4'):
        assert gen_vector32(context(arch, name, ['dst'], ['a', 'b', 'c'], True)) is None


@pytest.mark.parametrize('op', ['ADD_MIN', 'ADD_MAX', 'MIN3', 'MAX3'])
@pytest.mark.parametrize('dtype', ['I16', 'U16'])
def test_cdna5_packed_ternary_bindings(op, dtype):
    from amdisa.codegen.execute.rj_arith_vector import gen_packed_integer

    name = f'V_PK_{op}_{dtype}'
    body = gen_packed_integer(
        context(
            'cdna5',
            name,
            ['dst'],
            ['a', 'b', 'c'],
            False,
            enc_name='ENC_VOP3P',
            opsel_exprs=('inst_.opsel', 'inst_.opsel_hi'),
            op_sel_hi_2_expr='inst_.opsel_hi_2',
        )
    )
    assert 'execute_ternary32' in body
    assert f'rj_arith_{name.lower()}_w32' in body
    assert f'rj_arith_{name.lower()}_w64' in body
    assert 'inst_.opsel_hi_2' in body
    assert 'RJ_ARITH_PK_CLAMP' in body
    for arch in ('cdna3', 'cdna4'):
        assert (
            gen_packed_integer(
                context(
                    arch, name, ['dst'], ['a', 'b', 'c'], False, enc_name='ENC_VOP3P'
                )
            )
            is None
        )


@pytest.mark.parametrize('arch', ['cdna3', 'cdna4', 'cdna5'])
def test_integer_migration_preserves_routing_boundary_and_conversions(arch):
    from amdisa.codegen.execute.rj_arith_vector import (
        gen_integer16,
        gen_widening_integer,
    )

    for name, sources in [
        ('V_MOV_B32', ['a']),
        ('V_CNDMASK_B32', ['a', 'b', 'condition']),
        ('V_CVT_I32_I16', ['a']),
        ('V_CVT_U32_U16', ['a']),
        ('V_SAT_PK_U8_I16', ['a']),
    ]:
        body = gen_vector32(context(arch, name, ['d'], sources, True))
        deferred = name in ('V_CVT_I32_I16', 'V_CVT_U32_U16') and arch != 'cdna5'
        assert (body is None) == deferred
    assert (gen_integer16(context(arch, 'V_MOV_B16', ['d'], ['a'], True)) is None) == (
        arch != 'cdna5'
    )
    for name in ['V_CVT_PK_I16_I32', 'V_CVT_PK_U16_U32']:
        assert (
            gen_widening_integer(context(arch, name, ['d'], ['a', 'b'], True))
            is not None
        )


@pytest.mark.parametrize('arch', ['rdna3', 'rdna4', 'cdna3', 'cdna4', 'cdna5'])
def test_integer_sdwa_destination_placement(arch):
    for name, sources in [('V_NOT_B32', ['a']), ('V_OR_B32', ['a', 'b'])]:
        for vop3 in (False, True):
            body = gen_vector32(context(arch, name, ['d'], sources, vop3))
            assert ('try_execute_sdwa_integer32' in body) == (
                arch in ('cdna3', 'cdna4') and not vop3
            )


@pytest.mark.parametrize('arch', ['cdna3', 'cdna4', 'cdna5'])
def test_unsigned_halfword_multiply_uses_architectural_clamp_policy(arch):
    from amdisa.codegen.execute.rj_arith_vector import gen_integer16, gen_packed_integer

    unpacked = gen_integer16(
        context(arch, 'V_MUL_LO_U16', ['d'], ['a', 'b'], True, enc_name='ENC_VOP3')
    )
    packed = gen_packed_integer(
        context(
            arch,
            'V_PK_MUL_LO_U16',
            ['d'],
            ['a', 'b'],
            False,
            enc_name='ENC_VOP3P',
            opsel_exprs=('inst_.op_sel', 'inst_.op_sel_hi'),
        )
    )
    for body in [unpacked, packed]:
        assert body is not None
        assert ('_gfx942' in body) == (arch == 'cdna3')


def test_gfx942_half_multiply_forwards_sdwa_clamp():
    from amdisa.codegen.execute.rj_arith_vector import gen_integer16

    body = gen_integer16(
        context('cdna3', 'V_MUL_LO_U16', ['d'], ['a', 'b'], False, enc_name='ENC_VOP2')
    )
    assert 'inst_.src0 == amdgpu::SRC_SDWA && sdwa_clamp_' in body
    assert 'rj_arith_v_mul_lo_u16_w64_gfx942' in body
