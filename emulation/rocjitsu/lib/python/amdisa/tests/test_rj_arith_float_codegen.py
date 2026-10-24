# Copyright (c) 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

from rj_arith_headers import public_headers

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


@pytest.mark.parametrize('arch', ['rdna3', 'rdna4'])
@pytest.mark.parametrize('name', sorted(UNARY_F32))
def test_unary_float_bindings_and_policy(arch, name):
    body = gen_float_unary32(arch, name, ['dst'], ['src'], True)
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
    assert 'arithmetic_reports_fp_status()' in body
    assert 'inst_.clamp' not in gen_float_unary32(arch, name, ['dst'], ['src'], False)
    assert gen_float_unary32(arch, name, ['dst'], ['src', 'extra'], False) is None


def test_other_architectures_keep_existing_path():
    for arch in ('rdna1', 'rdna2', 'rdna3_5', 'cdna4', 'cdna5'):
        assert gen_float_unary32(arch, 'V_CEIL_F32', ['dst'], ['src'], False) is None


@pytest.mark.parametrize('arch', ['rdna3', 'rdna4'])
@pytest.mark.parametrize('name', sorted(BINARY_F32))
def test_binary_float_bindings_match_architecture_and_policy(arch, name):
    body = gen_float_binary32(arch, name, ['dst'], ['a', 'b'], True)
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
    assert 'arithmetic_reports_fp_status()' in body
    assert 'inst_.clamp' not in gen_float_binary32(
        arch, name, ['dst'], ['a', 'b'], False
    )
    assert gen_float_binary32(arch, name, ['dst'], ['a'], False) is None
    assert gen_float_binary32('cdna4', name, ['dst'], ['a', 'b'], False) is None


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


@pytest.mark.parametrize('arch', ['rdna3', 'rdna4'])
def test_dot2_operand_transformations_and_accumulator(arch):
    from amdisa.codegen.execute.rj_arith_float import gen_float_dot32

    for bf16 in (False, True):
        name = 'V_DOT2_F32_' + ('BF16' if bf16 else 'F16')
        body = gen_float_dot32(
            arch, name, ['d'], ['a', 'b', 'c'], 'ENC_VOP3P', ('low', 'high')
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
        arch, 'V_DOT2ACC_F32_F16', ['d'], ['a', 'b'], 'ENC_VOP2', ('low', 'high')
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
    assert 'arithmetic_flags()' in body
    assert gen_half_dot('cdna4', name, ['d'], ['a', 'b', 'c'], True) is None


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
def test_cube_reuses_ternary_adapter_and_guest_policies(arch):
    from amdisa.codegen.execute.rj_arith_float import CUBE_NAMES, gen_cube

    for name in CUBE_NAMES:
        body = gen_cube(arch, name, ['dst'], ['x', 'y', 'z'], True)
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
        assert gen_cube('cdna4', name, ['dst'], ['x', 'y', 'z'], True) is None
        assert gen_cube(arch, name, ['dst'], ['x', 'y'], True) is None
        assert gen_cube(arch, name, ['dst'], ['x', 'y', 'z'], False) is None


@pytest.mark.parametrize('arch', ['rdna3', 'rdna4'])
def test_byte_conversion_policy_and_selected_read_byte(arch):
    from amdisa.codegen.execute.rj_arith_float import gen_byte_nibble_conversion

    for index in range(5):
        byte = index % 4
        name = f'V_CVT_F32_UBYTE{index}' if index < 4 else 'V_CVT_OFF_F32_I4'
        for vop3 in (False, True):
            body = gen_byte_nibble_conversion(arch, name, ['dst'], ['src'], vop3)
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
        assert gen_byte_nibble_conversion('cdna4', name, ['dst'], ['src'], True) is None


@pytest.mark.parametrize('packed', [False, True])
def test_fp8_widen_selectors_and_pair_result(packed):
    from amdisa.codegen.execute.rj_arith_float import gen_fp8_widen

    for kind in ('FP8', 'BF8'):
        name = 'V_CVT_' + ('PK_' if packed else '') + 'F32_' + kind
        for vop3 in (False, True):
            body = gen_fp8_widen('rdna4', name, ['dst'], ['src'], vop3)
            assert 'rj_arith_' + name.lower() + '_w64' in body
            assert ('execute_widen32_to_pair' if packed else 'execute_unary32') in body
            assert 'arithmetic_flags()' in body
            assert ('inst_.opsel' in body) == vop3
            assert 'source_bytes' in body
            assert 'arithmetic_reports_fp_status' not in body
            assert 'inst_.clamp' not in body
        assert gen_fp8_widen('rdna3', name, ['dst'], ['src'], True) is None
        assert gen_fp8_widen('cdna5', name, ['dst'], ['src'], True) is None
        assert gen_fp8_widen('rdna4', name, ['dst'], [], True) is None


@pytest.mark.parametrize('stochastic', [False, True])
def test_fp8_narrow_modes_and_partial_destination(stochastic):
    from amdisa.codegen.execute.rj_arith_float import gen_fp8_narrow

    for kind in ('FP8', 'BF8'):
        name = 'V_CVT_' + ('SR_' if stochastic else 'PK_') + kind + '_F32'
        body = gen_fp8_narrow('rdna4', name, ['dst'], ['a', 'b'], True)
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
        assert gen_fp8_narrow('rdna3', name, ['dst'], ['a', 'b'], True) is None
        assert gen_fp8_narrow('cdna5', name, ['dst'], ['a', 'b'], True) is None
        assert gen_fp8_narrow('rdna4', name, ['dst'], ['a', 'b'], False) is None


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
            body = gen_conversion64(arch, name, ['dst'], ['src'], vop3)
            arch_suffix = '_gfx12' if arch == 'rdna4' else ''
            api = 'rj_arith_' + name.lower()
            assert api + '_w64' + arch_suffix in body
            assert 'arithmetic_flags()' in body
            assert (
                'execute_widen32_to_pair'
                if from_integer
                else 'execute_float_conversion'
            ) in body
            assert ('arithmetic_reports_fp_status' in body) != from_integer
            assert ('RJ_ARITH_ALU_ABS_A' in body) == (vop3 and not from_integer)
            assert ('RJ_ARITH_ALU_NEG_A' in body) == (vop3 and not from_integer)
            assert ('inst_.omod' in body) == vop3
            assert ('RJ_ARITH_FP_FLUSH_OUTPUT_DENORMALS' in body) == output.startswith(
                'F'
            )
        assert gen_conversion64('cdna4', name, ['dst'], ['src'], True) is None
        assert gen_conversion64(arch, name, ['dst'], [], True) is None


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
            body = gen_conversion32(arch, name, ['dst'], ['src'], vop3)
            api = 'rj_arith_' + name.lower()
            arch_suffix = '_gfx12' if arch == 'rdna4' and not fixed_round else ''
            assert api + '_w64' + arch_suffix in body
            assert 'float32_flags' in body
            assert 'arithmetic_flags()' in body
            assert ('arithmetic_reports_fp_status' in body) == (
                not to_float and not fixed_round
            )
            assert ('inst_.omod' in body) == (vop3 and not fixed_round)
            assert ('RJ_ARITH_ALU_ABS_A' in body) == (vop3 and not to_float)
            assert ('RJ_ARITH_ALU_NEG_A' in body) == (vop3 and not to_float)
        assert gen_conversion32('cdna4', name, ['dst'], ['src'], True) is None
        assert gen_conversion32(arch, name, ['dst'], [], True) is None


@pytest.mark.parametrize('arch', ['rdna3', 'rdna4'])
def test_conversion16_halves_and_guest_modes(arch):
    from amdisa.codegen.execute.rj_arith_float import (
        CONVERSION16_NAMES,
        gen_conversion16,
    )

    for name in CONVERSION16_NAMES:
        output, source = name.split('_')[-2:]
        for vop3 in (False, True):
            body = gen_conversion16(arch, name, ['dst'], ['src'], vop3)
            arch_suffix = '_gfx12' if arch == 'rdna4' else ''
            api = 'rj_arith_' + name.lower()
            assert api + '_w64' + arch_suffix in body
            assert 'execute_float_conversion' in body
            assert 'arithmetic_flags()' in body
            assert 'arithmetic_reports_fp_status()' in body
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
        assert gen_conversion16('cdna4', name, ['dst'], ['src'], True) is None
        assert gen_conversion16(arch, name, ['dst'], [], True) is None


@pytest.mark.parametrize('arch', ['rdna3', 'rdna4'])
def test_packed_conversion_uses_fixed_rounding_and_optional_state(arch):
    from amdisa.codegen.execute.rj_arith_float import (
        PACKED_CONVERSION_NAMES,
        gen_packed_conversion,
    )

    for name in PACKED_CONVERSION_NAMES:
        for vop3 in (False, True):
            body = gen_packed_conversion(arch, name, ['dst'], ['a', 'b'], vop3)
            arch_suffix = '_gfx12' if arch == 'rdna4' else ''
            api = 'rj_arith_' + name.lower()
            assert api + '_w64' + arch_suffix in body
            assert 'arithmetic_flags()' in body
            assert 'arithmetic_reports_fp_status()' in body
            assert 'execute_float_pack' in body
            assert '/*half_sources=*/false' in body
            assert 'RJ_ARITH_ALU_HIGH' not in body
            assert ('RJ_ARITH_ALU_ABS_B' in body) == vop3
            assert ('RJ_ARITH_ALU_NEG_A' in body) == vop3
            assert ('RJ_ARITH_ALU_CLAMP' in body) == vop3
            assert ('RJ_ARITH_ALU_OMOD_2' in body) == (vop3 and '_RTZ_' in name)
        assert gen_packed_conversion('cdna4', name, ['dst'], ['a', 'b'], True) is None
        assert gen_packed_conversion(arch, name, ['dst'], ['a'], True) is None


@pytest.mark.parametrize('arch', ['rdna3', 'rdna4'])
def test_normalized_conversion_layouts_and_state_contract(arch):
    from amdisa.codegen.execute.rj_arith_float import NORMALIZED_NAMES, gen_normalized

    for name in NORMALIZED_NAMES:
        packed = '_PK_' in name
        half = name.endswith('_F16')
        sources = ['a', 'b'] if packed else ['a']
        for vop3 in (False, True):
            body = gen_normalized(arch, name, ['d'], sources, vop3)
            api = 'rj_arith_' + name.lower()
            arch_suffix = '_gfx12' if packed and half and arch == 'rdna4' else ''
            assert api + '_w64' + arch_suffix in body
            assert 'arithmetic_flags()' in body
            assert ('arithmetic_reports_fp_status' in body) == (packed and half)
            assert ('RJ_ARITH_ALU_HIGH_A' in body) == (half and (vop3 or not packed))
            assert ('RJ_ARITH_ALU_HIGH_B' in body) == (packed and half and vop3)
            assert ('RJ_ARITH_ALU_HIGH_D' in body) == (not packed)
            assert 'RJ_ARITH_ALU_OMOD_2' not in body
            assert ('RJ_ARITH_ALU_CLAMP' in body) == (packed and vop3)
            assert 'fp_round_mode' not in body
        assert gen_normalized('cdna4', name, ['d'], sources, True) is None
        assert gen_normalized(arch, name, ['d'], [], True) is None


@pytest.mark.parametrize('arch', ['rdna3', 'rdna4'])
def test_bit_pack_input_layouts_and_state_policy(arch):
    from amdisa.codegen.execute.rj_arith_float import BIT_PACK_NAMES, gen_bit_pack

    for name in BIT_PACK_NAMES:
        half = name == 'V_PACK_B32_F16'
        sources = ['a', 'b'] if half else ['a', 'b', 'c']
        body = gen_bit_pack(arch, name, ['d'], sources, True)
        arch_suffix = '_gfx12' if arch == 'rdna4' else ''
        api = 'rj_arith_' + name.lower()
        assert api + '_w64' + arch_suffix in body
        assert 'arithmetic_flags()' in body
        assert 'arithmetic_reports_fp_status()' in body
        assert ('execute_float_pack' in body) == half
        assert ('execute_byte_pack' in body) == (not half)
        assert ('RJ_ARITH_ALU_HIGH_B' in body) == half
        assert ('RJ_ARITH_ALU_ABS_B' in body) == half
        assert ('RJ_ARITH_ALU_CLAMP' in body) == (not half)
        assert 'RJ_ARITH_ALU_OMOD' not in body
        assert 'RJ_ARITH_ALU_NEG_C' not in body
        assert 'RJ_ARITH_ALU_HIGH_D' not in body
        assert gen_bit_pack('cdna4', name, ['d'], sources, True) is None
        assert gen_bit_pack(arch, name, ['d'], sources, False) is None
        assert gen_bit_pack(arch, name, ['d'], [], True) is None
