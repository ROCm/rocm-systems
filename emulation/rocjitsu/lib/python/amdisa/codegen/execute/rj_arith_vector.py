# Copyright (c) 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Architecture-shared rj-arith vector arithmetic bindings."""

from .rj_arith_names import wave_api

# These integer APIs use one 32-bit VGPR for each result and source.
# Routed operands already include DPP/SDWA transformations from the decoder.
BINARY32 = {
    'V_AND_B32',
    'V_OR_B32',
    'V_XOR_B32',
    'V_XNOR_B32',
    'V_ADD_NC_U32',
    'V_SUB_NC_U32',
    'V_SUBREV_NC_U32',
    'V_ADD_NC_I32',
    'V_SUB_NC_I32',
    'V_LSHLREV_B32',
    'V_LSHRREV_B32',
    'V_ASHRREV_I32',
    'V_MIN_I32',
    'V_MAX_I32',
    'V_MIN_U32',
    'V_MAX_U32',
    'V_MUL_LO_U32',
    'V_MUL_HI_U32',
    'V_MUL_HI_I32',
    'V_MUL_I32_I24',
    'V_MUL_HI_I32_I24',
    'V_MUL_U32_U24',
    'V_MUL_HI_U32_U24',
    'V_BFM_B32',
    'V_BCNT_U32_B32',
    'V_MBCNT_LO_U32_B32',
    'V_MBCNT_HI_U32_B32',
}


UNARY32 = {
    'V_MOV_B32',
    'V_NOT_B32',
    'V_BFREV_B32',
    'V_CLZ_I32_U32',
    'V_CTZ_I32_B32',
    'V_CLS_I32',
}


TERNARY32 = {
    'V_ADD3_U32',
    'V_LSHL_ADD_U32',
    'V_ADD_LSHL_U32',
    'V_LSHL_OR_B32',
    'V_AND_OR_B32',
    'V_OR3_B32',
    'V_XOR3_B32',
    'V_XAD_U32',
    'V_LERP_U8',
    'V_ALIGNBIT_B32',
    'V_ALIGNBYTE_B32',
    'V_BFE_U32',
    'V_BFE_I32',
    'V_BFI_B32',
    'V_PERM_B32',
    'V_MIN3_I32',
    'V_MIN3_U32',
    'V_MAX3_I32',
    'V_MAX3_U32',
    'V_MED3_I32',
    'V_MED3_U32',
    'V_MINMAX_I32',
    'V_MINMAX_U32',
    'V_MAXMIN_I32',
    'V_MAXMIN_U32',
    'V_MAD_U32_U24',
    'V_MAD_I32_I24',
    'V_SAD_U8',
    'V_SAD_HI_U8',
    'V_SAD_U16',
    'V_SAD_U32',
    'V_MSAD_U8',
}


WIDEN16_NAMES = {'V_CVT_I32_I16', 'V_CVT_U32_U16'}


SATURATE_NAMES = {'V_SAT_PK_U8_I16'}


CONDITIONAL_NAMES = {'V_CNDMASK_B32', 'V_CNDMASK_B16'}


# These multiplication APIs do not accept CLAMP. The ISA ignores it here;
# forwarding it would turn an ignored encoding bit into a rj-arith error.
UNMODIFIED_MULTIPLIES = {
    'V_MUL_LO_U32',
    'V_MUL_HI_U32',
    'V_MUL_HI_I32',
    'V_MUL_HI_I32_I24',
    'V_MUL_HI_U32_U24',
}


def gen_vector32(arch, mnemonic, dst, src, is_vop3):
    if arch not in ('rdna3', 'rdna4') or mnemonic not in VECTOR32_NAMES:
        return None
    if mnemonic in CONDITIONAL_NAMES:
        return gen_conditional(arch, mnemonic, dst, src, is_vop3)
    if mnemonic in UNARY32 | WIDEN16_NAMES | SATURATE_NAMES:
        kind, count = 'unary', 1
    elif mnemonic in BINARY32:
        kind, count = 'binary', 2
    else:
        kind, count = 'ternary', 3
    if len(dst) != 1 or len(src) != count:
        return None
    api = 'rj_arith_' + mnemonic.lower()
    if mnemonic in SATURATE_NAMES:
        modifiers = (
            '(amdgpu::vop3_opsel(inst_) & 8) ? RJ_ARITH_ALU_HIGH_D : 0u'
            if is_vop3
            else '0u'
        )
        return (
            f'  const uint64_t modifiers = {modifiers};\n'
            f'  amdgpu::arith::execute_unary32(wf, {wave_api(api, 32)}, {wave_api(api, 64)},\n'
            '      wf.cu().arithmetic_flags(),\n'
            f'      modifiers, {dst[0]}, {src[0]}, 0xF, modifiers & RJ_ARITH_ALU_HIGH_D ? 0xC : 0x3);'
        )
    if mnemonic in WIDEN16_NAMES:
        # VOP1 true16 operands already select a half; VOP3 uses explicit OPSEL.
        modifiers = (
            '(amdgpu::vop3_opsel(inst_) & 1) ? RJ_ARITH_ALU_HIGH_A : 0u'
            if is_vop3
            else '0u'
        )
        return (
            f'  const uint64_t modifiers = {modifiers};\n'
            f'  amdgpu::arith::execute_unary32(wf, {wave_api(api, 32)}, {wave_api(api, 64)},\n'
            '      wf.cu().arithmetic_flags(),\n'
            f'      modifiers, {dst[0]}, {src[0]}, modifiers & RJ_ARITH_ALU_HIGH_A ? 0xC : 0x3);'
        )
    modifiers = (
        'inst_.clamp ? RJ_ARITH_ALU_CLAMP : 0u'
        if is_vop3 and mnemonic not in UNMODIFIED_MULTIPLIES and mnemonic != 'V_MOV_B32'
        else '0u'
    )
    return (
        f'  amdgpu::arith::execute_{kind}32(wf, {wave_api(api, 32)}, {wave_api(api, 64)},\n'
        '      wf.cu().arithmetic_flags(),\n'
        f'      {modifiers}, {dst[0]}, {", ".join(src)});'
    )


def gen_conditional(arch, mnemonic, dst, src, is_vop3):
    if len(dst) != 1 or len(src) != (3 if is_vop3 else 2):
        return None
    half = mnemonic.endswith('B16')
    # A zero execution mask must not observe the scalar condition operand.
    lines = ['  if (!(wf.exec() & wf.vgpr_write_mask())) return;']
    modifiers = '0'
    if is_vop3:
        lines += [
            '  uint64_t modifiers = ((inst_.abs & 1) ? RJ_ARITH_ALU_ABS_A : 0u) |',
            '      ((inst_.abs & 2) ? RJ_ARITH_ALU_ABS_B : 0u) |',
            '      ((inst_.neg & 1) ? RJ_ARITH_ALU_NEG_A : 0u) |',
            '      ((inst_.neg & 2) ? RJ_ARITH_ALU_NEG_B : 0u);',
        ]
        if half:
            lines += [
                '  const uint32_t opsel = amdgpu::vop3_opsel(inst_);',
                '  modifiers |= ((opsel & 1) ? RJ_ARITH_ALU_HIGH_A : 0u) |',
                '      ((opsel & 2) ? RJ_ARITH_ALU_HIGH_B : 0u) |',
                '      ((opsel & 8) ? RJ_ARITH_ALU_HIGH_D : 0u);',
            ]
        modifiers = 'modifiers'
    condition = (
        f'amdgpu::read_wave_mask_scalar({src[2]}, wf)' if is_vop3 else 'wf.vcc_mask()'
    )
    api = 'rj_arith_' + mnemonic.lower()
    lines += [
        f'  amdgpu::arith::execute_conditional<{str(half).lower()}>(wf, {wave_api(api, 32)}, {wave_api(api, 64)},',
        '      wf.cu().arithmetic_flags(),',
        f'      {modifiers}, {dst[0]}, {src[0]}, {src[1]}, {condition});',
    ]
    return '\n'.join(lines)


CARRY32_NAMES = {
    f'V_{op}_CO{carry}_U32' for op in ('ADD', 'SUB', 'SUBREV') for carry in ('', '_CI')
}


def gen_carry32(arch, mnemonic, dst, src, is_vop3, result_writer=None):
    if arch not in ('rdna3', 'rdna4') or mnemonic not in CARRY32_NAMES:
        return None
    with_carry = '_CI_' in mnemonic
    if len(dst) != (2 if is_vop3 else 1) or len(src) != (
        3 if is_vop3 and with_carry else 2
    ):
        return None
    if result_writer is None:
        write = (
            f'amdgpu::write_wave_mask_scalar({dst[1]}, wf, result);'
            if is_vop3
            else 'wf.set_vcc_mask(result);'
        )
        result_writer = f'[&](uint64_t result) {{ {write} }}'
    carry = (
        (f'amdgpu::read_wave_mask_scalar({src[2]}, wf)' if is_vop3 else 'wf.vcc()')
        if with_carry
        else '0u'
    )
    modifiers = 'inst_.clamp ? RJ_ARITH_ALU_CLAMP : 0u' if is_vop3 else '0u'
    api = 'rj_arith_' + mnemonic.lower()
    return (
        f'  const uint64_t input_carry = {carry};\n'
        f'  amdgpu::arith::execute_carry32<{str(with_carry).lower()}>(wf, {wave_api(api, 32)}, {wave_api(api, 64)},\n'
        '      wf.cu().arithmetic_flags(),\n'
        f'      {modifiers}, {dst[0]}, {src[0]}, {src[1]}, input_carry, {result_writer});'
    )


SHIFT64_NAMES = {'V_LSHLREV_B64', 'V_LSHRREV_B64', 'V_ASHRREV_I64'}


MAD64_NAMES = {'V_MAD_U64_U32', 'V_MAD_I64_I32', 'V_MAD_CO_U64_U32', 'V_MAD_CO_I64_I32'}


QSAD_NAMES = {'V_QSAD_PK_U16_U8', 'V_MQSAD_PK_U16_U8', 'V_MQSAD_U32_U8'}


WIDE_INTEGER_NAMES = SHIFT64_NAMES | MAD64_NAMES | QSAD_NAMES


def gen_wide_integer(arch, mnemonic, dst, src, is_vop3, result_writer=None):
    if arch not in ('rdna3', 'rdna4') or mnemonic not in WIDE_INTEGER_NAMES:
        return None
    api = 'rj_arith_' + mnemonic.lower()
    modifiers = 'inst_.clamp ? RJ_ARITH_ALU_CLAMP : 0u' if is_vop3 else '0u'
    if mnemonic in QSAD_NAMES:
        if not is_vop3 or len(dst) != 1 or len(src) != 3 or result_writer is not None:
            return None
        packed = str(mnemonic != 'V_MQSAD_U32_U8').lower()
        return (
            f'  amdgpu::arith::execute_qsad(wf, {wave_api(api, 32)}, {wave_api(api, 64)},\n'
            '      wf.cu().arithmetic_flags(),\n'
            f'      {modifiers}, {dst[0]}, {", ".join(src)}, {packed});'
        )
    if mnemonic in SHIFT64_NAMES:
        if len(dst) != 1 or len(src) != 2 or result_writer is not None:
            return None
        return (
            f'  amdgpu::arith::execute_shift64(wf, {wave_api(api, 32)}, {wave_api(api, 64)},\n'
            '      wf.cu().arithmetic_flags(),\n'
            f'      {modifiers}, {dst[0]}, {src[0]}, {src[1]});'
        )
    if len(dst) != 2 or len(src) != 3:
        return None
    if result_writer is None:
        result_writer = f'[&](uint64_t result) {{ amdgpu::write_wave_mask_scalar({dst[1]}, wf, result); }}'
    return (
        f'  amdgpu::arith::execute_mad64(wf, {wave_api(api, 32)}, {wave_api(api, 64)},\n'
        '      wf.cu().arithmetic_flags(),\n'
        f'      {modifiers}, {dst[0]}, {", ".join(src)}, {result_writer});'
    )


UNARY16 = {'V_MOV_B16', 'V_NOT_B16'}


BINARY16 = {
    'V_ADD_NC_I16',
    'V_SUB_NC_I16',
    'V_ADD_NC_U16',
    'V_SUB_NC_U16',
    'V_MIN_I16',
    'V_MAX_I16',
    'V_MIN_U16',
    'V_MAX_U16',
    'V_MUL_LO_U16',
    'V_LSHLREV_B16',
    'V_LSHRREV_B16',
    'V_ASHRREV_I16',
    'V_AND_B16',
    'V_OR_B16',
    'V_XOR_B16',
}


TERNARY16 = {
    'V_MAD_U16',
    'V_MAD_I16',
    'V_MIN3_I16',
    'V_MIN3_U16',
    'V_MAX3_I16',
    'V_MAX3_U16',
    'V_MED3_I16',
    'V_MED3_U16',
}


INTEGER16_NAMES = UNARY16 | BINARY16 | TERNARY16


def gen_integer16(arch, mnemonic, dst, src, is_vop3):
    if arch not in ('rdna3', 'rdna4') or mnemonic not in INTEGER16_NAMES:
        return None
    count = 1 if mnemonic in UNARY16 else 2 if mnemonic in BINARY16 else 3
    if len(dst) != 1 or len(src) != count:
        return None
    lines = []
    if is_vop3:
        lines += [
            '  const uint32_t opsel = amdgpu::vop3_opsel(inst_);',
            '  const uint64_t modifiers = (inst_.clamp ? RJ_ARITH_ALU_CLAMP : 0u) |',
            '      ((opsel & 8) ? RJ_ARITH_ALU_HIGH_D : 0u) |',
        ]
        for i, selector in enumerate(('A', 'B', 'C')[:count]):
            suffix = ';' if i + 1 == count else ' |'
            lines.append(
                f'      ((opsel & {1 << i}) ? RJ_ARITH_ALU_HIGH_{selector} : 0u){suffix}'
            )
    else:
        # VOP1 operands already resolve their encoded source/destination halves.
        lines.append('  const uint64_t modifiers = 0;')
    api = 'rj_arith_' + mnemonic.lower()
    inputs = ', '.join([src[0]] + ['&' + operand for operand in src[1:]])
    lines += [
        f'  amdgpu::arith::execute_integer16<{count}>(wf, {wave_api(api, 32)}, {wave_api(api, 64)},',
        '      wf.cu().arithmetic_flags(),',
        f'      modifiers, {dst[0]}, {inputs});',
    ]
    return '\n'.join(lines)


WIDENING_MAD_NAMES = {'V_MAD_U32_U16', 'V_MAD_I32_I16'}


INTEGER_PACK_NAMES = {'V_CVT_PK_U16_U32', 'V_CVT_PK_I16_I32'}


WIDENING_INTEGER_NAMES = WIDENING_MAD_NAMES | INTEGER_PACK_NAMES


def gen_widening_integer(arch, mnemonic, dst, src, is_vop3):
    if arch not in ('rdna3', 'rdna4') or mnemonic not in WIDENING_INTEGER_NAMES:
        return None
    mad = mnemonic in WIDENING_MAD_NAMES
    if not is_vop3 or len(dst) != 1 or len(src) != (3 if mad else 2):
        return None
    api = 'rj_arith_' + mnemonic.lower()
    if not mad:
        # Saturating pack has no numerical modifiers; encoding bits are ignored.
        return (
            f'  amdgpu::arith::execute_binary32(wf, {wave_api(api, 32)}, {wave_api(api, 64)},\n'
            '      wf.cu().arithmetic_flags(),\n'
            f'      0u, {dst[0]}, {", ".join(src)});'
        )
    return (
        '  const uint32_t opsel = amdgpu::vop3_opsel(inst_);\n'
        '  const uint64_t modifiers = (inst_.clamp ? RJ_ARITH_ALU_CLAMP : 0u) |\n'
        '      ((opsel & 1) ? RJ_ARITH_ALU_HIGH_A : 0u) |\n'
        '      ((opsel & 2) ? RJ_ARITH_ALU_HIGH_B : 0u);\n'
        f'  amdgpu::arith::execute_ternary32(wf, {wave_api(api, 32)}, {wave_api(api, 64)},\n'
        '      wf.cu().arithmetic_flags(),\n'
        f'      modifiers, {dst[0]}, {", ".join(src)},\n'
        '      (opsel & 1) ? 0xC : 0x3, (opsel & 2) ? 0xC : 0x3);'
    )


PACKED_INTEGER_BINARY = {
    'V_PK_ADD_I16',
    'V_PK_ADD_U16',
    'V_PK_SUB_I16',
    'V_PK_SUB_U16',
    'V_PK_MIN_I16',
    'V_PK_MIN_U16',
    'V_PK_MAX_I16',
    'V_PK_MAX_U16',
    'V_PK_MUL_LO_U16',
    'V_PK_LSHLREV_B16',
    'V_PK_LSHRREV_B16',
    'V_PK_ASHRREV_I16',
}


PACKED_INTEGER_TERNARY = {'V_PK_MAD_I16', 'V_PK_MAD_U16'}


PACKED_INTEGER_NAMES = PACKED_INTEGER_BINARY | PACKED_INTEGER_TERNARY


def gen_packed_integer(arch, mnemonic, dst, src, enc_name, opsel_exprs, hi_c_expr):
    if arch not in ('rdna3', 'rdna4') or mnemonic not in PACKED_INTEGER_NAMES:
        return None
    count = 3 if mnemonic in PACKED_INTEGER_TERNARY else 2
    if enc_name != 'ENC_VOP3P' or len(dst) != 1 or len(src) != count:
        return None
    low, high = opsel_exprs
    lines = ['  const uint64_t modifiers = (inst_.clamp ? RJ_ARITH_PK_CLAMP : 0u) |']
    terms = []
    for index, source in enumerate('ABC'[:count]):
        terms.append(f'(({low} & {1 << index}) ? RJ_ARITH_PK_LO_{source}_HIGH : 0u)')
        selected = f'({high} & {1 << index})' if index < 2 else hi_c_expr
        terms.append(f'({selected} ? 0u : RJ_ARITH_PK_HI_{source}_LOW)')
    lines.append('      ' + ' |\n      '.join(terms) + ';')
    api = 'rj_arith_' + mnemonic.lower()
    kind = 'ternary' if count == 3 else 'binary'
    lines += [
        f'  amdgpu::arith::execute_{kind}32(wf, {wave_api(api, 32)}, {wave_api(api, 64)},',
        '      wf.cu().arithmetic_flags(),',
        f'      modifiers, {dst[0]}, {", ".join(src)});',
    ]
    return '\n'.join(lines)


INTEGER_DOT_NAMES = {
    'V_DOT4_I32_IU8',
    'V_DOT4_U32_U8',
    'V_DOT8_I32_IU4',
    'V_DOT8_U32_U4',
}


def gen_integer_dot(arch, mnemonic, dst, src, enc_name, opsel_exprs):
    if arch not in ('rdna3', 'rdna4') or mnemonic not in INTEGER_DOT_NAMES:
        return None
    if enc_name != 'ENC_VOP3P' or len(dst) != 1 or len(src) != 3:
        return None
    low, high = opsel_exprs
    modifiers = (
        '(inst_.clamp ? RJ_ARITH_DOT_CLAMP : 0u)'
        f' | (({low} & 1) ? RJ_ARITH_DOT_LO_A_HIGH : 0u)'
        f' | (({low} & 2) ? RJ_ARITH_DOT_LO_B_HIGH : 0u)'
        f' | (({high} & 1) ? 0u : RJ_ARITH_DOT_HI_A_LOW)'
        f' | (({high} & 2) ? 0u : RJ_ARITH_DOT_HI_B_LOW)'
    )
    if '_I32_' in mnemonic:
        modifiers += (
            ' |\n      ((inst_.neg & 1) ? RJ_ARITH_DOT_SIGNED_A : 0u)'
            ' |\n      ((inst_.neg & 2) ? RJ_ARITH_DOT_SIGNED_B : 0u)'
        )
    api = 'rj_arith_' + mnemonic.lower()
    return (
        f'  const uint64_t modifiers = {modifiers};\n'
        f'  amdgpu::arith::execute_ternary32(wf, {wave_api(api, 32)}, {wave_api(api, 64)},\n'
        '      wf.cu().arithmetic_flags(),\n'
        f'      modifiers, {dst[0]}, {", ".join(src)});'
    )


VECTOR32_NAMES = (
    UNARY32 | BINARY32 | TERNARY32 | WIDEN16_NAMES | SATURATE_NAMES | CONDITIONAL_NAMES
)
