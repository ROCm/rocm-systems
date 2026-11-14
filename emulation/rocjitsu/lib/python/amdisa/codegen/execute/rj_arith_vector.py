# Copyright (c) 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Architecture-shared rj-arith vector arithmetic bindings."""

from . import ExecuteContext
from .rj_arith_names import wave_api

# CDNA spellings with the same integer arithmetic and zero-input search sentinel.
# Keep these aliases in the simulator binding; no duplicated public arithmetic is needed.
CDNA_INTEGER_ALIASES = {
    'V_MAD_LEGACY_I16': 'V_MAD_I16',
    'V_MAD_LEGACY_U16': 'V_MAD_U16',
    'V_ADD_U16': 'V_ADD_NC_U16',
    'V_SUB_U16': 'V_SUB_NC_U16',
    'V_SUBREV_U16': 'V_SUB_NC_U16',
    'V_ADD_I16': 'V_ADD_NC_I16',
    'V_SUB_I16': 'V_SUB_NC_I16',
    'V_ADD_U32': 'V_ADD_NC_U32',
    'V_SUB_U32': 'V_SUB_NC_U32',
    'V_SUBREV_U32': 'V_SUBREV_NC_U32',
    'V_ADD_I32': 'V_ADD_NC_I32',
    'V_SUB_I32': 'V_SUB_NC_I32',
    'V_FFBH_U32': 'V_CLZ_I32_U32',
    'V_FFBH_I32': 'V_CLS_I32',
    'V_FFBL_B32': 'V_CTZ_I32_B32',
    'V_ADDC_CO_U32': 'V_ADD_CO_CI_U32',
    'V_SUBB_CO_U32': 'V_SUB_CO_CI_U32',
    'V_SUBBREV_CO_U32': 'V_SUBREV_CO_CI_U32',
}
INTEGER_ARCHITECTURES = ('rdna3', 'rdna4', 'cdna3', 'cdna4', 'cdna5')
# Routing and conversions migrate independently of integer arithmetic.
ROUTING_ARCHITECTURES = INTEGER_ARCHITECTURES


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
    'V_BFREV_B32',
    'V_CLS_I32',
    'V_CLZ_I32_U32',
    'V_CTZ_I32_B32',
    'V_MOV_B32',
    'V_NOT_B32',
    'V_PRNG_B32',
}


CDNA5_COMBINED32 = {
    'V_MAD_U32',
    'V_ADD_MIN_I32',
    'V_ADD_MIN_U32',
    'V_ADD_MAX_I32',
    'V_ADD_MAX_U32',
}


TERNARY32 = CDNA5_COMBINED32 | {
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
SATURATE_NAMES = {'V_SAT_PK_U8_I16', 'V_SAT_PK4_I4_I8', 'V_SAT_PK4_U4_U8'}
# Accumulator operands resolve their register bank before reaching the API.
ACCUMULATOR_MOVES = {'V_ACCVGPR_MOV_B32', 'V_ACCVGPR_READ', 'V_ACCVGPR_WRITE'}
UNARY32 |= ACCUMULATOR_MOVES | {'V_SCREEN_PARTITION_4SE_B32'}

CONDITIONAL_NAMES = {'V_CNDMASK_B32', 'V_CNDMASK_B16'}
VECTOR32_NAMES = (
    UNARY32 | BINARY32 | TERNARY32 | WIDEN16_NAMES | SATURATE_NAMES | CONDITIONAL_NAMES
)

# These multiplication APIs do not accept CLAMP. The ISA ignores it here;
# forwarding it would turn an ignored encoding bit into a rj-arith error.
UNMODIFIED_MULTIPLIES = {
    'V_MUL_LO_U32',
    'V_MUL_HI_U32',
    'V_MUL_HI_I32',
    'V_MUL_HI_I32_I24',
    'V_MUL_HI_U32_U24',
}


def gen_vector32(ctx: ExecuteContext) -> str | None:
    arch = ctx.arch_name
    mnemonic = (
        CDNA_INTEGER_ALIASES.get(ctx.inst.name, ctx.inst.name)
        if arch.startswith('cdna')
        else ctx.inst.name
    )
    dst = ctx.dst_ops
    src = ctx.src_ops
    is_vop3 = ctx.is_vop3
    if mnemonic in CDNA5_COMBINED32 and arch != 'cdna5':
        return None
    if mnemonic == 'V_PRNG_B32' and arch not in ('cdna4', 'cdna5'):
        return None
    if arch not in INTEGER_ARCHITECTURES or mnemonic not in VECTOR32_NAMES:
        return None
    if (
        mnemonic in CONDITIONAL_NAMES | {'V_MOV_B32'}
        and arch not in ROUTING_ARCHITECTURES
    ):
        return None
    if mnemonic in WIDEN16_NAMES and arch not in ('rdna3', 'rdna4', 'cdna5'):
        return None
    if mnemonic in ACCUMULATOR_MOVES:
        if arch not in ('cdna3', 'cdna4') or len(dst) != 1 or len(src) != 1:
            return None
        return f'  amdgpu::arith::execute_move32(*this, wf, {dst[0]}, {src[0]});'
    if mnemonic == 'V_MOV_B32' and not is_vop3 and len(dst) == len(src) == 1:
        return f'  amdgpu::arith::execute_move32(*this, wf, {dst[0]}, {src[0]});'
    if mnemonic == 'V_SCREEN_PARTITION_4SE_B32':
        if arch not in ('cdna3', 'cdna4') or len(dst) != 1 or len(src) != 1:
            return None
        # These CDNA targets only execute Wave64. Decoder operands retain DPP/SDWA routing.
        return (
            f'  amdgpu::arith::execute_screen_partition(*this, wf, {dst[0]}, {src[0]});'
        )
    if mnemonic.startswith('V_SAT_PK4_') and arch != 'cdna5':
        return None
    if mnemonic in CONDITIONAL_NAMES:
        return gen_conditional(ctx)
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
        if arch in ('cdna3', 'cdna4'):
            api += '_gfx942'
            sdwa = (
                ''
                if is_vop3
                else (
                    '  if (inst_.src0 == amdgpu::SRC_SDWA) {\n'
                    f'    amdgpu::arith::execute_saturating_pack_sdwa(*this, wf, {wave_api(api, 32)}, {wave_api(api, 64)}, {dst[0]}, {src[0]});\n'
                    '    return;\n'
                    '  }\n'
                )
            )
            modifiers = (
                '(amdgpu::vop3_opsel(inst_) & 8) ? RJ_ARITH_ALU_HIGH_D : 0u'
                if is_vop3
                else '0u'
            )
            return sdwa + (
                f'  const uint64_t modifiers = {modifiers};\n'
                f'  amdgpu::arith::execute_unary32(wf, {wave_api(api, 32)}, {wave_api(api, 64)},\n'
                f'      wf.cu().arithmetic_flags(), modifiers, {dst[0]}, {src[0]}, 0xF, modifiers & RJ_ARITH_ALU_HIGH_D ? 0xC : 0xF);'
            )
        modifiers = (
            '(amdgpu::vop3_opsel(inst_) & 8) ? RJ_ARITH_ALU_HIGH_D : 0u'
            if is_vop3
            else f'((amdgpu::e32_packed_vgpr_operand({dst[0]}) && ({dst[0]}.encoding_value() & 0x80)) ? RJ_ARITH_ALU_HIGH_D : 0u)'
        )
        destination = dst[0] if is_vop3 else 'word_destination'
        preamble = (
            ''
            if is_vop3
            else f'  auto word_destination = amdgpu::e32_word_operand({dst[0]});\n'
        )
        return (
            preamble + f'  const uint64_t modifiers = {modifiers};\n'
            f'  amdgpu::arith::execute_unary32(wf, {wave_api(api, 32)}, {wave_api(api, 64)},\n'
            '      wf.cu().arithmetic_flags(),\n'
            f'      modifiers, {destination}, {src[0]}, 0xF, modifiers & RJ_ARITH_ALU_HIGH_D ? 0xC : 0x3);'
        )
    if mnemonic in WIDEN16_NAMES:
        # VOP1 true16 operands already select a half; VOP3 uses explicit OPSEL.
        modifiers = (
            '(amdgpu::vop3_opsel(inst_) & 1) ? RJ_ARITH_ALU_HIGH_A : 0u'
            if is_vop3
            else f'((amdgpu::e32_packed_vgpr_operand({src[0]}) && ({src[0]}.encoding_value() & 0x80)) ? RJ_ARITH_ALU_HIGH_A : 0u)'
        )
        source = src[0] if is_vop3 else 'word_source'
        preamble = (
            ''
            if is_vop3
            else f'  auto word_source = amdgpu::e32_word_operand({src[0]});\n'
        )
        return (
            preamble + f'  const uint64_t modifiers = {modifiers};\n'
            f'  amdgpu::arith::execute_unary32(wf, {wave_api(api, 32)}, {wave_api(api, 64)},\n'
            '      wf.cu().arithmetic_flags(),\n'
            f'      modifiers, {dst[0]}, {source}, modifiers & RJ_ARITH_ALU_HIGH_A ? 0xC : 0x3);'
        )
    modifiers = (
        'inst_.clamp ? RJ_ARITH_ALU_CLAMP : 0u'
        if is_vop3 and mnemonic not in UNMODIFIED_MULTIPLIES and mnemonic != 'V_MOV_B32'
        else '0u'
    )
    if mnemonic == 'V_MOV_B32' and is_vop3 and arch in ('rdna3', 'rdna4', 'cdna5'):
        modifiers = '((inst_.abs & 1) ? RJ_ARITH_ALU_ABS_A : 0u) | ((inst_.neg & 1) ? RJ_ARITH_ALU_NEG_A : 0u)'
    extra = ''
    if mnemonic in ('V_ALIGNBIT_B32', 'V_ALIGNBYTE_B32') and arch in ('cdna3', 'cdna4'):
        # CDNA selects one of four count bytes; RDNA selects the low/high half.
        extra = ', 0xF, 0xF, ~uint64_t{0}, 0, amdgpu::vop3_opsel(inst_) & 3u'
    elif mnemonic in ('V_ALIGNBIT_B32', 'V_ALIGNBYTE_B32'):
        modifiers = (
            f'({modifiers}) | '
            '((amdgpu::vop3_opsel(inst_) & 4u) ? RJ_ARITH_ALU_HIGH_C : 0u)'
        )
    sdwa = ''
    if arch in ('cdna3', 'cdna4') and not is_vop3 and count <= 2:
        sdwa = (
            f'  if (amdgpu::arith::try_execute_sdwa_integer32(*this, wf, {wave_api(api, 32)}, {wave_api(api, 64)},\n'
            f'      wf.cu().arithmetic_flags(), 0, {dst[0]}, {", ".join(src)})) return;\n'
        )
    return (
        sdwa
        + f'  amdgpu::arith::execute_{kind}32(wf, {wave_api(api, 32)}, {wave_api(api, 64)},\n'
        '      wf.cu().arithmetic_flags(),\n'
        f'      {modifiers}, {dst[0]}, {", ".join(src)}{extra});'
    )


def gen_conditional(ctx: ExecuteContext) -> str | None:
    mnemonic = ctx.inst.name
    dst, src, is_vop3 = ctx.dst_ops, ctx.src_ops, ctx.is_vop3
    if len(dst) != 1 or len(src) != (3 if is_vop3 else 2):
        return None
    half = mnemonic.endswith('B16')
    # A zero execution mask must not observe the scalar condition operand.
    lines = ['  if (!wf.exec()) return;']
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
        f'  amdgpu::arith::execute_conditional<{str(half).lower()}>(*this, wf, {wave_api(api, 32)}, {wave_api(api, 64)},',
        '      wf.cu().arithmetic_flags(),',
        f'      {modifiers}, {dst[0]}, {src[0]}, {src[1]}, {condition});',
    ]
    return '\n'.join(lines)


CARRY32_NAMES = {
    f'V_{op}_CO{carry}_U32' for op in ('ADD', 'SUB', 'SUBREV') for carry in ('', '_CI')
}


def gen_carry32(ctx: ExecuteContext) -> str | None:
    arch, mnemonic = ctx.arch_name, ctx.inst.name
    if arch.startswith('cdna'):
        mnemonic = CDNA_INTEGER_ALIASES.get(mnemonic, mnemonic)
    dst, src = ctx.dst_ops, ctx.src_ops
    is_vop3, result_writer = ctx.is_vop3, ctx.result_writer
    if arch not in INTEGER_ARCHITECTURES or mnemonic not in CARRY32_NAMES:
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
    if with_carry:
        carry = f'wf.exec() ? {carry} : 0u'
    modifiers = 'inst_.clamp ? RJ_ARITH_ALU_CLAMP : 0u' if is_vop3 else '0u'
    api = 'rj_arith_' + mnemonic.lower()
    carry_policy = ', false' if arch.startswith('cdna') else ''
    sdwa = ''
    if arch in ('cdna3', 'cdna4') and not is_vop3:
        sdwa = (
            '  if (inst_.src0 == amdgpu::SRC_SDWA) {\n'
            f'    amdgpu::arith::execute_sdwa_carry32<{str(with_carry).lower()}>(*this, wf, {wave_api(api, 32)}, {wave_api(api, 64)},\n'
            f'        wf.cu().arithmetic_flags(), {modifiers}, {dst[0]}, {src[0]}, {src[1]}, input_carry, {result_writer});\n'
            '    return;\n  }\n'
        )
    return (
        f'  const uint64_t input_carry = {carry};\n'
        + sdwa
        + f'  amdgpu::arith::execute_carry32<{str(with_carry).lower()}{carry_policy}>(wf, {wave_api(api, 32)}, {wave_api(api, 64)},\n'
        '      wf.cu().arithmetic_flags(),\n'
        f'      {modifiers}, {dst[0]}, {src[0]}, {src[1]}, input_carry, {result_writer});'
    )


SHIFT64_NAMES = {'V_LSHLREV_B64', 'V_LSHRREV_B64', 'V_ASHRREV_I64'}
MAD64_NAMES = {'V_MAD_U64_U32', 'V_MAD_I64_I32', 'V_MAD_CO_U64_U32', 'V_MAD_CO_I64_I32'}
QSAD_NAMES = {'V_QSAD_PK_U16_U8', 'V_MQSAD_PK_U16_U8', 'V_MQSAD_U32_U8'}


MAD64_NC_NAMES = {'V_MAD_NC_U64_U32', 'V_MAD_NC_I64_I32'}


BINARY64_NAMES = {
    'V_ADD_NC_U64',
    'V_SUB_NC_U64',
    'V_MUL_U64',
    'V_MIN_U64',
    'V_MAX_U64',
    'V_MIN_I64',
    'V_MAX_I64',
}
SHIFT_ADD64_NAMES = {'V_LSHL_ADD_U64'}
WIDE_INTEGER_NAMES = (
    SHIFT64_NAMES
    | MAD64_NAMES
    | MAD64_NC_NAMES
    | QSAD_NAMES
    | BINARY64_NAMES
    | SHIFT_ADD64_NAMES
)


def gen_wide_integer(ctx: ExecuteContext) -> str | None:
    arch, mnemonic = ctx.arch_name, ctx.inst.name
    dst, src = ctx.dst_ops, ctx.src_ops
    is_vop3, result_writer = ctx.is_vop3, ctx.result_writer
    if arch not in INTEGER_ARCHITECTURES or mnemonic not in WIDE_INTEGER_NAMES:
        return None
    api = 'rj_arith_' + mnemonic.lower()
    modifiers = 'inst_.clamp ? RJ_ARITH_ALU_CLAMP : 0u' if is_vop3 else '0u'
    if mnemonic in BINARY64_NAMES | SHIFT_ADD64_NAMES:
        shift_add = mnemonic in SHIFT_ADD64_NAMES
        if (
            len(dst) != 1
            or len(src) != (3 if shift_add else 2)
            or result_writer is not None
        ):
            return None
        bridge = 'execute_shift_add64' if shift_add else 'execute_binary64'
        return (
            f'  amdgpu::arith::{bridge}(wf, {wave_api(api, 32)}, {wave_api(api, 64)},\n'
            '      wf.cu().arithmetic_flags(),\n'
            f'      {modifiers}, {dst[0]}, {", ".join(src)});'
        )
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
    if mnemonic in MAD64_NC_NAMES:
        if len(dst) != 1 or len(src) != 3 or not is_vop3 or result_writer is not None:
            return None
        return (
            f'  amdgpu::arith::execute_mad64_nc(wf, {wave_api(api, 32)}, {wave_api(api, 64)},\n'
            '      wf.cu().arithmetic_flags(),\n'
            f'      {modifiers}, {dst[0]}, {", ".join(src)});'
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


BYTE_SHIFT_NAMES = {'V_ASHR_PK_I8_I32', 'V_ASHR_PK_U8_I32'}


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


INTEGER16_NAMES = UNARY16 | BINARY16 | TERNARY16 | BYTE_SHIFT_NAMES


def gen_integer16(ctx: ExecuteContext) -> str | None:
    arch = ctx.arch_name
    mnemonic = (
        CDNA_INTEGER_ALIASES.get(ctx.inst.name, ctx.inst.name)
        if arch.startswith('cdna')
        else ctx.inst.name
    )
    dst = ctx.dst_ops
    src = ctx.src_ops
    reverse = ctx.inst.name == 'V_SUBREV_U16' and arch in ('cdna3', 'cdna4')
    if reverse:
        src = list(reversed(src))
    is_vop3 = ctx.is_vop3
    if arch not in INTEGER_ARCHITECTURES or mnemonic not in INTEGER16_NAMES:
        return None
    if mnemonic == 'V_MOV_B16' and arch not in ('rdna3', 'rdna4', 'cdna5'):
        return None
    count = 1 if mnemonic in UNARY16 else 2 if mnemonic in BINARY16 else 3
    if len(dst) != 1 or len(src) != count:
        return None
    # Legacy MAD wraps even when the encoding's CLAMP bit is set.
    clamp = (
        '0u'
        if ctx.inst.name.startswith('V_MAD_LEGACY_')
        else '(inst_.clamp ? RJ_ARITH_ALU_CLAMP : 0u)'
    )
    lines = []
    sign_modifiers = (
        '((inst_.abs & 1) ? RJ_ARITH_ALU_ABS_A : 0u) | ((inst_.neg & 1) ? RJ_ARITH_ALU_NEG_A : 0u) |'
        if mnemonic == 'V_MOV_B16' and is_vop3
        else ''
    )
    full_sources = mnemonic in BYTE_SHIFT_NAMES
    if full_sources:
        lines += [
            '  const uint32_t opsel = amdgpu::vop3_opsel(inst_);',
            '  const uint64_t modifiers = (opsel & 8) ? RJ_ARITH_ALU_HIGH_D : 0u;',
        ]
    elif is_vop3:
        lines += [
            '  const uint32_t opsel = amdgpu::vop3_opsel(inst_);',
            f'  const uint64_t modifiers = {sign_modifiers} {clamp} |',
            '      ((opsel & 8) ? RJ_ARITH_ALU_HIGH_D : 0u) |',
        ]
        for i, selector in enumerate(('A', 'B', 'C')[:count]):
            source_index = 1 - i if reverse else i
            suffix = ';' if i + 1 == count else ' |'
            lines.append(
                f'      ((opsel & {1 << source_index}) ? RJ_ARITH_ALU_HIGH_{selector} : 0u){suffix}'
            )
    elif (
        mnemonic == 'V_MUL_LO_U16'
        and ctx.enc_name in ctx.profile.mul_lo_u16_clamp_encodings
    ):
        lines.append(
            '  const uint64_t modifiers = (inst_.src0 == amdgpu::SRC_SDWA && sdwa_clamp_) ? RJ_ARITH_ALU_CLAMP : 0;'
        )
    else:
        # VOP1 operands already resolve their encoded source/destination halves.
        lines.append('  const uint64_t modifiers = 0;')
    api = 'rj_arith_' + mnemonic.lower()
    if (
        mnemonic == 'V_MUL_LO_U16'
        and ctx.enc_name in ctx.profile.mul_lo_u16_clamp_encodings
    ):
        api += '_gfx942'
    inputs = ', '.join([src[0]] + ['&' + operand for operand in src[1:]])
    # gfx942/gfx950 zero the upper half only when writing the low half.
    shape = f'{count}, true' if arch in ('cdna3', 'cdna4') else str(count)
    if full_sources:
        shape = f'{count}, {str(arch in ("cdna3", "cdna4")).lower()}, true'
    if arch in ('cdna3', 'cdna4') and not is_vop3 and count <= 2:
        lines += [
            f'  if (amdgpu::arith::try_execute_sdwa_integer32(*this, wf, {wave_api(api, 32)}, {wave_api(api, 64)},',
            f'      wf.cu().arithmetic_flags(), modifiers, {dst[0]}, {", ".join(src)})) return;',
        ]
    lines += [
        f'  amdgpu::arith::execute_integer16<{shape}>(wf, {wave_api(api, 32)}, {wave_api(api, 64)},',
        '      wf.cu().arithmetic_flags(),',
        f'      modifiers, {dst[0]}, {inputs});',
    ]
    return '\n'.join(lines)


WIDENING_MAD_NAMES = {'V_MAD_U32_U16', 'V_MAD_I32_I16'}
INTEGER_PACK_NAMES = {'V_CVT_PK_U16_U32', 'V_CVT_PK_I16_I32'}
WIDENING_INTEGER_NAMES = WIDENING_MAD_NAMES | INTEGER_PACK_NAMES


def gen_widening_integer(ctx: ExecuteContext) -> str | None:
    arch = ctx.arch_name
    mnemonic = ctx.inst.name
    dst = ctx.dst_ops
    src = ctx.src_ops
    is_vop3 = ctx.is_vop3
    if arch not in INTEGER_ARCHITECTURES or mnemonic not in WIDENING_INTEGER_NAMES:
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


CDNA5_PACKED_TERNARY = {
    f'V_PK_{op}_{dtype}'
    for op in ('ADD_MIN', 'ADD_MAX', 'MIN3', 'MAX3')
    for dtype in ('I16', 'U16')
}

PACKED_INTEGER_TERNARY = {'V_PK_MAD_I16', 'V_PK_MAD_U16'} | CDNA5_PACKED_TERNARY


PACKED_INTEGER_NAMES = PACKED_INTEGER_BINARY | PACKED_INTEGER_TERNARY


def gen_packed_integer(ctx: ExecuteContext) -> str | None:
    arch = ctx.arch_name
    mnemonic = ctx.inst.name
    dst = ctx.dst_ops
    src = ctx.src_ops
    enc_name = ctx.enc_name
    opsel_exprs = ctx.opsel_exprs
    hi_c_expr = ctx.op_sel_hi_2_expr
    if mnemonic in CDNA5_PACKED_TERNARY and arch != 'cdna5':
        return None
    if arch not in INTEGER_ARCHITECTURES or mnemonic not in PACKED_INTEGER_NAMES:
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
    if mnemonic == 'V_PK_MUL_LO_U16' and ctx.profile.packed_mul_lo_u16_clamp:
        api += '_gfx942'
    kind = 'ternary' if count == 3 else 'binary'
    access = (
        '/*a_bytes=*/0xF, /*b_bytes=*/0xF, /*source_exec_mask=*/~uint64_t{0}'
        if count == 3
        else '/*destination_bytes=*/0xF'
    )
    lines += [
        f'  amdgpu::arith::execute_{kind}32(wf, {wave_api(api, 32)}, {wave_api(api, 64)},',
        '      wf.cu().arithmetic_flags(),',
        f'      modifiers, {dst[0]}, {", ".join(src)}, {access},',
        '      amdgpu::packed_integer_negations(wf.cu().arch(), inst_));',
    ]
    return '\n'.join(lines)


MODERN_INTEGER_DOT_NAMES = {
    'V_DOT4_I32_IU8',
    'V_DOT4_U32_U8',
    'V_DOT8_I32_IU4',
    'V_DOT8_U32_U4',
}

CDNA_INTEGER_DOT_NAMES = {
    'V_DOT2_I32_I16',
    'V_DOT2_U32_U16',
    'V_DOT4_I32_I8',
    'V_DOT4_U32_U8',
    'V_DOT8_I32_I4',
    'V_DOT8_U32_U4',
}
ACCUMULATING_INTEGER_DOT_NAMES = {
    'V_DOT2C_I32_I16',
    'V_DOT4C_I32_I8',
    'V_DOT8C_I32_I4',
}
INTEGER_DOT_NAMES = (
    MODERN_INTEGER_DOT_NAMES | CDNA_INTEGER_DOT_NAMES | ACCUMULATING_INTEGER_DOT_NAMES
)


def gen_integer_dot(ctx: ExecuteContext) -> str | None:
    arch = ctx.arch_name
    mnemonic = ctx.inst.name
    dst = ctx.dst_ops
    src = ctx.src_ops
    enc_name = ctx.enc_name
    opsel_exprs = ctx.opsel_exprs
    if arch in ('cdna3', 'cdna4'):
        names = CDNA_INTEGER_DOT_NAMES | ACCUMULATING_INTEGER_DOT_NAMES
    elif arch in ('rdna3', 'rdna4', 'cdna5'):
        names = MODERN_INTEGER_DOT_NAMES
    else:
        return None
    if mnemonic not in names or len(dst) != 1:
        return None
    if mnemonic in ACCUMULATING_INTEGER_DOT_NAMES:
        if enc_name not in ('ENC_VOP2', 'ENC_VOP3') or len(src) != 2:
            return None
        # The ternary bridge stages old D alongside A/B, including DPP sources.
        # DOTC wraps and has no packed selectors or signedness controls.
        api = 'rj_arith_' + mnemonic.lower().replace('c_i32', '_i32')
        return (
            f'  amdgpu::arith::execute_ternary32(wf, {wave_api(api, 32)}, {wave_api(api, 64)},\n'
            '      wf.cu().arithmetic_flags(),\n'
            f'      0, {dst[0]}, {", ".join(src)}, {dst[0]});'
        )
    if enc_name != 'ENC_VOP3P' or len(src) != 3:
        return None
    low, high = opsel_exprs
    modifiers = (
        '(inst_.clamp ? RJ_ARITH_DOT_CLAMP : 0u)'
        f' | (({low} & 1) ? RJ_ARITH_DOT_LO_A_HIGH : 0u)'
        f' | (({low} & 2) ? RJ_ARITH_DOT_LO_B_HIGH : 0u)'
        f' | (({high} & 1) ? 0u : RJ_ARITH_DOT_HI_A_LOW)'
        f' | (({high} & 2) ? 0u : RJ_ARITH_DOT_HI_B_LOW)'
    )
    if '_I32_IU' in mnemonic:
        modifiers += (
            ' |\n      ((inst_.neg & 1) ? RJ_ARITH_DOT_SIGNED_A : 0u)'
            ' |\n      ((inst_.neg & 2) ? RJ_ARITH_DOT_SIGNED_B : 0u)'
        )
    api = 'rj_arith_' + mnemonic.lower()
    if arch == 'cdna5' and 'DOT4' in mnemonic:
        api += '_gfx1250'
    return (
        f'  const uint64_t modifiers = {modifiers};\n'
        f'  amdgpu::arith::execute_ternary32(wf, {wave_api(api, 32)}, {wave_api(api, 64)},\n'
        '      wf.cu().arithmetic_flags(),\n'
        f'      modifiers, {dst[0]}, {", ".join(src)});'
    )


BITOP3_NAMES = {'V_BITOP3_B16', 'V_BITOP3_B32'}


def gen_bitop3(ctx: ExecuteContext) -> str | None:
    arch, mnemonic = ctx.arch_name, ctx.inst.name
    if arch not in ('cdna4', 'cdna5') or mnemonic not in BITOP3_NAMES:
        return None
    if not ctx.is_vop3 or len(ctx.dst_ops) != 1 or len(ctx.src_ops) != 3:
        return None
    half = mnemonic == 'V_BITOP3_B16'
    lines = [
        '  uint64_t modifiers = (inst_.omod << 6) | (inst_.abs << 3) | inst_.neg;',
        '  modifiers |= inst_.clamp ? RJ_ARITH_ALU_CLAMP : 0u;',
    ]
    if half and arch == 'cdna5':
        lines += [
            '  const uint32_t opsel = amdgpu::vop3_opsel(inst_);',
            '  modifiers |= ((opsel & 1) ? RJ_ARITH_ALU_HIGH_A : 0u) |',
            '               ((opsel & 2) ? RJ_ARITH_ALU_HIGH_B : 0u) |',
            '               ((opsel & 4) ? RJ_ARITH_ALU_HIGH_C : 0u) |',
            '               ((opsel & 8) ? RJ_ARITH_ALU_HIGH_D : 0u);',
        ]
    api = 'rj_arith_' + mnemonic.lower()
    if half:
        bridge = f'execute_integer16<3, {str(arch == "cdna4").lower()}>'
        inputs = ', '.join([ctx.src_ops[0]] + ['&' + src for src in ctx.src_ops[1:]])
    else:
        bridge = 'execute_ternary32'
        inputs = ', '.join(ctx.src_ops)
    lines += [
        f'  amdgpu::arith::{bridge}(wf, {wave_api(api, 32)}, {wave_api(api, 64)},',
        f'      wf.cu().arithmetic_flags(), modifiers, {ctx.dst_ops[0]}, {inputs});',
    ]
    return '\n'.join(lines)
