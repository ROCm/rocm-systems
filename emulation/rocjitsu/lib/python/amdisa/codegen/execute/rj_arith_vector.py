# Copyright (c) 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Architecture-shared rj-arith vector arithmetic bindings."""

from . import ExecuteContext
from .rj_arith_names import wave_api

# These integer APIs use one 32-bit VGPR for each result and source.
# Routed operands already include DPP/SDWA transformations from the decoder.
BINARY32 = set()


UNARY32 = {'V_MOV_B32'}

CONDITIONAL_NAMES = {'V_CNDMASK_B32', 'V_CNDMASK_B16'}


TERNARY32 = set()


WIDEN16_NAMES = {'V_CVT_I32_I16', 'V_CVT_U32_U16'}


SATURATE_NAMES = {'V_SAT_PK_U8_I16'}


# These multiplication APIs do not accept CLAMP. The ISA ignores it here;
# forwarding it would turn an ignored encoding bit into a rj-arith error.
UNMODIFIED_MULTIPLIES = set()


def gen_vector32(ctx: ExecuteContext) -> str | None:
    arch = ctx.arch_name
    mnemonic = ctx.inst.name
    dst = ctx.dst_ops
    src = ctx.src_ops
    is_vop3 = ctx.is_vop3
    if arch not in ('rdna3', 'rdna4') or mnemonic not in VECTOR32_NAMES:
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
    if mnemonic == 'V_MOV_B32' and is_vop3:
        modifiers = '((inst_.abs & 1) ? RJ_ARITH_ALU_ABS_A : 0u) | ((inst_.neg & 1) ? RJ_ARITH_ALU_NEG_A : 0u)'
    return (
        f'  amdgpu::arith::execute_{kind}32(wf, {wave_api(api, 32)}, {wave_api(api, 64)},\n'
        '      wf.cu().arithmetic_flags(),\n'
        f'      {modifiers}, {dst[0]}, {", ".join(src)});'
    )


WIDENING_MAD_NAMES = set()


INTEGER_PACK_NAMES = {'V_CVT_PK_U16_U32', 'V_CVT_PK_I16_I32'}


WIDENING_INTEGER_NAMES = WIDENING_MAD_NAMES | INTEGER_PACK_NAMES


def gen_widening_integer(ctx: ExecuteContext) -> str | None:
    arch = ctx.arch_name
    mnemonic = ctx.inst.name
    dst = ctx.dst_ops
    src = ctx.src_ops
    is_vop3 = ctx.is_vop3
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
        f'  amdgpu::arith::execute_conditional<{str(half).lower()}>(wf, {wave_api(api, 32)}, {wave_api(api, 64)},',
        '      wf.cu().arithmetic_flags(),',
        f'      {modifiers}, {dst[0]}, {src[0]}, {src[1]}, {condition});',
    ]
    return '\n'.join(lines)


UNARY16 = {'V_MOV_B16'}


BINARY16 = set()


TERNARY16 = set()


INTEGER16_NAMES = UNARY16 | BINARY16 | TERNARY16


def gen_integer16(ctx: ExecuteContext) -> str | None:
    arch = ctx.arch_name
    mnemonic = ctx.inst.name
    dst = ctx.dst_ops
    src = ctx.src_ops
    is_vop3 = ctx.is_vop3
    if arch not in ('rdna3', 'rdna4') or mnemonic not in INTEGER16_NAMES:
        return None
    count = 1 if mnemonic in UNARY16 else 2 if mnemonic in BINARY16 else 3
    if len(dst) != 1 or len(src) != count:
        return None
    lines = []
    sign_modifiers = (
        '((inst_.abs & 1) ? RJ_ARITH_ALU_ABS_A : 0u) | ((inst_.neg & 1) ? RJ_ARITH_ALU_NEG_A : 0u) |'
        if mnemonic == 'V_MOV_B16' and is_vop3
        else ''
    )
    if is_vop3:
        lines += [
            '  const uint32_t opsel = amdgpu::vop3_opsel(inst_);',
            f'  const uint64_t modifiers = {sign_modifiers} (inst_.clamp ? RJ_ARITH_ALU_CLAMP : 0u) |',
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


INTEGER_DOT_NAMES = {
    'V_DOT4_I32_IU8',
    'V_DOT4_U32_U8',
    'V_DOT8_I32_IU4',
    'V_DOT8_U32_U4',
}


def gen_integer_dot(ctx: ExecuteContext) -> str | None:
    arch = ctx.arch_name
    mnemonic = ctx.inst.name
    dst = ctx.dst_ops
    src = ctx.src_ops
    enc_name = ctx.enc_name
    opsel_exprs = ctx.opsel_exprs
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


VECTOR32_NAMES = UNARY32 | CONDITIONAL_NAMES | WIDEN16_NAMES | SATURATE_NAMES
