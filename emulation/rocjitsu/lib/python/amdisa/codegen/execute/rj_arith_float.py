# Copyright (c) 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""rj-arith floating arithmetic bindings with explicit guest-mode and state policy."""

from . import ExecuteContext
from .rj_arith_names import wave_api

# RCP_IFLAG and SQRT carry their legacy exception policies into rj-arith;
# other entries here have no legacy classifier.
UNARY_F32 = {
    'V_TRUNC_F32',
    'V_CEIL_F32',
    'V_RNDNE_F32',
    'V_FLOOR_F32',
    'V_FRACT_F32',
    'V_RCP_F32',
    'V_RCP_IFLAG_F32',
    'V_SQRT_F32',
    'V_RSQ_F32',
    'V_EXP_F32',
    'V_LOG_F32',
    'V_SIN_F32',
    'V_COS_F32',
    'V_FREXP_EXP_I32_F32',
    'V_FREXP_MANT_F32',
}


def gen_float_unary32(ctx: ExecuteContext) -> str | None:
    arch = ctx.arch_name
    mnemonic = ctx.inst.name
    dst = ctx.dst_ops
    src = ctx.src_ops
    is_vop3 = ctx.is_vop3
    if arch not in ('rdna3', 'rdna4') or mnemonic not in UNARY_F32:
        return None
    if len(dst) != 1 or len(src) != 1:
        return None
    api = 'rj_arith_' + mnemonic.lower() + ('_gfx12' if arch == 'rdna4' else '')
    lines = []
    modifiers = '0'
    if is_vop3:
        lines += [
            '  const uint64_t modifiers = ((inst_.abs & 1) ? RJ_ARITH_ALU_ABS_A : 0u) |',
            '      ((inst_.neg & 1) ? RJ_ARITH_ALU_NEG_A : 0u) |',
        ]
        if mnemonic.startswith('V_FREXP_EXP_'):
            # Integer results ignore floating-point output modifiers.
            lines[-1] = lines[-1].removesuffix(' |') + ';'
        else:
            lines += [
                '      (uint64_t(inst_.omod) * RJ_ARITH_ALU_OMOD_2) |',
                '      (inst_.clamp ? RJ_ARITH_ALU_CLAMP : 0u);',
            ]
        modifiers = 'modifiers'
    guest_rounding = mnemonic.split('_')[1] in {
        'TRUNC',
        'CEIL',
        'FLOOR',
        'RNDNE',
        'FRACT',
    }
    lines += [
        f'  amdgpu::arith::execute_float_unary32(wf, {wave_api(api, 32)}, {wave_api(api, 64)},',
        f'      wf.cu().arithmetic_flags(), {modifiers},',
        f'      {dst[0]}, {src[0]}, {str(arch == "rdna3").lower()}, /*guest_rounding=*/{str(guest_rounding).lower()});',
    ]
    return '\n'.join(lines)


# FP64 unary operations use the shared guest-control adapter. Fixed-rounding
# cores and native SFU approximations still apply guest rounding to OMOD.
UNARY_F64 = {
    'V_TRUNC_F64',
    'V_CEIL_F64',
    'V_RNDNE_F64',
    'V_FLOOR_F64',
    'V_FRACT_F64',
    'V_FREXP_MANT_F64',
    'V_FREXP_EXP_I32_F64',
    'V_RCP_F64',
    'V_RSQ_F64',
    'V_SQRT_F64',
}


def gen_float_unary64(ctx: ExecuteContext) -> str | None:
    arch = ctx.arch_name
    mnemonic = ctx.inst.name
    dst = ctx.dst_ops
    src = ctx.src_ops
    is_vop3 = ctx.is_vop3
    if arch not in ('rdna3', 'rdna4') or mnemonic not in UNARY_F64:
        return None
    if len(dst) != 1 or len(src) != 1:
        return None
    api = 'rj_arith_' + mnemonic.lower() + ('_gfx12' if arch == 'rdna4' else '')
    lines = []
    modifiers = '0'
    if is_vop3:
        lines += [
            '  const uint64_t modifiers = ((inst_.abs & 1) ? RJ_ARITH_ALU_ABS_A : 0u) |',
            '      ((inst_.neg & 1) ? RJ_ARITH_ALU_NEG_A : 0u) |',
        ]
        if mnemonic.startswith('V_FREXP_EXP_'):
            # Integer results ignore floating-point output modifiers.
            lines[-1] = lines[-1].removesuffix(' |') + ';'
        else:
            lines += [
                '      (uint64_t(inst_.omod) * RJ_ARITH_ALU_OMOD_2) |',
                '      (inst_.clamp ? RJ_ARITH_ALU_CLAMP : 0u);',
            ]
        modifiers = 'modifiers'
    integer_result = (
        ', /*integer_result=*/true' if mnemonic == 'V_FREXP_EXP_I32_F64' else ''
    )
    lines += [
        f'  amdgpu::arith::execute_float_unary64(wf, {wave_api(api, 32)}, {wave_api(api, 64)},',
        f'      wf.cu().arithmetic_flags(), {modifiers},',
        f'      {dst[0]}, {src[0]}, {str(arch == "rdna3").lower()}{integer_result});',
    ]
    return '\n'.join(lines)


SELECTION_F64 = {
    'rdna3': {'V_MIN_F64', 'V_MAX_F64'},
    'rdna4': {'V_MIN_NUM_F64', 'V_MAX_NUM_F64', 'V_MINIMUM_F64', 'V_MAXIMUM_F64'},
}


BINARY_F64 = {
    arch: names | {'V_ADD_F64', 'V_MUL_F64'} for arch, names in SELECTION_F64.items()
}


def gen_float_binary64(ctx: ExecuteContext) -> str | None:
    arch = ctx.arch_name
    mnemonic = ctx.inst.name
    dst = ctx.dst_ops
    src = ctx.src_ops
    is_vop3 = ctx.is_vop3
    if mnemonic not in BINARY_F64.get(arch, set()):
        return None
    if len(dst) != 1 or len(src) != 2:
        return None
    api = 'rj_arith_' + mnemonic.lower()
    if arch == 'rdna4' and mnemonic in {'V_ADD_F64', 'V_MUL_F64'}:
        api += '_gfx12'
    lines = (
        [
            '  const uint64_t modifiers = ((inst_.abs & 1) ? RJ_ARITH_ALU_ABS_A : 0u) |',
            '      ((inst_.abs & 2) ? RJ_ARITH_ALU_ABS_B : 0u) |',
            '      ((inst_.neg & 1) ? RJ_ARITH_ALU_NEG_A : 0u) |',
            '      ((inst_.neg & 2) ? RJ_ARITH_ALU_NEG_B : 0u) |',
            '      (uint64_t(inst_.omod) * RJ_ARITH_ALU_OMOD_2) |',
            '      (inst_.clamp ? RJ_ARITH_ALU_CLAMP : 0u);',
        ]
        if is_vop3
        else ['  const uint64_t modifiers = 0;']
    )
    lines += [
        f'  amdgpu::arith::execute_float_binary64(wf, {wave_api(api, 32)}, {wave_api(api, 64)},',
        '      wf.cu().arithmetic_flags(), modifiers,',
        f'      {dst[0]}, {src[0]}, {src[1]}, {str(arch == "rdna3").lower()});',
    ]
    return '\n'.join(lines)


class RjArithBody(str):
    """An emitted body whose arithmetic and SIMD dispatch belong to rj-arith."""


def gen_float_fma64(ctx: ExecuteContext) -> str | None:
    arch = ctx.arch_name
    mnemonic = ctx.inst.name
    dst = ctx.dst_ops
    src = ctx.src_ops
    is_vop3 = ctx.is_vop3
    if arch not in ('rdna3', 'rdna4') or mnemonic != 'V_FMA_F64' or not is_vop3:
        return None
    if len(dst) != 1 or len(src) != 3:
        return None
    api = 'rj_arith_v_fma_f64' + ('_gfx12' if arch == 'rdna4' else '')
    return '\n'.join(
        [
            '  const uint64_t modifiers = ((inst_.abs & 1) ? RJ_ARITH_ALU_ABS_A : 0u) |',
            '      ((inst_.abs & 2) ? RJ_ARITH_ALU_ABS_B : 0u) |',
            '      ((inst_.abs & 4) ? RJ_ARITH_ALU_ABS_C : 0u) |',
            '      ((inst_.neg & 1) ? RJ_ARITH_ALU_NEG_A : 0u) |',
            '      ((inst_.neg & 2) ? RJ_ARITH_ALU_NEG_B : 0u) |',
            '      ((inst_.neg & 4) ? RJ_ARITH_ALU_NEG_C : 0u) |',
            '      (uint64_t(inst_.omod) * RJ_ARITH_ALU_OMOD_2) |',
            '      (inst_.clamp ? RJ_ARITH_ALU_CLAMP : 0u);',
            f'  amdgpu::arith::execute_float_ternary64(wf, {wave_api(api, 32)}, {wave_api(api, 64)},',
            '      wf.cu().arithmetic_flags(), modifiers,',
            f'      {dst[0]}, {src[0]}, {src[1]}, {src[2]}, {str(arch == "rdna3").lower()});',
        ]
    )


def gen_float64_word(ctx: ExecuteContext) -> str | None:
    arch = ctx.arch_name
    mnemonic = ctx.inst.name
    dst = ctx.dst_ops
    src = ctx.src_ops
    is_vop3 = ctx.is_vop3
    if arch not in ('rdna3', 'rdna4') or not is_vop3:
        return None
    if mnemonic not in {'V_LDEXP_F64', 'V_TRIG_PREOP_F64'}:
        return None
    if len(dst) != 1 or len(src) != 2:
        return None
    api = 'rj_arith_' + mnemonic.lower() + ('_gfx12' if arch == 'rdna4' else '')
    lines = [
        '  const uint64_t modifiers = ((inst_.abs & 1) ? RJ_ARITH_ALU_ABS_A : 0u) |',
        '      ((inst_.neg & 1) ? RJ_ARITH_ALU_NEG_A : 0u) |',
        '      (uint64_t(inst_.omod) * RJ_ARITH_ALU_OMOD_2) |',
        '      (inst_.clamp ? RJ_ARITH_ALU_CLAMP : 0u);',
        f'  amdgpu::arith::execute_float64_word(wf, {wave_api(api, 32)}, {wave_api(api, 64)}, wf.cu().arithmetic_flags(), modifiers,',
        f'      {dst[0]}, {src[0]}, {src[1]}, {str(arch == "rdna3").lower()}, /*ldexp=*/{str(mnemonic == "V_LDEXP_F64").lower()});',
    ]
    return '\n'.join(lines)


# MUL carries its inherited nearest-even exception classifier into rj-arith.
BINARY_F32 = {
    'V_LDEXP_F32',
    'V_ADD_F32',
    'V_MUL_F32',
    'V_SUB_F32',
    'V_SUBREV_F32',
    'V_MUL_DX9_ZERO_F32',
    'V_MIN_F32',
    'V_MAX_F32',
    'V_MIN_NUM_F32',
    'V_MAX_NUM_F32',
    'V_MINIMUM_F32',
    'V_MAXIMUM_F32',
}


FMA_F32 = {
    'V_FMA_F32',
    'V_FMAC_F32',
    'V_FMA_DX9_ZERO_F32',
    'V_FMAC_DX9_ZERO_F32',
    'V_FMAAK_F32',
    'V_FMAMK_F32',
}


MIXED_FMA_NAMES = {'V_FMA_MIX_F32', 'V_FMA_MIXLO_F16', 'V_FMA_MIXHI_F16'}


PACKED_FMA16_NAMES = {'V_PK_FMA_F16', 'V_PK_FMAC_F16'}


PACKED_BINARY16_NAMES = {
    'V_PK_ADD_F16',
    'V_PK_MUL_F16',
    'V_PK_MIN_F16',
    'V_PK_MAX_F16',
    'V_PK_MIN_NUM_F16',
    'V_PK_MAX_NUM_F16',
    'V_PK_MINIMUM_F16',
    'V_PK_MAXIMUM_F16',
}


DOT2_F32_NAMES = {'V_DOT2_F32_F16', 'V_DOT2_F32_BF16', 'V_DOT2ACC_F32_F16'}


DOT4_F32_NAMES = {
    'V_DOT4_F32_FP8_FP8',
    'V_DOT4_F32_FP8_BF8',
    'V_DOT4_F32_BF8_FP8',
    'V_DOT4_F32_BF8_BF8',
}


BINARY_ARITHMETIC_F16 = {
    'V_ADD_F16',
    'V_SUB_F16',
    'V_SUBREV_F16',
    'V_MUL_F16',
    'V_LDEXP_F16',
}


BINARY_SELECTION_F16 = {
    'V_MIN_F16',
    'V_MAX_F16',
    'V_MIN_NUM_F16',
    'V_MAX_NUM_F16',
    'V_MINIMUM_F16',
    'V_MAXIMUM_F16',
}


UNARY_F16 = {
    'V_TRUNC_F16',
    'V_CEIL_F16',
    'V_RNDNE_F16',
    'V_FLOOR_F16',
    'V_RCP_F16',
    'V_RSQ_F16',
    'V_SQRT_F16',
    'V_EXP_F16',
    'V_LOG_F16',
    'V_FRACT_F16',
    'V_FREXP_MANT_F16',
    'V_FREXP_EXP_I16_F16',
    'V_SIN_F16',
    'V_COS_F16',
}


FMA_F16 = {'V_FMA_F16', 'V_FMAC_F16', 'V_FMAAK_F16', 'V_FMAMK_F16'}


BINARY_F16 = BINARY_ARITHMETIC_F16 | BINARY_SELECTION_F16


HALF_DOT_NAMES = {'V_DOT2_F16_F16', 'V_DOT2_BF16_BF16'}


TERNARY_SELECTION_RDNA3 = {
    f'V_{op}_F{width}'
    for op in ('MIN3', 'MAX3', 'MINMAX', 'MAXMIN', 'MED3')
    for width in (16, 32)
}


TERNARY_SELECTION_RDNA4 = {
    f'V_{op}_F{width}'
    for op in (
        'MIN3_NUM',
        'MAX3_NUM',
        'MINMAX_NUM',
        'MAXMIN_NUM',
        'MINIMUM3',
        'MAXIMUM3',
        'MINIMUMMAXIMUM',
        'MAXIMUMMINIMUM',
        'MED3_NUM',
    )
    for width in (16, 32)
}


def gen_float_binary32(ctx: ExecuteContext) -> str | None:
    arch = ctx.arch_name
    mnemonic = ctx.inst.name
    dst = ctx.dst_ops
    src = ctx.src_ops
    is_vop3 = ctx.is_vop3
    if arch not in ('rdna3', 'rdna4') or mnemonic not in BINARY_F32:
        return None
    if len(dst) != 1 or len(src) != 2:
        return None
    if mnemonic in ('V_MIN_F32', 'V_MAX_F32') and arch != 'rdna3':
        return None
    if (
        mnemonic in ('V_MIN_NUM_F32', 'V_MAX_NUM_F32', 'V_MINIMUM_F32', 'V_MAXIMUM_F32')
        and arch != 'rdna4'
    ):
        return None
    arithmetic = mnemonic in (
        'V_ADD_F32',
        'V_MUL_F32',
        'V_SUB_F32',
        'V_SUBREV_F32',
        'V_MUL_DX9_ZERO_F32',
        'V_LDEXP_F32',
    )
    api = (
        'rj_arith_'
        + mnemonic.lower()
        + ('_gfx12' if arch == 'rdna4' and arithmetic else '')
    )
    lines = []
    modifiers = '0'
    if is_vop3:
        lines += [
            '  const uint64_t modifiers = ((inst_.abs & 1) ? RJ_ARITH_ALU_ABS_A : 0u) |',
            *(
                ['      ((inst_.abs & 2) ? RJ_ARITH_ALU_ABS_B : 0u) |']
                if mnemonic != 'V_LDEXP_F32'
                else []
            ),
            '      ((inst_.neg & 1) ? RJ_ARITH_ALU_NEG_A : 0u) |',
            *(
                ['      ((inst_.neg & 2) ? RJ_ARITH_ALU_NEG_B : 0u) |']
                if mnemonic != 'V_LDEXP_F32'
                else []
            ),
            '      (uint64_t(inst_.omod) * RJ_ARITH_ALU_OMOD_2) |',
            '      (inst_.clamp ? RJ_ARITH_ALU_CLAMP : 0u);',
        ]
        modifiers = 'modifiers'
    controls = f'/*rdna3_dx10_clamp=*/{str(arch == "rdna3").lower()}, /*compatibility_mul=*/{str(mnemonic == "V_MUL_F32").lower()}'
    lines += [
        f'  amdgpu::arith::execute_float_binary32(wf, {wave_api(api, 32)}, {wave_api(api, 64)},',
        f'      wf.cu().arithmetic_flags(), {modifiers},',
        f'      {dst[0]}, {src[0]}, {src[1]},',
        f'      {controls});',
    ]
    return '\n'.join(lines)


def gen_float_fma32(ctx: ExecuteContext) -> str | None:
    arch = ctx.arch_name
    mnemonic = ctx.inst.name
    dst = ctx.dst_ops
    src = ctx.src_ops
    is_vop3 = ctx.is_vop3
    if arch not in ('rdna3', 'rdna4') or mnemonic not in FMA_F32:
        return None
    if arch == 'rdna4' and mnemonic == 'V_FMAC_DX9_ZERO_F32':
        return None
    accumulate = mnemonic in ('V_FMAC_F32', 'V_FMAC_DX9_ZERO_F32')
    if len(dst) != 1 or len(src) != (2 if accumulate else 3):
        return None
    # The FMA API explicitly represents the accumulator/literal input, so
    # staging the destination cannot discard its old value.
    inputs = src + dst if accumulate else src
    dx9 = 'DX9_ZERO' in mnemonic
    lines = []
    modifiers = '0'
    if is_vop3:
        lines = [
            '  const uint64_t modifiers = ((inst_.abs & 1) ? RJ_ARITH_ALU_ABS_A : 0u) |',
            '      ((inst_.abs & 2) ? RJ_ARITH_ALU_ABS_B : 0u) |',
            '      ((inst_.neg & 1) ? RJ_ARITH_ALU_NEG_A : 0u) |',
            '      ((inst_.neg & 2) ? RJ_ARITH_ALU_NEG_B : 0u) |',
        ]
        if not accumulate:
            lines += [
                '      ((inst_.abs & 4) ? RJ_ARITH_ALU_ABS_C : 0u) |',
                '      ((inst_.neg & 4) ? RJ_ARITH_ALU_NEG_C : 0u) |',
            ]
        lines += [
            '      (uint64_t(inst_.omod) * RJ_ARITH_ALU_OMOD_2) |',
            '      (inst_.clamp ? RJ_ARITH_ALU_CLAMP : 0u);',
        ]
        modifiers = 'modifiers'
    lines += [
        '  amdgpu::arith::execute_float_fma32(wf,',
        f'      wf.cu().arithmetic_flags(), {modifiers},',
        f'      {dst[0]}, {", ".join(inputs)},',
        f'      {str(arch == "rdna3").lower()}, {str(dx9).lower()});',
    ]
    return '\n'.join(lines)


def gen_mixed_fma(ctx: ExecuteContext) -> str | None:
    arch = ctx.arch_name
    mnemonic = ctx.inst.name
    dst = ctx.dst_ops
    src = ctx.src_ops
    enc_name = ctx.enc_name
    opsel_exprs = ctx.opsel_exprs
    hi_c_expr = ctx.op_sel_hi_2_expr
    if arch not in ('rdna3', 'rdna4') or mnemonic not in MIXED_FMA_NAMES:
        return None
    if enc_name != 'ENC_VOP3P' or len(dst) != 1 or len(src) != 3:
        return None
    low, high = opsel_exprs
    terms = ['(inst_.clamp ? RJ_ARITH_ALU_CLAMP : 0u)']
    narrow = []
    for index, source in enumerate('ABC'):
        selected = f'({high} & {1 << index})' if index < 2 else hi_c_expr
        terms += [
            f'({selected} ? RJ_ARITH_MIX_F16_{source} : 0u)',
            f'(({low} & {1 << index}) ? RJ_ARITH_ALU_HIGH_{source} : 0u)',
            f'((inst_.neg & {1 << index}) ? RJ_ARITH_ALU_NEG_{source} : 0u)',
            f'((inst_.neg_hi & {1 << index}) ? RJ_ARITH_ALU_ABS_{source} : 0u)',
        ]
        narrow += [
            f'({selected} && amdgpu::is_inline_float_src(inst_.src{index}) ? {1 << index}u : 0u)'
        ]
    api = 'rj_arith_' + mnemonic.lower() + ('_gfx12' if arch == 'rdna4' else '')
    byte_mask = (
        '0xF'
        if mnemonic == 'V_FMA_MIX_F32'
        else '0xC' if 'MIXHI' in mnemonic else '0x3'
    )
    return '\n'.join(
        [
            '  const uint64_t modifiers = ' + ' |\n      '.join(terms) + ';',
            '  const unsigned narrow_sources = ' + ' | '.join(narrow) + ';',
            f'  amdgpu::arith::execute_mixed_fma(wf, {wave_api(api, 32)}, {wave_api(api, 64)},',
            '      wf.cu().arithmetic_flags(), modifiers,',
            f'      {dst[0]}, {", ".join(src)}, narrow_sources, {byte_mask}, {str(arch == "rdna3").lower()});',
        ]
    )


def gen_packed_fma16(ctx: ExecuteContext) -> str | None:
    arch = ctx.arch_name
    mnemonic = ctx.inst.name
    dst = ctx.dst_ops
    src = ctx.src_ops
    enc_name = ctx.enc_name
    opsel_exprs = ctx.opsel_exprs
    hi_c_expr = ctx.op_sel_hi_2_expr
    if arch not in ('rdna3', 'rdna4') or mnemonic not in PACKED_FMA16_NAMES:
        return None
    accumulate = mnemonic == 'V_PK_FMAC_F16'
    count = 2 if accumulate else 3
    if len(dst) != 1 or len(src) != count:
        return None
    if not accumulate and enc_name != 'ENC_VOP3P':
        return None
    lines = []
    terms = []
    if accumulate:
        if 'VOP3' in enc_name:
            terms = [
                '((inst_.abs & 1) ? RJ_ARITH_ALU_ABS_A : 0u)',
                '((inst_.abs & 2) ? RJ_ARITH_ALU_ABS_B : 0u)',
                '((inst_.neg & 1) ? RJ_ARITH_ALU_NEG_A : 0u)',
                '((inst_.neg & 2) ? RJ_ARITH_ALU_NEG_B : 0u)',
                '(inst_.clamp ? RJ_ARITH_ALU_CLAMP : 0u)',
            ]
    else:
        low, high = opsel_exprs
        terms = ['(inst_.clamp ? RJ_ARITH_PK_CLAMP : 0u)']
        for index, source in enumerate('ABC'):
            selected = f'({high} & {1 << index})' if index < 2 else hi_c_expr
            terms += [
                f'(({low} & {1 << index}) ? RJ_ARITH_PK_LO_{source}_HIGH : 0u)',
                f'({selected} ? 0u : RJ_ARITH_PK_HI_{source}_LOW)',
                f'((inst_.neg & {1 << index}) ? RJ_ARITH_PK_NEG_LO_{source} : 0u)',
                f'((inst_.neg_hi & {1 << index}) ? RJ_ARITH_PK_NEG_HI_{source} : 0u)',
            ]
    lines.append(
        '  const uint64_t modifiers = '
        + (' |\n      '.join(terms) if terms else '0')
        + ';'
    )
    narrow, replicate = [], []
    for index, operand in enumerate(src):
        # VOP2's B is a VGPR field, never an inline selector.
        if accumulate and 'VOP3' not in enc_name and index == 1:
            continue
        selector = f'inst_.src{index}'
        narrow.append(
            f'(amdgpu::pk16_src_needs_narrowing({selector}, {operand}.size_bits()) ? {1 << index}u : 0u)'
        )
        if accumulate:
            replicate.append(
                f'(amdgpu::dot2_src_needs_half_replication({selector}) ? {1 << index}u : 0u)'
            )
    lines.append('  const unsigned narrow_sources = ' + ' | '.join(narrow) + ';')
    lines.append(
        '  const unsigned replicate_sources = '
        + (' | '.join(replicate) if replicate else '0')
        + ';'
    )
    addend = dst[0] if accumulate else src[2]
    lines += [
        f'  amdgpu::arith::execute_packed_fma16<{str(accumulate).lower()}>(wf,',
        '      wf.cu().arithmetic_flags(), modifiers,',
        f'      {dst[0]}, {src[0]}, {src[1]}, {addend}, {str(arch == "rdna3").lower()},',
        '      narrow_sources, replicate_sources);',
    ]
    return '\n'.join(lines)


def gen_packed_binary16(ctx: ExecuteContext) -> str | None:
    arch = ctx.arch_name
    mnemonic = ctx.inst.name
    dst = ctx.dst_ops
    src = ctx.src_ops
    enc_name = ctx.enc_name
    opsel_exprs = ctx.opsel_exprs
    if arch not in ('rdna3', 'rdna4') or mnemonic not in PACKED_BINARY16_NAMES:
        return None
    if enc_name != 'ENC_VOP3P' or len(dst) != 1 or len(src) != 2:
        return None
    legacy = mnemonic in ('V_PK_MIN_F16', 'V_PK_MAX_F16')
    arithmetic = mnemonic in ('V_PK_ADD_F16', 'V_PK_MUL_F16')
    if not arithmetic and legacy != (arch == 'rdna3'):
        return None
    low, high = opsel_exprs
    terms = ['(inst_.clamp ? RJ_ARITH_PK_CLAMP : 0u)']
    narrow = []
    for index, source in enumerate('AB'):
        terms += [
            f'(({low} & {1 << index}) ? RJ_ARITH_PK_LO_{source}_HIGH : 0u)',
            f'(({high} & {1 << index}) ? 0u : RJ_ARITH_PK_HI_{source}_LOW)',
            f'((inst_.neg & {1 << index}) ? RJ_ARITH_PK_NEG_LO_{source} : 0u)',
            f'((inst_.neg_hi & {1 << index}) ? RJ_ARITH_PK_NEG_HI_{source} : 0u)',
        ]
        narrow.append(
            f'(amdgpu::pk16_src_needs_narrowing(inst_.src{index}, {src[index]}.size_bits()) ? {1 << index}u : 0u)'
        )
    api = (
        'rj_arith_'
        + mnemonic.lower()
        + ('_gfx12' if arch == 'rdna4' and arithmetic else '')
    )
    lines = [
        '  const uint64_t modifiers = ' + ' |\n      '.join(terms) + ';',
        '  const unsigned narrow_sources = ' + ' | '.join(narrow) + ';',
        f'  amdgpu::arith::execute_packed_binary16(wf, {wave_api(api, 32)}, {wave_api(api, 64)},',
        '      wf.cu().arithmetic_flags(), modifiers,',
        f'      {dst[0]}, {src[0]}, {src[1]}, {str(arch == "rdna3").lower()}, narrow_sources);',
    ]
    return '\n'.join(lines)


def gen_float_dot32(ctx: ExecuteContext) -> str | None:
    arch = ctx.arch_name
    mnemonic = ctx.inst.name
    dst = ctx.dst_ops
    src = ctx.src_ops
    enc_name = ctx.enc_name
    opsel_exprs = ctx.opsel_exprs
    if arch not in ('rdna3', 'rdna4') or len(dst) != 1:
        return None
    if mnemonic in DOT4_F32_NAMES:
        if arch != 'rdna4' or enc_name != 'ENC_VOP3P' or len(src) != 3:
            return None
        # RDNA4 ISA 7.7: only C has sign modifiers; NEG_HI[2] is ABS_C.
        low, high = opsel_exprs
        api = 'rj_arith_' + mnemonic.lower()
        return '\n'.join(
            [
                '  const uint64_t modifiers = ((inst_.neg & 4) ? RJ_ARITH_DOT_NEG_C : 0u) |',
                '      ((inst_.neg_hi & 4) ? RJ_ARITH_DOT_ABS_C : 0u) |',
                f'      (({low} & 1) ? RJ_ARITH_DOT_LO_A_HIGH : 0u) |',
                f'      (({low} & 2) ? RJ_ARITH_DOT_LO_B_HIGH : 0u) |',
                f'      (({high} & 1) ? 0u : RJ_ARITH_DOT_HI_A_LOW) |',
                f'      (({high} & 2) ? 0u : RJ_ARITH_DOT_HI_B_LOW);',
                f'  amdgpu::arith::execute_ternary32(wf, {wave_api(api, 32)}, {wave_api(api, 64)},',
                '      wf.cu().arithmetic_flags(), modifiers,',
                f'      {dst[0]}, {", ".join(src)});',
            ]
        )
    if mnemonic not in DOT2_F32_NAMES:
        return None
    accumulate = mnemonic == 'V_DOT2ACC_F32_F16'
    if accumulate:
        if arch != 'rdna3' or enc_name != 'ENC_VOP2' or len(src) != 2:
            return None
    elif enc_name != 'ENC_VOP3P' or len(src) != 3:
        return None
    bf16 = mnemonic == 'V_DOT2_F32_BF16'
    terms = []
    if not accumulate:
        low, high = opsel_exprs
        terms = [
            '((inst_.neg & 4) ? RJ_ARITH_DOT_NEG_C : 0u)',
            '((inst_.neg_hi & 4) ? RJ_ARITH_DOT_ABS_C : 0u)',
            '(inst_.clamp ? RJ_ARITH_DOT_CLAMP : 0u)',
        ]
        for index, source in enumerate('AB'):
            terms += [
                f'(({low} & {1 << index}) ? RJ_ARITH_DOT_LO_{source}_HIGH : 0u)',
                f'(({high} & {1 << index}) ? 0u : RJ_ARITH_DOT_HI_{source}_LOW)',
                f'((inst_.neg & {1 << index}) ? RJ_ARITH_DOT_NEG_LO_{source} : 0u)',
                f'((inst_.neg_hi & {1 << index}) ? RJ_ARITH_DOT_NEG_HI_{source} : 0u)',
            ]
    narrow, replicate = [], []
    for index in range(1 if accumulate else 2):
        selector = f'inst_.src{index}'
        if bf16:
            broadcast = f'amdgpu::dot2_src_needs_half_replication({selector})'
            replicate.append(f'({broadcast} ? {1 << index}u : 0u)')
            # BF16 broadcasts upper16(FP32). GFX11 integer inline constants
            # also use upper16; GFX12 integer constants instead use low16.
            condition = (
                broadcast
                if arch == 'rdna3'
                else f'amdgpu::is_inline_float_src({selector})'
            )
        else:
            condition = f'amdgpu::pk16_src_needs_narrowing({selector}, {src[index]}.size_bits())'
        narrow.append(f'({condition} ? {1 << index}u : 0u)')
    api = (
        'rj_arith_v_dot2_f32_'
        + ('bf16' if bf16 else 'f16')
        + ('_gfx12' if arch == 'rdna4' else '')
    )
    addend = dst[0] if accumulate else src[2]
    return '\n'.join(
        [
            '  const uint64_t modifiers = '
            + (' |\n      '.join(terms) if terms else '0')
            + ';',
            '  const unsigned narrow_sources = ' + ' | '.join(narrow) + ';',
            '  const unsigned replicate_sources = '
            + (' | '.join(replicate) if replicate else '0')
            + ';',
            f'  amdgpu::arith::execute_dot2_f32(wf, {wave_api(api, 32)}, {wave_api(api, 64)},',
            '      wf.cu().arithmetic_flags(), modifiers,',
            f'      {dst[0]}, {src[0]}, {src[1]}, {addend}, {str(bf16).lower()}, narrow_sources, replicate_sources);',
        ]
    )


def gen_half_dot(ctx: ExecuteContext) -> str | None:
    arch = ctx.arch_name
    mnemonic = ctx.inst.name
    dst = ctx.dst_ops
    src = ctx.src_ops
    is_vop3 = ctx.is_vop3
    if arch not in ('rdna3', 'rdna4') or mnemonic not in HALF_DOT_NAMES:
        return None
    if not is_vop3 or len(dst) != 1 or len(src) != 3:
        return None
    api = 'rj_arith_' + mnemonic.lower() + ('_gfx12' if arch == 'rdna4' else '')
    bf16 = str(mnemonic == 'V_DOT2_BF16_BF16').lower()
    lines = [
        '  const uint32_t opsel = amdgpu::vop3_opsel(inst_);',
        '  const uint64_t modifiers = ((inst_.abs & 1) ? RJ_ARITH_ALU_ABS_A : 0u) |',
        '      ((inst_.abs & 2) ? RJ_ARITH_ALU_ABS_B : 0u) |',
        '      ((inst_.abs & 4) ? RJ_ARITH_ALU_ABS_C : 0u) |',
        '      ((inst_.neg & 1) ? RJ_ARITH_ALU_NEG_A : 0u) |',
        '      ((inst_.neg & 2) ? RJ_ARITH_ALU_NEG_B : 0u) |',
        '      ((inst_.neg & 4) ? RJ_ARITH_ALU_NEG_C : 0u) |',
        '      ((opsel & 4) ? RJ_ARITH_ALU_HIGH_C : 0u) |',
        '      ((opsel & 8) ? RJ_ARITH_ALU_HIGH_D : 0u);',
    ]
    narrow, replicate = [], []
    for index in range(2):
        selector = f'inst_.src{index}'
        narrow.append(
            f'(amdgpu::pk16_src_needs_narrowing({selector}, {src[index]}.size_bits()) ? {1 << index}u : 0u)'
        )
        replicate.append(
            f'(amdgpu::dot2_src_needs_half_replication({selector}) ? {1 << index}u : 0u)'
        )
    replicate.append('(amdgpu::dot2_src_needs_half_replication(inst_.src2) ? 4u : 0u)')
    lines += [
        '  const unsigned narrow_sources = ' + ' | '.join(narrow) + ';',
        '  const unsigned replicate_sources = ' + ' | '.join(replicate) + ';',
        f'  amdgpu::arith::execute_dot2_f16(wf, {wave_api(api, 32)}, {wave_api(api, 64)},',
        '      wf.cu().arithmetic_flags(), modifiers,',
        f'      {dst[0]}, {", ".join(src)}, {bf16}, narrow_sources, replicate_sources);',
    ]
    return '\n'.join(lines)


def gen_float_binary16(ctx: ExecuteContext) -> str | None:
    arch = ctx.arch_name
    mnemonic = ctx.inst.name
    dst = ctx.dst_ops
    src = ctx.src_ops
    is_vop3 = ctx.is_vop3
    if arch not in ('rdna3', 'rdna4') or mnemonic not in BINARY_F16:
        return None
    if len(dst) != 1 or len(src) != 2:
        return None
    if mnemonic in ('V_MIN_F16', 'V_MAX_F16') and arch != 'rdna3':
        return None
    if (
        mnemonic in BINARY_SELECTION_F16 - {'V_MIN_F16', 'V_MAX_F16'}
        and arch != 'rdna4'
    ):
        return None
    api = (
        'rj_arith_'
        + mnemonic.lower()
        + ('_gfx12' if arch == 'rdna4' and mnemonic in BINARY_ARITHMETIC_F16 else '')
    )
    lines = []
    modifiers = '0'
    if is_vop3:
        lines += [
            '  const uint32_t opsel = amdgpu::vop3_opsel(inst_);',
            '  const uint64_t modifiers = ((inst_.abs & 1) ? RJ_ARITH_ALU_ABS_A : 0u) |',
            *(
                [
                    '      ((inst_.abs & 2) ? RJ_ARITH_ALU_ABS_B : 0u) |',
                ]
                if mnemonic != "V_LDEXP_F16"
                else []
            ),
            '      ((inst_.neg & 1) ? RJ_ARITH_ALU_NEG_A : 0u) |',
            *(
                [
                    '      ((inst_.neg & 2) ? RJ_ARITH_ALU_NEG_B : 0u) |',
                ]
                if mnemonic != "V_LDEXP_F16"
                else []
            ),
            '      ((opsel & 1) ? RJ_ARITH_ALU_HIGH_A : 0u) |',
            '      ((opsel & 2) ? RJ_ARITH_ALU_HIGH_B : 0u) |',
            '      ((opsel & 8) ? RJ_ARITH_ALU_HIGH_D : 0u) |',
            '      (uint64_t(inst_.omod) * RJ_ARITH_ALU_OMOD_2) |',
            '      (inst_.clamp ? RJ_ARITH_ALU_CLAMP : 0u);',
        ]
        modifiers = 'modifiers'
    lines += [
        f'  amdgpu::arith::execute_float_binary16(wf, {wave_api(api, 32)}, {wave_api(api, 64)},',
        f'      wf.cu().arithmetic_flags(), {modifiers},',
        f'      {dst[0]}, {src[0]}, {src[1]}, {str(arch == "rdna3").lower()});',
    ]
    return '\n'.join(lines)


def gen_float_fma16(ctx: ExecuteContext) -> str | None:
    arch = ctx.arch_name
    mnemonic = ctx.inst.name
    dst = ctx.dst_ops
    src = ctx.src_ops
    is_vop3 = ctx.is_vop3
    if arch not in ('rdna3', 'rdna4') or mnemonic not in FMA_F16:
        return None
    accumulate = mnemonic == 'V_FMAC_F16'
    if len(dst) != 1 or len(src) != (2 if accumulate else 3):
        return None
    # Literal operands already appear in multiplication/addition order. FMAC
    # supplies old D explicitly so output staging cannot discard its addend.
    inputs = src + dst if accumulate else src
    api = 'rj_arith_v_fma_f16' + ('_gfx12' if arch == 'rdna4' else '')
    modifiers = '0'
    lines = []
    if is_vop3:
        terms = [
            '((inst_.abs & 1) ? RJ_ARITH_ALU_ABS_A : 0u)',
            '((inst_.abs & 2) ? RJ_ARITH_ALU_ABS_B : 0u)',
            '((inst_.neg & 1) ? RJ_ARITH_ALU_NEG_A : 0u)',
            '((inst_.neg & 2) ? RJ_ARITH_ALU_NEG_B : 0u)',
            '((opsel & 1) ? RJ_ARITH_ALU_HIGH_A : 0u)',
            '((opsel & 2) ? RJ_ARITH_ALU_HIGH_B : 0u)',
            '((opsel & 8) ? RJ_ARITH_ALU_HIGH_D : 0u)',
            '(uint64_t(inst_.omod) * RJ_ARITH_ALU_OMOD_2)',
            '(inst_.clamp ? RJ_ARITH_ALU_CLAMP : 0u)',
        ]
        if accumulate:
            terms += ['((opsel & 8) ? RJ_ARITH_ALU_HIGH_C : 0u)']
        else:
            terms += [
                '((opsel & 4) ? RJ_ARITH_ALU_HIGH_C : 0u)',
                '((inst_.abs & 4) ? RJ_ARITH_ALU_ABS_C : 0u)',
                '((inst_.neg & 4) ? RJ_ARITH_ALU_NEG_C : 0u)',
            ]
        lines += [
            '  const uint32_t opsel = amdgpu::vop3_opsel(inst_);',
            '  const uint64_t modifiers = ' + ' |\n      '.join(terms) + ';',
        ]
        modifiers = 'modifiers'
    lines += [
        f'  amdgpu::arith::execute_float_ternary16(wf, {wave_api(api, 32)}, {wave_api(api, 64)},',
        f'      wf.cu().arithmetic_flags(), {modifiers},',
        f'      {dst[0]}, {", ".join(inputs)}, {str(arch == "rdna3").lower()});',
    ]
    return '\n'.join(lines)


def gen_float_unary16(ctx: ExecuteContext) -> str | None:
    arch = ctx.arch_name
    mnemonic = ctx.inst.name
    dst = ctx.dst_ops
    src = ctx.src_ops
    is_vop3 = ctx.is_vop3
    if arch not in ('rdna3', 'rdna4') or mnemonic not in UNARY_F16:
        return None
    if len(dst) != 1 or len(src) != 1:
        return None
    api = 'rj_arith_' + mnemonic.lower() + ('_gfx12' if arch == 'rdna4' else '')
    modifiers = '0'
    lines = []
    if is_vop3:
        lines += [
            '  const uint32_t opsel = amdgpu::vop3_opsel(inst_);',
            '  const uint64_t modifiers = ((inst_.abs & 1) ? RJ_ARITH_ALU_ABS_A : 0u) |',
            '      ((inst_.neg & 1) ? RJ_ARITH_ALU_NEG_A : 0u) |',
            '      ((opsel & 1) ? RJ_ARITH_ALU_HIGH_A : 0u) |',
            '      ((opsel & 8) ? RJ_ARITH_ALU_HIGH_D : 0u) |',
        ]
        if mnemonic.startswith('V_FREXP_EXP_'):
            # Integer results ignore floating-point output modifiers.
            lines[-1] = lines[-1].removesuffix(' |') + ';'
        else:
            lines += [
                '      (uint64_t(inst_.omod) * RJ_ARITH_ALU_OMOD_2) |',
                '      (inst_.clamp ? RJ_ARITH_ALU_CLAMP : 0u);',
            ]
        modifiers = 'modifiers'
    guest_rounding = mnemonic.split('_')[1] in {
        'TRUNC',
        'CEIL',
        'FLOOR',
        'RNDNE',
        'FRACT',
    }
    lines += [
        f'  amdgpu::arith::execute_float_unary16(wf, {wave_api(api, 32)}, {wave_api(api, 64)},',
        f'      wf.cu().arithmetic_flags(), {modifiers},',
        f'      {dst[0]}, {src[0]}, {str(arch == "rdna3").lower()}, /*guest_rounding=*/{str(guest_rounding).lower()});',
    ]
    return '\n'.join(lines)


def gen_float_ternary_selection(ctx: ExecuteContext) -> str | None:
    arch = ctx.arch_name
    mnemonic = ctx.inst.name
    dst = ctx.dst_ops
    src = ctx.src_ops
    is_vop3 = ctx.is_vop3
    names = {'rdna3': TERNARY_SELECTION_RDNA3, 'rdna4': TERNARY_SELECTION_RDNA4}.get(
        arch, ()
    )
    if mnemonic not in names or not is_vop3 or len(dst) != 1 or len(src) != 3:
        return None
    half = mnemonic.endswith('_F16')
    width = 16 if half else 32
    api = 'rj_arith_' + mnemonic.lower()
    lines = []
    if half:
        lines.append('  const uint32_t opsel = amdgpu::vop3_opsel(inst_);')
    terms = []
    for index, letter in enumerate('ABC'):
        terms += [
            f'((inst_.abs & {1 << index}) ? RJ_ARITH_ALU_ABS_{letter} : 0u)',
            f'((inst_.neg & {1 << index}) ? RJ_ARITH_ALU_NEG_{letter} : 0u)',
        ]
        if half:
            terms.append(f'((opsel & {1 << index}) ? RJ_ARITH_ALU_HIGH_{letter} : 0u)')
    if half:
        terms.append('((opsel & 8) ? RJ_ARITH_ALU_HIGH_D : 0u)')
    terms += [
        '(uint64_t(inst_.omod) * RJ_ARITH_ALU_OMOD_2)',
        '(inst_.clamp ? RJ_ARITH_ALU_CLAMP : 0u)',
    ]
    lines += [
        '  const uint64_t modifiers = ' + ' |\n      '.join(terms) + ';',
        f'  amdgpu::arith::execute_float_ternary{width}(wf, {wave_api(api, 32)}, {wave_api(api, 64)},',
        '      wf.cu().arithmetic_flags(), modifiers,',
        f'      {dst[0]}, {", ".join(src)}, {str(arch == "rdna3").lower()});',
    ]
    return '\n'.join(lines)


FLOAT_NAMES = (
    BINARY_ARITHMETIC_F16
    | BINARY_F16
    | BINARY_F32
    | BINARY_F64['rdna3']
    | BINARY_F64['rdna4']
    | BINARY_SELECTION_F16
    | DOT2_F32_NAMES
    | DOT4_F32_NAMES
    | FMA_F16
    | FMA_F32
    | HALF_DOT_NAMES
    | MIXED_FMA_NAMES
    | PACKED_BINARY16_NAMES
    | PACKED_FMA16_NAMES
    | TERNARY_SELECTION_RDNA3
    | TERNARY_SELECTION_RDNA4
    | UNARY_F16
    | UNARY_F32
    | UNARY_F64
    | {'V_TRIG_PREOP_F64', 'V_FMA_F64', 'V_LDEXP_F64'}
)
