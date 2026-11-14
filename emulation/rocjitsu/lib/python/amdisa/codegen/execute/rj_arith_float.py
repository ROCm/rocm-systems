# Copyright (c) 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""rj-arith floating arithmetic bindings with explicit guest-mode and state policy."""

from . import ExecuteContext
from .rj_arith_names import wave_api
from .rj_arith_graphics import CUBE_NAMES

PSEUDO_SCALAR_NAMES = {
    f'V_S_{operation}_F{bits}'
    for operation in ('EXP', 'LOG', 'RCP', 'RSQ', 'SQRT')
    for bits in (16, 32)
}

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

PACKED_CONVERSION_NAMES = {
    'V_CVT_PK_RTZ_F16_F32',
    'V_CVT_PK_I16_F32',
    'V_CVT_PK_U16_F32',
}

NORMALIZED_NAMES = {
    'V_CVT_PK_NORM_I16_F32',
    'V_CVT_PK_NORM_U16_F32',
    'V_CVT_PK_NORM_I16_F16',
    'V_CVT_PK_NORM_U16_F16',
    'V_CVT_NORM_I16_F16',
    'V_CVT_NORM_U16_F16',
}

CONVERSION16_NAMES = {
    'V_CVT_F16_I16',
    'V_CVT_F16_U16',
    'V_CVT_I16_F16',
    'V_CVT_U16_F16',
    'V_CVT_F16_F32',
    'V_CVT_F32_F16',
}

CONVERSION32_NAMES = {
    'V_CVT_F32_I32',
    'V_CVT_F32_U32',
    'V_CVT_I32_F32',
    'V_CVT_U32_F32',
    'V_CVT_NEAREST_I32_F32',
    'V_CVT_FLOOR_I32_F32',
}

CONVERSION64_NAMES = {
    'V_CVT_F64_I32',
    'V_CVT_F64_U32',
    'V_CVT_I32_F64',
    'V_CVT_U32_F64',
    'V_CVT_F64_F32',
    'V_CVT_F32_F64',
}

FLOAT16_CONVERSION_NAMES = {
    'V_CVT_F32_BF16',
    'V_CVT_PK_BF16_F32',
    'V_CVT_SR_BF16_F32',
    'V_CVT_SR_PK_BF16_F32',
    'V_CVT_PK_F16_F32',
    'V_CVT_SR_F16_F32',
    'V_CVT_SR_PK_F16_F32',
}

FP8_NARROW_NAMES = {
    'V_CVT_PK_FP8_F32',
    'V_CVT_PK_BF8_F32',
    'V_CVT_SR_FP8_F32',
    'V_CVT_SR_BF8_F32',
    'V_CVT_PK_FP8_F16',
    'V_CVT_PK_BF8_F16',
    'V_CVT_SR_FP8_F16',
    'V_CVT_SR_BF8_F16',
}

FP8_WIDEN_NAMES = {
    'V_CVT_F32_FP8',
    'V_CVT_F32_BF8',
    'V_CVT_PK_F32_FP8',
    'V_CVT_PK_F32_BF8',
    'V_CVT_F16_FP8',
    'V_CVT_F16_BF8',
    'V_CVT_PK_F16_FP8',
    'V_CVT_PK_F16_BF8',
}

BYTE_NIBBLE_CONVERSION_NAMES = {f'V_CVT_F32_UBYTE{byte}' for byte in range(4)} | {
    'V_CVT_OFF_F32_I4'
}

BIT_PACK_NAMES = {'V_PACK_B32_F16', 'V_CVT_PK_U8_F32'}

SELECTION_F64 = {
    'rdna3': {'V_MIN_F64', 'V_MAX_F64'},
    'rdna4': {'V_MIN_NUM_F64', 'V_MAX_NUM_F64', 'V_MINIMUM_F64', 'V_MAXIMUM_F64'},
}

BINARY_F64 = {
    arch: names | {'V_ADD_F64', 'V_MUL_F64'} for arch, names in SELECTION_F64.items()
}
BINARY_ARITHMETIC_F32 = {
    'V_LDEXP_F32',
    'V_ADD_F32',
    'V_MUL_F32',
    'V_SUB_F32',
    'V_SUBREV_F32',
    'V_MUL_DX9_ZERO_F32',
}
SELECTION_F32 = {
    'rdna3': {'V_MIN_F32', 'V_MAX_F32'},
    'rdna4': {'V_MIN_NUM_F32', 'V_MAX_NUM_F32', 'V_MINIMUM_F32', 'V_MAXIMUM_F32'},
}
BINARY_F32 = BINARY_ARITHMETIC_F32 | SELECTION_F32['rdna3'] | SELECTION_F32['rdna4']
GFX10_LEGACY_FMA = {'V_FMA_LEGACY_F32', 'V_FMAC_LEGACY_F32'}

FMA_F32 = {
    'V_FMA_F32',
    'V_FMAC_F32',
    'V_FMA_DX9_ZERO_F32',
    'V_FMAC_DX9_ZERO_F32',
    'V_FMAAK_F32',
    'V_FMAMK_F32',
}

MIXED_FMA_NAMES = {'V_FMA_MIX_F32', 'V_FMA_MIXLO_F16', 'V_FMA_MIXHI_F16'}
MIXED_BF16_NAMES = {'V_FMA_MIX_F32_BF16', 'V_FMA_MIXLO_BF16', 'V_FMA_MIXHI_BF16'}

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

DOT2_F32_NAMES = {
    'V_DOT2_F32_F16',
    'V_DOT2_F32_BF16',
    'V_DOT2ACC_F32_F16',
    'V_DOT2C_F32_F16',
    'V_DOT2C_F32_BF16',
}

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

GFX10_BINARY_F16 = BINARY_ARITHMETIC_F16 - {'V_LDEXP_F16'}

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

FLOAT_NAMES = (
    CUBE_NAMES
    | BINARY_F16
    | BINARY_F32
    | BINARY_F64['rdna3']
    | BINARY_F64['rdna4']
    | BIT_PACK_NAMES
    | BYTE_NIBBLE_CONVERSION_NAMES
    | CONVERSION16_NAMES
    | CONVERSION32_NAMES
    | CONVERSION64_NAMES
    | DOT2_F32_NAMES
    | DOT4_F32_NAMES
    | FMA_F16
    | FMA_F32
    | FP8_NARROW_NAMES
    | FP8_WIDEN_NAMES
    | HALF_DOT_NAMES
    | MIXED_FMA_NAMES
    | NORMALIZED_NAMES
    | PACKED_BINARY16_NAMES
    | PACKED_CONVERSION_NAMES
    | PACKED_FMA16_NAMES
    | PSEUDO_SCALAR_NAMES
    | TERNARY_SELECTION_RDNA3
    | TERNARY_SELECTION_RDNA4
    | UNARY_F16
    | UNARY_F32
    | UNARY_F64
    | {'V_MULLIT_F32', 'V_TRIG_PREOP_F64', 'V_FMA_F64', 'V_LDEXP_F64'}
)


class RjArithBody(str):
    """An emitted body whose arithmetic and SIMD dispatch belong to rj-arith."""


def gen_pseudo_scalar(ctx: ExecuteContext) -> str | None:
    arch = ctx.arch_name
    mnemonic = ctx.inst.name
    dst = ctx.dst_ops
    src = ctx.src_ops
    is_vop3 = ctx.is_vop3
    if (
        arch not in ('rdna4', 'cdna5')
        or mnemonic not in PSEUDO_SCALAR_NAMES
        or not is_vop3
    ):
        return None
    if len(dst) != 1 or len(src) != 1:
        return None
    half = mnemonic.endswith('F16')
    return '\n'.join(
        [
            '  const uint64_t modifiers = ((inst_.abs & 1) ? RJ_ARITH_ALU_ABS_A : 0u) |',
            '      ((inst_.neg & 1) ? RJ_ARITH_ALU_NEG_A : 0u) |',
            '      (uint64_t(inst_.omod) * RJ_ARITH_ALU_OMOD_2) |',
            '      (inst_.clamp ? RJ_ARITH_ALU_CLAMP : 0u);',
            f'  amdgpu::arith::execute_pseudo_scalar(wf, rj_arith_{mnemonic.lower()}'
            f'{"_gfx1250" if arch == "cdna5" else ""},',
            f'      wf.cu().arithmetic_flags(),',
            f'      modifiers, {dst[0]}, {src[0]}, {str(half).lower()});',
        ]
    )


# RCP_IFLAG and SQRT carry their legacy exception policies into rj-arith;
# other entries here have no legacy classifier.


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
    # These legacy classifiers run on EXEC even when DPP suppresses the destination.
    status_scope = (
        ', /*compatibility_all_exec=*/true'
        if mnemonic in {'V_RCP_IFLAG_F32', 'V_SQRT_F32'}
        else ''
    )
    lines += [
        f'  amdgpu::arith::execute_float_unary32(wf, {wave_api(api, 32)}, {wave_api(api, 64)},',
        f'      wf.cu().arithmetic_flags(), {modifiers},',
        f'      {dst[0]}, {src[0]}, {str(arch == "rdna3").lower()}, /*guest_rounding=*/{str(guest_rounding).lower()}{status_scope});',
    ]
    return '\n'.join(lines)


# FP64 unary operations use the shared guest-control adapter. Fixed-rounding
# cores and native math approximations apply guest rounding to OMOD, except
# RCP overflow, which produces signed infinity regardless of ROUND.


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


def gen_packed_conversion(ctx: ExecuteContext) -> str | None:
    arch = ctx.arch_name
    mnemonic = ctx.inst.name.replace('_PKRTZ_', '_PK_RTZ_')
    dst = ctx.dst_ops
    src = ctx.src_ops
    is_vop3 = ctx.is_vop3
    if (
        arch not in ('rdna3', 'rdna4', 'cdna3', 'cdna4', 'cdna5')
        or mnemonic not in PACKED_CONVERSION_NAMES
    ):
        return None
    if len(dst) != 1 or len(src) != 2:
        return None
    api = (
        'rj_arith_'
        + mnemonic.lower()
        + ('_gfx12' if arch in ('rdna4', 'cdna5') else '')
    )
    lines = ['  uint64_t flags = wf.cu().arithmetic_flags();']
    if mnemonic == 'V_CVT_PK_RTZ_F16_F32':
        lines += [
            '  if (!(wf.fp_denorm_mode_f32() & 1)) flags |= RJ_ARITH_FP_FLUSH_INPUT_DENORMALS;',
            '  if (!(wf.fp_denorm_mode_f16_f64() & 2)) flags |= RJ_ARITH_FP_FLUSH_OUTPUT_DENORMALS;',
            '  if (wf.fp16_ovfl()) flags |= RJ_ARITH_FP16_OVFL;',
        ]
    modifiers = '0'
    if is_vop3:
        terms = [
            '((inst_.abs & 1) ? RJ_ARITH_ALU_ABS_A : 0u)',
            '((inst_.abs & 2) ? RJ_ARITH_ALU_ABS_B : 0u)',
            '((inst_.neg & 1) ? RJ_ARITH_ALU_NEG_A : 0u)',
            '((inst_.neg & 2) ? RJ_ARITH_ALU_NEG_B : 0u)',
            '(inst_.clamp ? RJ_ARITH_ALU_CLAMP : 0u)',
        ]
        if mnemonic == 'V_CVT_PK_RTZ_F16_F32':
            terms += ['(uint64_t(inst_.omod) * RJ_ARITH_ALU_OMOD_2)']
        lines += ['  const uint64_t modifiers = ' + ' | '.join(terms) + ';']
        modifiers = 'modifiers'
    lines += [
        f'  amdgpu::arith::execute_float_pack(wf, {wave_api(api, 32)}, {wave_api(api, 64)},',
        f'      flags, {modifiers},',
        f'      {dst[0]}, {src[0]}, {src[1]}, /*half_sources=*/false);',
    ]
    return '\n'.join(lines)


def gen_normalized(ctx: ExecuteContext) -> str | None:
    arch = ctx.arch_name
    mnemonic = ctx.inst.name.replace('_PKNORM_', '_PK_NORM_')
    dst = ctx.dst_ops
    src = ctx.src_ops
    is_vop3 = ctx.is_vop3
    if (
        arch not in ('rdna3', 'rdna4', 'cdna3', 'cdna4', 'cdna5')
        or mnemonic not in NORMALIZED_NAMES
    ):
        return None
    packed = '_PK_' in mnemonic
    half = mnemonic.endswith('_F16')
    if len(dst) != 1 or len(src) != (2 if packed else 1):
        return None
    api = 'rj_arith_' + mnemonic.lower()
    if packed and half and arch in ('rdna4', 'cdna5'):
        api += '_gfx12'
    if not packed and arch in ('cdna3', 'cdna4'):
        api += '_gfx942'
    lines = ['  uint64_t flags = wf.cu().arithmetic_flags();']
    if half:
        lines += [
            '  if (!(wf.fp_denorm_mode_f16_f64() & 1)) flags |= RJ_ARITH_FP_FLUSH_INPUT_DENORMALS;'
        ]
    if not packed and not is_vop3 and arch in ('cdna3', 'cdna4'):
        lines += [
            '  if (inst_.src0 == amdgpu::SRC_SDWA) {',
            '    amdgpu::arith::execute_normalized_sdwa(*this, wf,',
            f'        {wave_api(api, 32)}, {wave_api(api, 64)}, flags, {dst[0]}, {src[0]});',
            '    return;',
            '  }',
        ]
    terms = []
    if is_vop3:
        if half:
            lines += ['  const uint32_t opsel = amdgpu::vop3_opsel(inst_);']
        if packed:
            terms += ['(inst_.clamp ? RJ_ARITH_ALU_CLAMP : 0u)']
        for i, operand in enumerate('AB' if packed else 'A'):
            terms += [
                f'((inst_.abs & {1 << i}) ? RJ_ARITH_ALU_ABS_{operand} : 0u)',
                f'((inst_.neg & {1 << i}) ? RJ_ARITH_ALU_NEG_{operand} : 0u)',
            ]
            if half:
                terms += [f'((opsel & {1 << i}) ? RJ_ARITH_ALU_HIGH_{operand} : 0u)']
        if not packed:
            terms += [
                '((opsel & 8) ? RJ_ARITH_ALU_HIGH_D : 0u)',
            ]
    source_operand, destination_operand = src[0], dst[0]
    if not packed and not is_vop3 and arch not in ('cdna3', 'cdna4'):
        for operand, flag in ((src[0], 'A'), (dst[0], 'D')):
            terms += [
                f'((amdgpu::e32_packed_vgpr_operand({operand}) && ({operand}.encoding_value() & 0x80)) ? RJ_ARITH_ALU_HIGH_{flag} : 0u)'
            ]
        lines += [
            f'  auto word_source = amdgpu::e32_word_operand({src[0]});',
            f'  auto word_destination = amdgpu::e32_word_operand({dst[0]});',
        ]
        source_operand, destination_operand = 'word_source', 'word_destination'
    lines += [
        '  const uint64_t modifiers = ' + (' | '.join(terms) if terms else '0u') + ';'
    ]
    if not packed:
        destination_bytes = '(modifiers & RJ_ARITH_ALU_HIGH_D) ? 0xC : ' + (
            '0xF' if arch in ('cdna3', 'cdna4') else '0x3'
        )
        lines += [
            '  const uint8_t source_bytes = (modifiers & RJ_ARITH_ALU_HIGH_A) ? 0xC : 0x3;',
            f'  const uint8_t destination_bytes = {destination_bytes};',
            f'  amdgpu::arith::execute_unary32(wf, {wave_api(api, 32)}, {wave_api(api, 64)}, flags, modifiers,',
            f'      {destination_operand}, {source_operand}, source_bytes, destination_bytes);',
        ]
    elif half:
        lines += [
            f'  amdgpu::arith::execute_float_pack(wf, {wave_api(api, 32)}, {wave_api(api, 64)}, flags, modifiers,',
            f'      {dst[0]}, {src[0]}, {src[1]}, /*half_sources=*/true);',
        ]
    else:
        lines += [
            f'  amdgpu::arith::execute_binary32(wf, {wave_api(api, 32)}, {wave_api(api, 64)}, flags, modifiers,',
            f'      {dst[0]}, {src[0]}, {src[1]});',
        ]
    return '\n'.join(lines)


def gen_conversion16(ctx: ExecuteContext) -> str | None:
    arch = ctx.arch_name
    mnemonic = ctx.inst.name
    dst = ctx.dst_ops
    src = ctx.src_ops
    is_vop3 = ctx.is_vop3
    if (
        arch not in ('rdna3', 'rdna4', 'cdna3', 'cdna4', 'cdna5')
        or mnemonic not in CONVERSION16_NAMES
    ):
        return None
    if len(dst) != 1 or len(src) != 1:
        return None
    output, source = mnemonic.split('_')[-2:]
    suffix = (
        '_gfx12'
        if arch in ('rdna4', 'cdna5')
        else '_gfx942' if arch in ('cdna3', 'cdna4') and output != 'F32' else ''
    )
    api = 'rj_arith_' + mnemonic.lower() + suffix
    lines = [
        '  uint64_t flags = wf.cu().arithmetic_flags();',
        '  if (wf.ieee_mode()) flags |= RJ_ARITH_FP_IEEE_MODE;',
    ]
    if arch in ('rdna3', 'cdna3', 'cdna4') and output.startswith('F'):
        lines += ['  if (!wf.dx10_clamp()) flags |= RJ_ARITH_FP_DX10_CLAMP_OFF;']
    if output == 'F16':
        lines += ['  if (wf.fp16_ovfl()) flags |= RJ_ARITH_FP16_OVFL;']
    for fmt, bit, flag in ((source, 1, 'INPUT'), (output, 2, 'OUTPUT')):
        if fmt.startswith('F'):
            denorm = 'fp_denorm_mode_f32' if fmt == 'F32' else 'fp_denorm_mode_f16_f64'
            lines += [
                f'  if (!(wf.{denorm}() & {bit})) flags |= RJ_ARITH_FP_FLUSH_{flag}_DENORMALS;'
            ]
    if output.startswith('F'):
        rounding = 'fp_round_mode_f32' if output == 'F32' else 'fp_round_mode_f16_f64'
        lines += [f'  flags |= uint64_t(wf.{rounding}()) << 24;']
    if arch in ('cdna3', 'cdna4') and not is_vop3:
        result_format = (
            'F16' if output == 'F16' else 'F32' if output == 'F32' else 'NONE'
        )
        lines += [
            '  if (inst_.src0 == amdgpu::SRC_SDWA) {',
            f'    amdgpu::arith::execute_conversion16_sdwa<amdgpu::sdwa::ResultFormat::{result_format}>(*this, wf,',
            f'        {wave_api(api, 32)}, {wave_api(api, 64)}, flags, {dst[0]}, {src[0]});',
            '    return;',
            '  }',
        ]
    terms = []
    source_operand, destination_operand = src[0], dst[0]
    if is_vop3:
        lines += ['  const uint32_t opsel = amdgpu::vop3_opsel(inst_);']
        terms += [
            '(uint64_t(inst_.omod) * RJ_ARITH_ALU_OMOD_2)',
            '(inst_.clamp ? RJ_ARITH_ALU_CLAMP : 0u)',
        ]
        if source != 'F32':
            terms += ['((opsel & 1) ? RJ_ARITH_ALU_HIGH_A : 0u)']
        if output != 'F32':
            terms += ['((opsel & 8) ? RJ_ARITH_ALU_HIGH_D : 0u)']
        if source.startswith('F'):
            terms += [
                '((inst_.abs & 1) ? RJ_ARITH_ALU_ABS_A : 0u)',
                '((inst_.neg & 1) ? RJ_ARITH_ALU_NEG_A : 0u)',
            ]
    elif arch not in ('cdna3', 'cdna4'):
        # E32 true16 selectors combine the register and half. Expose the word
        # to SIMD access and pass half selection separately; staged operands
        # already contain the selected half and must remain untouched.
        for operand, fmt, flag in ((src[0], source, 'A'), (dst[0], output, 'D')):
            if fmt != 'F32':
                terms += [
                    f'((amdgpu::e32_packed_vgpr_operand({operand}) && '
                    f'({operand}.encoding_value() & 0x80)) ? RJ_ARITH_ALU_HIGH_{flag} : 0u)'
                ]
        lines += [
            '  // Keep packed E32 halves on the bulk register-access path.',
            f'  auto word_source = amdgpu::e32_word_operand({src[0]});',
            f'  auto word_destination = amdgpu::e32_word_operand({dst[0]});',
        ]
        source_operand, destination_operand = 'word_source', 'word_destination'
    lines += [
        '  const uint64_t modifiers = ' + (' | '.join(terms) if terms else '0u') + ';'
    ]
    source_bytes = (
        '0xF' if source == 'F32' else '(modifiers & RJ_ARITH_ALU_HIGH_A) ? 0xC : 0x3'
    )
    destination_bytes = (
        '0xF' if output == 'F32' else '(modifiers & RJ_ARITH_ALU_HIGH_D) ? 0xC : 0x3'
    )
    if arch in ('cdna3', 'cdna4') and output != 'F32':
        destination_bytes = '(modifiers & RJ_ARITH_ALU_HIGH_D) ? 0xC : 0xF'
    lines += [
        f'  const uint8_t source_bytes = {source_bytes};',
        f'  const uint8_t destination_bytes = {destination_bytes};',
        f'  amdgpu::arith::execute_float_conversion(wf, {wave_api(api, 32)}, {wave_api(api, 64)},',
        f'      flags, modifiers, {destination_operand}, {source_operand}, /*from_double=*/false, /*to_double=*/false,',
        '      source_bytes, destination_bytes);',
    ]
    return '\n'.join(lines)


def gen_conversion32(ctx: ExecuteContext) -> str | None:
    arch = ctx.arch_name
    mnemonic = {
        'V_CVT_RPI_I32_F32': 'V_CVT_NEAREST_I32_F32',
        'V_CVT_FLR_I32_F32': 'V_CVT_FLOOR_I32_F32',
    }.get(ctx.inst.name, ctx.inst.name)
    dst = ctx.dst_ops
    src = ctx.src_ops
    is_vop3 = ctx.is_vop3
    if (
        arch not in ('rdna3', 'rdna4', 'cdna3', 'cdna4', 'cdna5')
        or mnemonic not in CONVERSION32_NAMES
    ):
        return None
    if len(dst) != 1 or len(src) != 1:
        return None
    to_float = mnemonic in ('V_CVT_F32_I32', 'V_CVT_F32_U32')
    fixed_round = mnemonic in ('V_CVT_NEAREST_I32_F32', 'V_CVT_FLOOR_I32_F32')
    reports = not to_float and not fixed_round
    api = (
        'rj_arith_'
        + mnemonic.lower()
        + ('_gfx12' if arch in ('rdna4', 'cdna5') and not fixed_round else '')
    )
    # Conversion does not use DX10 clamp; only integer-to-float uses MODE rounding.
    ignored = 'RJ_ARITH_FP_DX10_CLAMP_OFF'
    if not to_float:
        ignored += ' | RJ_ARITH_FP_ROUND_MASK'
    lines = [
        '  const uint64_t flags = amdgpu::arith::float32_flags(',
        f'      wf, wf.cu().arithmetic_flags(), {str(arch in ("rdna3", "cdna3", "cdna4")).lower()}) & ~({ignored});',
    ]
    modifiers = []
    if is_vop3:
        modifiers += ['(inst_.clamp ? RJ_ARITH_ALU_CLAMP : 0u)']
        if not fixed_round:
            modifiers += ['(uint64_t(inst_.omod) * RJ_ARITH_ALU_OMOD_2)']
        if not to_float:
            modifiers += [
                '((inst_.abs & 1) ? RJ_ARITH_ALU_ABS_A : 0u)',
                '((inst_.neg & 1) ? RJ_ARITH_ALU_NEG_A : 0u)',
            ]
    lines += [
        '  const uint64_t modifiers = '
        + (' | '.join(modifiers) if modifiers else '0u')
        + ';'
    ]
    if arch in ('cdna3', 'cdna4') and not is_vop3:
        result_format = 'F32' if to_float else 'NONE'
        lines += [
            '  if (inst_.src0 == amdgpu::SRC_SDWA) {',
            f'    amdgpu::arith::execute_conversion32_sdwa<amdgpu::sdwa::ResultFormat::{result_format}>(*this, wf,',
            f'        {wave_api(api, 32)}, {wave_api(api, 64)}, flags, {dst[0]}, {src[0]});',
            '    return;',
            '  }',
        ]
    if reports:
        lines += [
            f'  amdgpu::arith::execute_float_conversion(wf, {wave_api(api, 32)}, {wave_api(api, 64)},',
            f'      flags, modifiers, {dst[0]}, {src[0]}, /*from_double=*/false, /*to_double=*/false);',
        ]
    else:
        lines += [
            f'  amdgpu::arith::execute_unary32(wf, {wave_api(api, 32)}, {wave_api(api, 64)},',
            f'      flags, modifiers, {dst[0]}, {src[0]});',
        ]
    return '\n'.join(lines)


def gen_conversion64(ctx: ExecuteContext) -> str | None:
    arch = ctx.arch_name
    mnemonic = ctx.inst.name
    dst = ctx.dst_ops
    src = ctx.src_ops
    is_vop3 = ctx.is_vop3
    if (
        arch not in ('rdna3', 'rdna4', 'cdna3', 'cdna4', 'cdna5')
        or mnemonic not in CONVERSION64_NAMES
    ):
        return None
    if len(dst) != 1 or len(src) != 1:
        return None
    output, source = mnemonic.split('_')[-2:]
    from_integer = source in ('I32', 'U32')
    to_double = output == 'F64'
    api = (
        'rj_arith_'
        + mnemonic.lower()
        + ('_gfx12' if arch in ('rdna4', 'cdna5') else '')
    )
    lines = [
        '  uint64_t flags = wf.cu().arithmetic_flags();',
        '  if (wf.ieee_mode()) flags |= RJ_ARITH_FP_IEEE_MODE;',
    ]
    if not from_integer:
        denorm = 'fp_denorm_mode_f16_f64' if source == 'F64' else 'fp_denorm_mode_f32'
        lines += [
            f'  if (!(wf.{denorm}() & 1)) flags |= RJ_ARITH_FP_FLUSH_INPUT_DENORMALS;'
        ]
    if output.startswith('F'):
        denorm = 'fp_denorm_mode_f16_f64' if to_double else 'fp_denorm_mode_f32'
        lines += [
            f'  if (!(wf.{denorm}() & 2)) flags |= RJ_ARITH_FP_FLUSH_OUTPUT_DENORMALS;'
        ]
    if output.startswith('F'):
        rounding = 'fp_round_mode_f32' if output == 'F32' else 'fp_round_mode_f16_f64'
        lines += [f'  flags |= uint64_t(wf.{rounding}()) << 24;']
    if arch in ('rdna3', 'cdna3', 'cdna4') and output.startswith('F'):
        lines += ['  if (!wf.dx10_clamp()) flags |= RJ_ARITH_FP_DX10_CLAMP_OFF;']
    modifiers = '0u'
    if is_vop3:
        modifiers = '(uint64_t(inst_.omod) * RJ_ARITH_ALU_OMOD_2) | (inst_.clamp ? RJ_ARITH_ALU_CLAMP : 0u)'
        if not from_integer:
            modifiers += ' | ((inst_.abs & 1) ? RJ_ARITH_ALU_ABS_A : 0u) | ((inst_.neg & 1) ? RJ_ARITH_ALU_NEG_A : 0u)'
    lines += [f'  const uint64_t modifiers = {modifiers};']
    if from_integer:
        lines += [
            f'  amdgpu::arith::execute_widen32_to_pair(wf, {wave_api(api, 32)}, {wave_api(api, 64)},',
            f'      flags, modifiers, {dst[0]}, {src[0]}, 0xF);',
        ]
    else:
        lines += [
            f'  amdgpu::arith::execute_float_conversion(wf, {wave_api(api, 32)}, {wave_api(api, 64)},',
            f'      flags, modifiers, {dst[0]}, {src[0]}, /*from_double=*/{str(source == "F64").lower()},',
            f'      /*to_double=*/{str(to_double).lower()});',
        ]
    return '\n'.join(lines)


def gen_float16_conversion(ctx: ExecuteContext) -> str | None:
    arch = ctx.arch_name
    name = ctx.inst.name
    if arch not in ('cdna4', 'cdna5') or name not in FLOAT16_CONVERSION_NAMES:
        return None
    widen = name == 'V_CVT_F32_BF16'
    single_sr = name in ('V_CVT_SR_BF16_F32', 'V_CVT_SR_F16_F32')
    packed_sr = name in ('V_CVT_SR_PK_BF16_F32', 'V_CVT_SR_PK_F16_F32')
    if (single_sr and arch != 'cdna4') or (packed_sr and arch != 'cdna5'):
        return None
    operands = 1 if widen else 3 if packed_sr else 2
    if (
        len(ctx.dst_ops) != 1
        or len(ctx.src_ops) != operands
        or (not widen and not ctx.is_vop3)
    ):
        return None
    api = 'rj_arith_' + name.lower()
    selector = 'amdgpu::vop3_opsel(inst_)' if ctx.is_vop3 else '0u'
    mode = (
        f'({selector} & 1) ? RJ_ARITH_ALU_HIGH_A : 0u'
        if widen
        else (f'({selector} & 8) ? RJ_ARITH_ALU_HIGH_D : 0u' if single_sr else '0u')
    )
    if not widen and not packed_sr:
        mode = f'({mode}) | ((inst_.abs & 1) ? RJ_ARITH_ALU_ABS_A : 0u) | ((inst_.neg & 1) ? RJ_ARITH_ALU_NEG_A : 0u)'
        if not single_sr:
            mode += ' | ((inst_.abs & 2) ? RJ_ARITH_ALU_ABS_B : 0u) | ((inst_.neg & 2) ? RJ_ARITH_ALU_NEG_B : 0u)'
    source_operand = ctx.src_ops[0]
    prefix = []
    if widen and not ctx.is_vop3 and arch == 'cdna5':
        source_operand = 'word_source'
        prefix = [f'  auto word_source = amdgpu::e32_word_operand({ctx.src_ops[0]});']
        mode = f'(amdgpu::e32_packed_vgpr_operand({ctx.src_ops[0]}) && ({ctx.src_ops[0]}.encoding_value() & 0x80)) ? RJ_ARITH_ALU_HIGH_A : 0u'
    lines = prefix + [
        '  uint64_t flags = wf.cu().arithmetic_flags();',
        '  if (wf.fp16_ovfl()) flags |= RJ_ARITH_FP16_OVFL;',
    ]
    if arch == 'cdna5' and name == 'V_CVT_PK_F16_F32':
        lines.append(
            '  if (!(wf.fp_denorm_mode_f16_f64() & 2)) flags |= RJ_ARITH_FP_FLUSH_OUTPUT_DENORMALS;'
        )
    lines.append(f'  const uint64_t modifiers = {mode};')
    if widen and not ctx.is_vop3 and arch == 'cdna4':
        lines += [
            '  if (inst_.src0 == amdgpu::SRC_SDWA) {',
            f'    amdgpu::arith::execute_bf16_widen_sdwa(*this, wf, {wave_api(api, 32)}, {wave_api(api, 64)},',
            f'        {ctx.dst_ops[0]}, {ctx.src_ops[0]});',
            '    return;',
            '  }',
        ]
    adapter = (
        'execute_unary32'
        if widen
        else 'execute_ternary32' if packed_sr else 'execute_binary32'
    )
    extra = (
        ', (modifiers & RJ_ARITH_ALU_HIGH_A) ? 0xC : 0x3'
        if widen
        else ', (modifiers & RJ_ARITH_ALU_HIGH_D) ? 0xC : 0x3' if single_sr else ''
    )
    operands = ', '.join(ctx.dst_ops + [source_operand] + ctx.src_ops[1:])
    lines += [
        f'  amdgpu::arith::{adapter}(wf, {wave_api(api, 32)}, {wave_api(api, 64)},',
        f'      flags, modifiers, {operands}{extra});',
    ]
    return '\n'.join(lines)


def gen_fp8_narrow(ctx: ExecuteContext) -> str | None:
    arch = ctx.arch_name
    mnemonic = ctx.inst.name
    dst = ctx.dst_ops
    src = ctx.src_ops
    is_vop3 = ctx.is_vop3
    if (
        arch not in ('rdna4', 'cdna3', 'cdna4', 'cdna5')
        or mnemonic not in FP8_NARROW_NAMES
        or not is_vop3
    ):
        return None
    half = mnemonic.endswith('_F16')
    if half and arch != 'cdna5':
        return None
    stochastic = '_SR_' in mnemonic
    if len(dst) != 1 or len(src) != (1 if half and not stochastic else 2):
        return None
    suffix = ''
    if half:
        suffix = ''
    elif arch == 'cdna3':
        suffix = '_gfx942'
    elif arch in ('cdna4', 'cdna5'):
        suffix = '_gfx1250' if arch == 'cdna5' else '_gfx950'
    api = 'rj_arith_' + mnemonic.lower() + suffix
    selector = 'inst_.opsel' if arch == 'rdna4' else 'amdgpu::vop3_opsel(inst_)'
    lines = [
        '  uint64_t flags = wf.cu().arithmetic_flags();',
        '  if (wf.fp16_ovfl()) flags |= RJ_ARITH_FP16_OVFL;',
        '  uint64_t modifiers = ((inst_.abs & 1) ? RJ_ARITH_ALU_ABS_A : 0u) |',
        '      ((inst_.neg & 1) ? RJ_ARITH_ALU_NEG_A : 0u);',
    ]
    if half:
        lines.remove('  if (wf.fp16_ovfl()) flags |= RJ_ARITH_FP16_OVFL;')
        if stochastic:
            lines += [f'  modifiers |= ({selector} & 1) ? RJ_ARITH_ALU_HIGH_A : 0u;']
    if (half or suffix == '_gfx1250') and '_FP8_' in mnemonic:
        lines += ['  modifiers |= inst_.clamp ? RJ_ARITH_ALU_CLAMP : 0u;']
    if stochastic:
        lines += [
            f'  const unsigned byte = ({selector} >> 2) & 3;',
            '  modifiers |= uint64_t(byte) << 16;',
            '  const uint8_t destination_bytes = uint8_t(1u << byte);',
        ]
    else:
        lines += [
            '  modifiers |= ((inst_.abs & 2) ? RJ_ARITH_ALU_ABS_B : 0u) |',
            '      ((inst_.neg & 2) ? RJ_ARITH_ALU_NEG_B : 0u) |',
            f'      (({selector} & 8) ? RJ_ARITH_ALU_HIGH_D : 0u);',
            f'  const uint8_t destination_bytes = ({selector} & 8) ? 0xC : 0x3;',
        ]
    adapter = 'execute_unary32' if half and not stochastic else 'execute_binary32'
    operands = ', '.join(dst + src)
    source_bytes = ', 0xF' if half and not stochastic else ''
    lines += [
        f'  amdgpu::arith::{adapter}(wf, {wave_api(api, 32)}, {wave_api(api, 64)}, flags, modifiers,',
        f'      {operands}{source_bytes}, destination_bytes);',
    ]
    return '\n'.join(lines)


def gen_fp8_widen(ctx: ExecuteContext) -> str | None:
    arch = ctx.arch_name
    mnemonic = ctx.inst.name
    dst = ctx.dst_ops
    src = ctx.src_ops
    is_vop3 = ctx.is_vop3
    if (
        arch not in ('rdna4', 'cdna3', 'cdna4', 'cdna5')
        or mnemonic not in FP8_WIDEN_NAMES
    ):
        return None
    if len(dst) != 1 or len(src) != 1:
        return None
    half = '_F16_' in mnemonic
    if half and arch != 'cdna5':
        return None
    packed = '_PK_' in mnemonic
    suffix = ''
    if half:
        suffix = ''
    elif arch == 'cdna3':
        suffix = '_gfx942'
    elif arch in ('cdna4', 'cdna5'):
        suffix = (
            '_gfx1250' if arch == 'cdna5' and mnemonic == 'V_CVT_F32_FP8' else '_gfx950'
        )
    api = 'rj_arith_' + mnemonic.lower() + suffix
    selector = 'amdgpu::vop3_opsel(inst_)' if is_vop3 else '0u'
    source_operand, destination_operand = src[0], dst[0]
    lines = []
    if half and not is_vop3:
        operand = src[0] if packed else dst[0]
        bit = 1 if packed else 8
        selector = f'(amdgpu::e32_packed_vgpr_operand({operand}) && ({operand}.encoding_value() & 0x80)) ? {bit}u : 0u'
        lines += [
            f'  auto word_source = amdgpu::e32_word_operand({src[0]});',
            f'  auto word_destination = amdgpu::e32_word_operand({dst[0]});',
        ]
        source_operand, destination_operand = 'word_source', 'word_destination'
    lines += [f'  const unsigned selector = {selector};']
    flags = 'wf.cu().arithmetic_flags()'
    if half:
        lines += [
            '  uint64_t flags = wf.cu().arithmetic_flags();',
            '  if (wf.fp16_ovfl()) flags |= RJ_ARITH_FP16_OVFL;',
        ]
        flags = 'flags'
    if packed:
        lines += [
            '  const uint64_t modifiers = (selector & 1) ? RJ_ARITH_ALU_HIGH_A : 0u;',
            '  const uint8_t source_bytes = (selector & 1) ? 0xC : 0x3;',
        ]
    else:
        lines += [
            '  const unsigned byte = ((selector & 1) << 1) | ((selector & 2) >> 1);',
            '  const uint64_t modifiers = (uint64_t(byte) << 16)'
            + (
                ' | (inst_.clamp ? RJ_ARITH_ALU_CLAMP : 0u)'
                if is_vop3
                and (suffix == '_gfx1250' or (half and mnemonic.endswith('_FP8')))
                else ''
            )
            + (' | ((selector & 8) ? RJ_ARITH_ALU_HIGH_D : 0u)' if half else '')
            + ';',
            '  const uint8_t source_bytes = uint8_t(1u << byte);',
        ]
    if not packed and not is_vop3 and arch in ('cdna3', 'cdna4'):
        lines += [
            '  if (inst_.src0 == amdgpu::SRC_SDWA) {',
            f'    amdgpu::arith::execute_fp8_widen_sdwa(*this, wf, {wave_api(api, 32)}, {wave_api(api, 64)},',
            f'        {dst[0]}, {src[0]});',
            '    return;',
            '  }',
        ]
    adapter = 'execute_widen32_to_pair' if packed and not half else 'execute_unary32'
    destination = ', (selector & 8) ? 0xC : 0x3' if half and not packed else ''
    lines += [
        f'  amdgpu::arith::{adapter}(wf, {wave_api(api, 32)}, {wave_api(api, 64)},',
        f'      {flags}, modifiers,',
        f'      {destination_operand}, {source_operand}, source_bytes{destination});',
    ]
    return '\n'.join(lines)


def gen_byte_nibble_conversion(ctx: ExecuteContext) -> str | None:
    arch = ctx.arch_name
    mnemonic = ctx.inst.name
    dst = ctx.dst_ops
    src = ctx.src_ops
    is_vop3 = ctx.is_vop3
    if (
        arch not in ('rdna3', 'rdna4', 'cdna3', 'cdna4', 'cdna5')
        or mnemonic not in BYTE_NIBBLE_CONVERSION_NAMES
    ):
        return None
    if len(dst) != 1 or len(src) != 1:
        return None
    byte = 0 if mnemonic == 'V_CVT_OFF_F32_I4' else int(mnemonic[-1])
    api = (
        'rj_arith_'
        + mnemonic.lower()
        + ('_gfx12' if arch in ('rdna4', 'cdna5') else '')
    )
    modifiers = (
        '(uint64_t(inst_.omod) * RJ_ARITH_ALU_OMOD_2) | (inst_.clamp ? RJ_ARITH_ALU_CLAMP : 0u)'
        if is_vop3
        else '0u'
    )
    sdwa = []
    if arch in ('cdna3', 'cdna4') and not is_vop3:
        sdwa = [
            '  if (inst_.src0 == amdgpu::SRC_SDWA) {',
            f'    amdgpu::arith::execute_byte_conversion_sdwa(*this, wf, {wave_api(api, 32)}, {wave_api(api, 64)},',
            f'        flags, {dst[0]}, {src[0]});',
            '    return;',
            '  }',
        ]
    return '\n'.join(
        [
            '  uint64_t flags = wf.cu().arithmetic_flags();',
            '  if (wf.ieee_mode()) flags |= RJ_ARITH_FP_IEEE_MODE;',
            '  if (!(wf.fp_denorm_mode_f32() & 2)) flags |= RJ_ARITH_FP_FLUSH_OUTPUT_DENORMALS;',
            f'  const uint64_t modifiers = {modifiers};',
            *sdwa,
            f'  amdgpu::arith::execute_unary32(wf, {wave_api(api, 32)}, {wave_api(api, 64)}, flags, modifiers,',
            f'      {dst[0]}, {src[0]}, {1 << byte});',
        ]
    )


def gen_bit_pack(ctx: ExecuteContext) -> str | None:
    arch = ctx.arch_name
    mnemonic = ctx.inst.name
    dst = ctx.dst_ops
    src = ctx.src_ops
    is_vop3 = ctx.is_vop3
    if mnemonic == 'V_CVT_PKACCUM_U8_F32' and arch in ('cdna3', 'cdna4'):
        if not is_vop3 or len(dst) != 1 or len(src) != 2:
            return None
        return '\n'.join(
            [
                '  amdgpu::arith::execute_byte_pack(wf, rj_arith_v_cvt_pkaccum_u8_f32_w32,',
                '      rj_arith_v_cvt_pkaccum_u8_f32_w64, wf.cu().arithmetic_flags(), 0,',
                f'      {dst[0]}, {src[0]}, {src[1]}, {dst[0]});',
            ]
        )
    if (
        arch not in ('rdna3', 'rdna4', 'cdna3', 'cdna4', 'cdna5')
        or mnemonic not in BIT_PACK_NAMES
        or not is_vop3
    ):
        return None
    half = mnemonic == 'V_PACK_B32_F16'
    if len(dst) != 1 or len(src) != (2 if half else 3):
        return None
    api = (
        'rj_arith_'
        + mnemonic.lower()
        + ('_gfx12' if arch in ('rdna4', 'cdna5') else '')
    )
    lines = ['  uint64_t flags = wf.cu().arithmetic_flags();']
    denorm = 'fp_denorm_mode_f16_f64' if half else 'fp_denorm_mode_f32'
    lines += [
        f'  if (!(wf.{denorm}() & 1)) flags |= RJ_ARITH_FP_FLUSH_INPUT_DENORMALS;'
    ]
    if half:
        lines += ['  if (wf.ieee_mode()) flags |= RJ_ARITH_FP_IEEE_MODE;']
    if not half:
        lines += ['  flags |= uint64_t(wf.fp_round_mode_f32()) << 24;']
    terms = []
    if half:
        lines += ['  const uint32_t opsel = amdgpu::vop3_opsel(inst_);']
    for i, operand in enumerate('AB' if half else 'A'):
        terms += [
            f'((inst_.abs & {1 << i}) ? RJ_ARITH_ALU_ABS_{operand} : 0u)',
            f'((inst_.neg & {1 << i}) ? RJ_ARITH_ALU_NEG_{operand} : 0u)',
        ]
        if half:
            terms += [f'((opsel & {1 << i}) ? RJ_ARITH_ALU_HIGH_{operand} : 0u)']
    if not half:
        terms += ['(inst_.clamp ? RJ_ARITH_ALU_CLAMP : 0u)']
    lines += ['  const uint64_t modifiers = ' + ' | '.join(terms) + ';']
    helper = 'execute_float_pack' if half else 'execute_byte_pack'
    operands = ', '.join(dst + src) + (', /*half_sources=*/true' if half else '')
    lines += [
        f'  amdgpu::arith::{helper}(wf, {wave_api(api, 32)}, {wave_api(api, 64)},',
        f'      flags, modifiers,',
        f'      {operands});',
    ]
    return '\n'.join(lines)


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


def gen_float_fma64(ctx: ExecuteContext) -> str | None:
    """Emit VOP3 FP64 FMA, or decline unsupported architectures/forms."""
    arch = ctx.arch_name
    mnemonic = ctx.inst.name
    dst = ctx.dst_ops
    src = ctx.src_ops
    is_vop3 = ctx.is_vop3
    if (
        arch not in ('rdna2', 'rdna3', 'rdna4')
        or mnemonic != 'V_FMA_F64'
        or not is_vop3
    ):
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
            f'      {dst[0]}, {src[0]}, {src[1]}, {src[2]}, {str(arch != "rdna4").lower()});',
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
    if mnemonic not in BINARY_ARITHMETIC_F32 | SELECTION_F32[arch]:
        return None
    arithmetic = mnemonic in BINARY_ARITHMETIC_F32
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
    controls = f'/*rdna3=*/{str(arch == "rdna3").lower()}, /*compatibility_mul=*/{str(mnemonic == "V_MUL_F32").lower()}'
    status_kind = ''
    if arch == 'rdna4':
        kind = {
            'V_ADD_F32': 'Add',
            'V_SUB_F32': 'Sub',
            'V_SUBREV_F32': 'Subrev',
            'V_MUL_F32': 'Mul',
        }.get(mnemonic)
        if mnemonic in SELECTION_F32['rdna4']:
            kind = 'Select'
        if kind:
            status_kind = f'<::rocjitsu::rdna4::BinaryFpOperation::{kind}>'
    lines += [
        f'  amdgpu::arith::execute_float_binary32{status_kind}(wf, {wave_api(api, 32)}, {wave_api(api, 64)},',
        f'      wf.cu().arithmetic_flags(), {modifiers},',
        f'      {dst[0]}, {src[0]}, {src[1]},',
        f'      {controls});',
    ]
    return '\n'.join(lines)


def gen_float_fma32(ctx: ExecuteContext) -> str | None:
    """Emit FP32 FMA, FMAC and literal forms; decline unsupported forms."""
    arch = ctx.arch_name
    mnemonic = ctx.inst.name
    dst = ctx.dst_ops
    src = ctx.src_ops
    is_vop3 = ctx.is_vop3
    if not (arch == 'rdna2' and mnemonic in GFX10_LEGACY_FMA) and (
        arch not in ('rdna3', 'rdna4') or mnemonic not in FMA_F32
    ):
        return None
    if arch == 'rdna4' and mnemonic == 'V_FMAC_DX9_ZERO_F32':
        return None
    accumulate = mnemonic in ('V_FMAC_F32', 'V_FMAC_DX9_ZERO_F32', 'V_FMAC_LEGACY_F32')
    if len(dst) != 1 or len(src) != (2 if accumulate else 3):
        return None
    # The FMA API explicitly represents the accumulator/literal input, so
    # staging the destination cannot discard its old value.
    inputs = src + dst if accumulate else src
    dx9 = 'DX9_ZERO' in mnemonic or mnemonic in GFX10_LEGACY_FMA
    api = 'rj_arith_v_fma' + ('_dx9_zero' if dx9 else '') + '_f32'
    if arch == 'rdna4':
        api += '_gfx12'
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
        f'  amdgpu::arith::execute_float_fma32<{str(dx9).lower()}>(wf,',
        f'      {wave_api(api, 32)},',
        f'      {wave_api(api, 64)},',
        f'      wf.cu().arithmetic_flags(), {modifiers},',
        f'      {dst[0]}, {", ".join(inputs)},',
        f'      {str(arch != "rdna4").lower()});',
    ]
    return '\n'.join(lines)


def gen_mixed_fma(ctx: ExecuteContext) -> str | None:
    """Emit mixed FP16/FP32 FMA, or decline unsupported forms."""
    arch = ctx.arch_name
    mnemonic = ctx.inst.name
    dst = ctx.dst_ops
    src = ctx.src_ops
    enc_name = ctx.enc_name
    opsel_exprs = ctx.opsel_exprs
    hi_c_expr = ctx.op_sel_hi_2_expr
    bf16 = arch == 'cdna5' and mnemonic in MIXED_BF16_NAMES
    legacy_mad = arch in ('cdna3', 'cdna4') and mnemonic in {
        'V_MAD_MIX_F32',
        'V_MAD_MIXLO_F16',
        'V_MAD_MIXHI_F16',
    }
    if (
        not bf16
        and not legacy_mad
        and (arch not in ('rdna3', 'rdna4', 'cdna5') or mnemonic not in MIXED_FMA_NAMES)
    ):
        return None
    if enc_name != 'ENC_VOP3P' or len(dst) != 1 or len(src) != 3:
        return None
    low, high = opsel_exprs
    terms = ['(inst_.clamp ? RJ_ARITH_ALU_CLAMP : 0u)']
    narrow = []
    for index, source in enumerate('ABC'):
        selected = f'({high} & {1 << index})' if index < 2 else hi_c_expr
        terms += [
            f'({selected} ? RJ_ARITH_MIX_{"BF16" if bf16 else "F16"}_{source} : 0u)',
            f'(({low} & {1 << index}) ? RJ_ARITH_ALU_HIGH_{source} : 0u)',
            f'((inst_.neg & {1 << index}) ? RJ_ARITH_ALU_NEG_{source} : 0u)',
            f'((inst_.neg_hi & {1 << index}) ? RJ_ARITH_ALU_ABS_{source} : 0u)',
        ]
        narrow += [
            f'({selected} && amdgpu::is_inline_float_src(inst_.src{index}) ? {1 << index}u : 0u)'
        ]
    api = (
        'rj_arith_'
        + mnemonic.lower()
        + ('_gfx12' if not bf16 and arch in ('rdna4', 'cdna5') else '')
    )
    byte_mask = (
        '0xF'
        if mnemonic in ('V_FMA_MIX_F32', 'V_MAD_MIX_F32', 'V_FMA_MIX_F32_BF16')
        else '0xC' if 'MIXHI' in mnemonic else '0x3'
    )
    if bf16:
        return '\n'.join(
            [
                '  const uint64_t modifiers = ' + ' |\n      '.join(terms) + ';',
                f'  amdgpu::arith::execute_mixed_bf16(wf, {wave_api(api, 32)},',
                '      wf.cu().arithmetic_flags(), modifiers,',
                f'      {dst[0]}, {", ".join(src)}, {byte_mask});',
            ]
        )
    return '\n'.join(
        [
            '  const uint64_t modifiers = ' + ' |\n      '.join(terms) + ';',
            '  const unsigned narrow_sources = ' + ' | '.join(narrow) + ';',
            f'  amdgpu::arith::execute_mixed_fma(wf, {wave_api(api, 32)}, {wave_api(api, 64)},',
            '      wf.cu().arithmetic_flags(), modifiers,',
            f'      {dst[0]}, {", ".join(src)}, narrow_sources, {byte_mask}, {str(arch in ("rdna3", "cdna3", "cdna4")).lower()}, {str(arch == "cdna5").lower()});',
        ]
    )


def gen_packed_fma16(ctx: ExecuteContext) -> str | None:
    """Emit packed FP16 FMA/FMAC, or decline unsupported forms."""
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
    api = 'rj_arith_v_pk_fma' + ('c' if accumulate else '') + '_f16'
    if arch == 'rdna4':
        api += '_gfx12'
    addend = dst[0] if accumulate else src[2]
    lines += [
        f'  amdgpu::arith::execute_packed_fma16<{str(accumulate).lower()}>(wf,',
        f'      {wave_api(api, 32)},',
        f'      {wave_api(api, 64)},',
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
    if arch not in ('rdna3', 'rdna4', 'cdna3', 'cdna4') or len(dst) != 1:
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
                f'      {dst[0]}, {", ".join(src)}, 0xF, 0xF, wf.exec() & wf.vgpr_write_mask());',
            ]
        )
    if mnemonic not in DOT2_F32_NAMES:
        return None
    cdna = arch in ('cdna3', 'cdna4')
    cdna_accumulate = mnemonic in ('V_DOT2C_F32_F16', 'V_DOT2C_F32_BF16')
    if cdna_accumulate and not cdna:
        return None
    if arch == 'cdna3' and mnemonic.endswith('_BF16'):
        return None
    accumulate = mnemonic == 'V_DOT2ACC_F32_F16' or cdna_accumulate
    if cdna_accumulate:
        if enc_name not in ('ENC_VOP2', 'ENC_VOP3') or len(src) != 2:
            return None
    elif accumulate:
        if arch != 'rdna3' or enc_name != 'ENC_VOP2' or len(src) != 2:
            return None
    elif enc_name != 'ENC_VOP3P' or len(src) != 3:
        return None
    bf16 = mnemonic.endswith('_BF16')
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
    for index in range(0 if cdna_accumulate else 1 if accumulate else 2):
        selector = f'inst_.src{index}'
        if bf16 and not cdna:
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
        + (
            ('_gfx950' if bf16 else '_gfx942')
            if cdna
            else '_gfx12' if arch == 'rdna4' else ''
        )
    )
    addend = dst[0] if accumulate else src[2]
    return '\n'.join(
        [
            '  const uint64_t modifiers = '
            + (' |\n      '.join(terms) if terms else '0')
            + ';',
            '  const unsigned narrow_sources = '
            + (' | '.join(narrow) if narrow else '0')
            + ';',
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
    if not (arch == 'rdna2' and mnemonic in GFX10_BINARY_F16) and (
        arch not in ('rdna3', 'rdna4') or mnemonic not in BINARY_F16
    ):
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
            (
                '  const uint32_t opsel = 0;'
                if arch == 'rdna2'
                else '  const uint32_t opsel = amdgpu::vop3_opsel(inst_);'
            ),
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
    status_operation = ''
    if arch == 'rdna2':
        operation = mnemonic.split('_')[1].title()
        status_operation = f'<amdgpu::arith::HalfBinaryOperation::{operation}>'
    if arch == 'rdna2' and not is_vop3:
        lines += [
            '  if (inst_.src0 == amdgpu::SRC_SDWA) {',
            f'    amdgpu::arith::execute_sdwa_float_binary16{status_operation}(*this, wf, {wave_api(api, 32)}, {wave_api(api, 64)},',
            f'        {dst[0]}, {src[0]}, {src[1]});',
            '    return;',
            '  }',
        ]
    lines += [
        f'  amdgpu::arith::execute_float_binary16{status_operation}(wf, {wave_api(api, 32)}, {wave_api(api, 64)},',
        f'      wf.cu().arithmetic_flags(), {modifiers},',
        f'      {dst[0]}, {src[0]}, {src[1]}, {str(arch != "rdna4").lower()});',
    ]
    return '\n'.join(lines)


def gen_float_fma16(ctx: ExecuteContext) -> str | None:
    """Emit FP16 FMA, FMAC and literal forms; decline unsupported forms."""
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
