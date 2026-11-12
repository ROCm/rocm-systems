# Copyright (c) 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

from . import ExecuteContext
from .rj_arith_names import wave_api

CUBE_NAMES = {'V_CUBEID_F32', 'V_CUBESC_F32', 'V_CUBETC_F32', 'V_CUBEMA_F32'}

INTERP_NAMES = {
    'V_INTERP_P10_F32',
    'V_INTERP_P2_F32',
    'V_INTERP_P10_F16_F32',
    'V_INTERP_P2_F16_F32',
    'V_INTERP_P10_RTZ_F16_F32',
    'V_INTERP_P2_RTZ_F16_F32',
}


def gen_interp(ctx: ExecuteContext) -> str | None:
    arch, mnemonic = ctx.arch_name, ctx.inst.name
    dst, src, enc_name = ctx.dst_ops, ctx.src_ops, ctx.enc_name
    if arch not in ('rdna3', 'rdna4') or mnemonic not in INTERP_NAMES:
        return None
    if enc_name != 'ENC_VINTERP' or len(dst) != 1 or len(src) != 3:
        return None
    half = 'F16' in mnemonic
    p2 = '_P2_' in mnemonic
    api = 'rj_arith_' + mnemonic.lower()
    terms = [
        f'((inst_.neg & {1 << i}) ? RJ_ARITH_ALU_NEG_{letter} : 0u)'
        for i, letter in enumerate('ABC')
    ]
    terms += [
        '(inst_.clamp ? RJ_ARITH_ALU_CLAMP : 0u)',
    ]
    if half:
        opsel = 'inst_.op_sel' if arch == 'rdna3' else 'inst_.opsel'
        terms += [f'(({opsel} & 1) ? RJ_ARITH_ALU_HIGH_A : 0u)']
        terms += [
            f'(({opsel} & {8 if p2 else 4}) ? RJ_ARITH_ALU_HIGH_{"D" if p2 else "C"} : 0u)'
        ]
    return '\n'.join(
        [
            '  const uint64_t modifiers = ' + ' | '.join(terms) + ';',
            f'  amdgpu::arith::execute_interp<{str(p2).lower()}, {str(half).lower()}>(',
            f'      wf, {wave_api(api, 32)}, {wave_api(api, 64)},',
            '      wf.cu().arithmetic_flags(), modifiers,',
            f'      {dst[0]}, {", ".join(src)});',
        ]
    )


def gen_cube(ctx: ExecuteContext) -> str | None:
    arch = ctx.arch_name
    mnemonic = ctx.inst.name
    dst = ctx.dst_ops
    src = ctx.src_ops
    is_vop3 = ctx.is_vop3
    if arch not in ('rdna3', 'rdna4') or mnemonic not in CUBE_NAMES or not is_vop3:
        return None
    if len(dst) != 1 or len(src) != 3:
        return None
    api = 'rj_arith_' + mnemonic.lower() + ('_gfx12' if arch == 'rdna4' else '')
    lines = [
        '  uint64_t flags = wf.cu().arithmetic_flags();',
        '  flags |= uint64_t(wf.fp_round_mode_f32()) << 24;',
        '  if (wf.ieee_mode()) flags |= RJ_ARITH_FP_IEEE_MODE;',
    ]
    if arch == 'rdna3':
        lines += ['  if (!wf.dx10_clamp()) flags |= RJ_ARITH_FP_DX10_CLAMP_OFF;']
    lines += [
        '  const uint64_t modifiers = ((inst_.abs & 1) ? RJ_ARITH_ALU_ABS_A : 0u) |',
        '      ((inst_.abs & 2) ? RJ_ARITH_ALU_ABS_B : 0u) |',
        '      ((inst_.abs & 4) ? RJ_ARITH_ALU_ABS_C : 0u) |',
        '      ((inst_.neg & 1) ? RJ_ARITH_ALU_NEG_A : 0u) |',
        '      ((inst_.neg & 2) ? RJ_ARITH_ALU_NEG_B : 0u) |',
        '      ((inst_.neg & 4) ? RJ_ARITH_ALU_NEG_C : 0u) |',
        '      (uint64_t(inst_.omod) * RJ_ARITH_ALU_OMOD_2) |',
        '      (inst_.clamp ? RJ_ARITH_ALU_CLAMP : 0u);',
        f'  amdgpu::arith::execute_ternary32(wf, {wave_api(api, 32)}, {wave_api(api, 64)}, flags, modifiers,',
        f'      {dst[0]}, {src[0]}, {src[1]}, {src[2]}, 0xF, 0xF,',
        '      wf.exec() & wf.vgpr_write_mask());',
    ]
    return '\n'.join(lines)


def gen_mullit(ctx: ExecuteContext) -> str | None:
    arch = ctx.arch_name
    mnemonic = ctx.inst.name
    dst = ctx.dst_ops
    src = ctx.src_ops
    is_vop3 = ctx.is_vop3
    if arch not in ('rdna3', 'rdna4') or mnemonic != 'V_MULLIT_F32' or not is_vop3:
        return None
    if len(dst) != 1 or len(src) != 3:
        return None
    api = 'rj_arith_v_mullit_f32' + ('_gfx12' if arch == 'rdna4' else '')
    terms = []
    for i, letter in enumerate('ABC'):
        terms += [
            f'((inst_.abs & {1 << i}) ? RJ_ARITH_ALU_ABS_{letter} : 0u)',
            f'((inst_.neg & {1 << i}) ? RJ_ARITH_ALU_NEG_{letter} : 0u)',
        ]
    terms += [
        '(uint64_t(inst_.omod) * RJ_ARITH_ALU_OMOD_2)',
        '(inst_.clamp ? RJ_ARITH_ALU_CLAMP : 0u)',
    ]
    return '\n'.join(
        [
            '  const uint64_t modifiers = ' + ' | '.join(terms) + ';',
            f'  amdgpu::arith::execute_float_ternary32(wf, {wave_api(api, 32)}, {wave_api(api, 64)},',
            '      wf.cu().arithmetic_flags(), modifiers,',
            f'      {dst[0]}, {", ".join(src)}, {str(arch == "rdna3").lower()});',
        ]
    )
