# Copyright (c) 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

from . import ExecuteContext
from .rj_arith_names import wave_api

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
        '(uint64_t(inst_.wait_exp) << RJ_ARITH_INTERP_WAIT_EXP_SHIFT)',
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
