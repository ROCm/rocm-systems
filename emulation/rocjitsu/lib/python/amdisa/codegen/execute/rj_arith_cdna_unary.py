# Copyright (c) 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""CDNA integral rounding and mantissa extraction, including legacy SDWA stores."""

from . import ExecuteContext
from .rj_arith_names import wave_api

CDNA_INTEGRAL_UNARY = {
    f'V_{op}_F{bits}'
    for op in ('TRUNC', 'CEIL', 'FLOOR', 'RNDNE', 'FREXP_MANT', 'FRACT')
    for bits in (16, 32)
}


def gen_cdna_integral_unary(ctx: ExecuteContext) -> str | None:
    if (
        ctx.arch_name not in ('cdna3', 'cdna4', 'cdna5')
        or ctx.inst.name not in CDNA_INTEGRAL_UNARY
    ):
        return None
    if len(ctx.dst_ops) != 1 or len(ctx.src_ops) != 1:
        return None
    half = ctx.inst.name.endswith('_F16')
    old = ctx.arch_name != 'cdna5'
    api = 'rj_arith_' + ctx.inst.name.lower() + ('_gfx942' if old else '_gfx12')
    lines = []
    modifiers = '0'
    if ctx.is_vop3:
        lines = [
            '  const uint64_t modifiers = ((inst_.abs & 1) ? RJ_ARITH_ALU_ABS_A : 0u) |',
            '      ((inst_.neg & 1) ? RJ_ARITH_ALU_NEG_A : 0u) |',
            '      (uint64_t(inst_.omod) * RJ_ARITH_ALU_OMOD_2) |',
            '      (inst_.clamp ? RJ_ARITH_ALU_CLAMP : 0u)',
        ]
        if half:
            lines[-1] += ' |'
            lines += [
                '      ((amdgpu::vop3_opsel(inst_) & 1) ? RJ_ARITH_ALU_HIGH_A : 0u) |',
                '      ((amdgpu::vop3_opsel(inst_) & 8) ? RJ_ARITH_ALU_HIGH_D : 0u);',
            ]
        else:
            lines[-1] += ';'
        modifiers = 'modifiers'
    lines += [
        f'  amdgpu::arith::execute_cdna_integral_unary<{str(half).lower()}, {str(old).lower()}>(*this, wf,',
        f'      {wave_api(api, 32)}, {wave_api(api, 64)}, wf.cu().arithmetic_flags(), {modifiers},',
        f'      {ctx.dst_ops[0]}, {ctx.src_ops[0]});',
    ]
    return '\n'.join(lines)
