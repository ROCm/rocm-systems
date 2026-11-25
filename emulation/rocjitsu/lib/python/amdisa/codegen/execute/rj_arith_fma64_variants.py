# Copyright (c) 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""CDNA FP64 accumulating and literal forms of fused multiply-add."""

from . import ExecuteContext

FMA64_VARIANTS = {'V_FMAC_F64', 'V_FMAMK_F64', 'V_FMAAK_F64'}


def gen_fma64_variants(ctx: ExecuteContext) -> str | None:
    if (
        ctx.arch_name not in ('cdna3', 'cdna4', 'cdna5')
        or ctx.inst.name not in FMA64_VARIANTS
    ):
        return None
    if len(ctx.dst_ops) != 1:
        return None
    if ctx.inst.name != 'V_FMAC_F64':
        if ctx.arch_name != 'cdna5' or ctx.is_vop3 or len(ctx.src_ops) != 3:
            return None
        multiply = ctx.inst.name == 'V_FMAMK_F64'
        # Semantic operand order is A,K,B for FMAMK and A,B,K for FMAAK.
        a, second, third = ctx.src_ops
        b, literal = (third, second) if multiply else (second, third)
        return f'  amdgpu::arith::execute_literal_fma64<{str(multiply).lower()}>(wf, {ctx.dst_ops[0]}, {a}, {b}, {literal});'
    if len(ctx.src_ops) != 2:
        return None
    lines = []
    modifiers = '0'
    if ctx.is_vop3:
        lines = [
            '  const uint64_t modifiers = ((inst_.abs & 1) ? RJ_ARITH_ALU_ABS_A : 0u) |',
            '      ((inst_.abs & 2) ? RJ_ARITH_ALU_ABS_B : 0u) |',
            '      ((inst_.neg & 1) ? RJ_ARITH_ALU_NEG_A : 0u) |',
            '      ((inst_.neg & 2) ? RJ_ARITH_ALU_NEG_B : 0u) |',
            '      (uint64_t(inst_.omod) * RJ_ARITH_ALU_OMOD_2) |',
            '      (inst_.clamp ? RJ_ARITH_ALU_CLAMP : 0u);',
        ]
        modifiers = 'modifiers'
    lines += [
        f'  amdgpu::arith::execute_fmac64<{str(ctx.arch_name != "cdna5").lower()}>(wf, {modifiers}, {ctx.dst_ops[0]}, {", ".join(ctx.src_ops)});'
    ]
    return '\n'.join(lines)
