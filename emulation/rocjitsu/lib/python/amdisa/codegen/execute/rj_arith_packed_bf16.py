# Copyright (c) 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""gfx1250 packed BF16 arithmetic; inline constants remain raw packed words."""

from . import ExecuteContext

PACKED_BF16_NAMES = {
    'V_PK_ADD_BF16',
    'V_PK_MUL_BF16',
    'V_PK_FMA_BF16',
    'V_PK_MIN_NUM_BF16',
    'V_PK_MAX_NUM_BF16',
}


def gen_packed_bf16(ctx: ExecuteContext) -> str | None:
    if ctx.arch_name != 'cdna5' or ctx.inst.name not in PACKED_BF16_NAMES:
        return None
    ternary = ctx.inst.name == 'V_PK_FMA_BF16'
    count = 3 if ternary else 2
    if (
        ctx.enc_name != 'ENC_VOP3P'
        or len(ctx.dst_ops) != 1
        or len(ctx.src_ops) != count
    ):
        return None
    low, high = ctx.opsel_exprs
    terms = ['(inst_.clamp ? RJ_ARITH_PK_CLAMP : 0u)']
    for i, letter in enumerate('ABC'[:count]):
        hi = f'({high} & {1 << i})' if i < 2 else ctx.op_sel_hi_2_expr
        terms += [
            f'(({low} & {1 << i}) ? RJ_ARITH_PK_LO_{letter}_HIGH : 0u)',
            f'({hi} ? 0u : RJ_ARITH_PK_HI_{letter}_LOW)',
            f'((inst_.neg & {1 << i}) ? RJ_ARITH_PK_NEG_LO_{letter} : 0u)',
            f'((inst_.neg_hi & {1 << i}) ? RJ_ARITH_PK_NEG_HI_{letter} : 0u)',
        ]
    third = '&' + ctx.src_ops[2] if ternary else 'nullptr'
    return '\n'.join(
        [
            '  const uint64_t modifiers = ' + ' | '.join(terms) + ';',
            f'  amdgpu::arith::execute_packed_bf16<{str(ternary).lower()}>(wf, rj_arith_{ctx.inst.name.lower()}_w32,',
            f'      modifiers, {ctx.dst_ops[0]}, {ctx.src_ops[0]}, {ctx.src_ops[1]}, {third});',
        ]
    )
