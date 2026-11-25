# Copyright (c) 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Bindings for CDNA packed FP32 arithmetic with whole-word selectors."""

from . import ExecuteContext


def gen_packed_f32(ctx: ExecuteContext) -> str | None:
    """Map the paired arithmetic operands and independent low/high modifiers."""
    if (
        ctx.arch_name not in ('cdna3', 'cdna4', 'cdna5')
        or ctx.inst.name not in ('V_PK_FMA_F32', 'V_PK_ADD_F32', 'V_PK_MUL_F32')
        or ctx.enc_name != 'ENC_VOP3P'
        or len(ctx.dst_ops) != 1
        or len(ctx.src_ops) != (3 if ctx.inst.name == 'V_PK_FMA_F32' else 2)
    ):
        return None
    low, high = ctx.opsel_exprs
    terms = ['(inst_.clamp ? RJ_ARITH_PK_CLAMP : 0u)']
    for index, source in enumerate('ABC'[: len(ctx.src_ops)]):
        selected = f'({high} & {1 << index})' if index < 2 else ctx.op_sel_hi_2_expr
        terms += [
            f'(({low} & {1 << index}) ? RJ_ARITH_PK_LO_{source}_HIGH : 0u)',
            f'({selected} ? 0u : RJ_ARITH_PK_HI_{source}_LOW)',
            f'((inst_.neg & {1 << index}) ? RJ_ARITH_PK_NEG_LO_{source} : 0u)',
            f'((inst_.neg_hi & {1 << index}) ? RJ_ARITH_PK_NEG_HI_{source} : 0u)',
        ]
    old = ctx.arch_name != 'cdna5'
    arch = 'gfx942' if old else 'gfx1250'
    operation = ctx.inst.name.lower()
    wave32 = f'rj_arith_{operation}_w32_{arch}'
    wave64 = f'rj_arith_{operation}_w64_gfx942' if old else 'nullptr'
    helper = (
        'execute_packed_fma32' if len(ctx.src_ops) == 3 else 'execute_packed_binary32'
    )
    operands = ', '.join([ctx.dst_ops[0], *ctx.src_ops])
    return (
        '  const uint64_t modifiers = ' + ' |\n      '.join(terms) + ';\n'
        f'  amdgpu::arith::{helper}(wf, ' + wave32 + ', ' + wave64 + ',\n'
        '      wf.cu().arithmetic_flags(), modifiers, '
        + operands
        + ', '
        + ('true' if old else 'false')
        + ');'
    )
