# Copyright (c) 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Register-pair move bindings; generated modifier wrappers retain DPP/SDWA routing."""

from . import ExecuteContext

PAIR_MOVE_NAMES = {'V_MOV_B64', 'V_PK_MOV_B32'}


def gen_pair_move(ctx: ExecuteContext) -> str | None:
    if (
        ctx.arch_name not in ('cdna3', 'cdna4', 'cdna5')
        or ctx.inst.name not in PAIR_MOVE_NAMES
    ):
        return None
    packed = ctx.inst.name == 'V_PK_MOV_B32'
    if len(ctx.dst_ops) != 1 or len(ctx.src_ops) != (2 if packed else 1):
        return None
    if packed:
        low, _ = ctx.opsel_exprs
        modifiers = (
            f'(({low} & 1) ? RJ_ARITH_PK_LO_A_HIGH : 0u) | '
            f'(({low} & 2) ? RJ_ARITH_PK_LO_B_HIGH : 0u)'
        )
    else:
        modifiers = '0'
    second = '&' + ctx.src_ops[1] if packed else 'nullptr'
    return (
        f'  amdgpu::arith::execute_pair_move<{str(packed).lower()}>(wf, {modifiers}, '
        f'{ctx.dst_ops[0]}, {ctx.src_ops[0]}, {second});'
    )
