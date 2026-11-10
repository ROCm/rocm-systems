# Copyright (c) 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""GFX11.5 and GFX12 scalar floating-point operations on SGPRs, independent of EXEC."""

from . import ExecuteContext


def gen_scalar_fp(ctx: ExecuteContext) -> str | None:
    if ctx.arch_name not in ('rdna3_5', 'rdna4'):
        return None
    parts = ctx.inst.name.split('_')
    if parts[0] != 'S' or parts[-1] not in ('F16', 'F32', 'I32', 'U32'):
        return None
    op = parts[1]
    fmt = parts[-1]
    if op == 'CMP' and fmt.startswith('F'):
        rel = parts[2].capitalize()
        body = f'compare<amdgpu::scalar_fp::{fmt}, amdgpu::comparison::{rel}>(wf, ssrc0, ssrc1)'
    elif op in ('ADD', 'SUB', 'MUL', 'FMAC', 'FMAAK', 'FMAMK') and fmt.startswith('F'):
        if op == 'FMAAK':
            operands = 'ssrc0, ssrc1, &simm32'
        elif op == 'FMAMK':
            operands = 'ssrc0, simm32, &ssrc1'
        elif op == 'FMAC':
            operands = 'ssrc0, ssrc1, &sdst'
        else:
            operands = 'ssrc0, ssrc1'
        kind = 'Fma' if op.startswith('FMA') else op.capitalize()
        body = f'arithmetic<amdgpu::scalar_fp::{fmt}, amdgpu::scalar_fp::Op::{kind}>(wf, sdst, {operands})'
    elif op in ('MIN', 'MAX', 'MINIMUM', 'MAXIMUM') and fmt.startswith('F'):
        kind = ''.join(part.capitalize() for part in parts[1:-1])
        if ctx.arch_name == 'rdna3_5':
            kind = 'Legacy' + kind
        body = f'select<amdgpu::scalar_fp::{fmt}, amdgpu::minmax::{kind}>(wf, sdst, ssrc0, ssrc1)'
    elif op in ('CEIL', 'FLOOR', 'TRUNC', 'RNDNE') and fmt.startswith('F'):
        body = f'round_integral<amdgpu::scalar_fp::{fmt}, amdgpu::scalar_fp::Op::{op.capitalize()}>(wf, sdst, ssrc0)'
    elif op == 'CVT':
        if parts[2] == 'PK':
            body = 'pack_rtz(wf, sdst, ssrc0, ssrc1)'
        elif parts[-1] in ('F16', 'F32', 'I32', 'U32') and parts[-2] in (
            'F16',
            'F32',
            'I32',
            'U32',
        ):
            high = ', true' if parts[2] == 'HI' else ''
            body = f'convert<amdgpu::scalar_fp::{parts[-1]}, amdgpu::scalar_fp::{parts[-2]}>(wf, sdst, ssrc0{high})'
        else:
            return None
    else:
        return None
    return f'  amdgpu::scalar_fp::{body};'
