# Copyright (c) 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""rj-arith comparison bindings with decoder-owned mask result commits."""

from . import ExecuteContext
from .rj_arith_names import wave_api

INTEGER_COMPARISONS = {}


for prefix in ('V_CMP', 'V_CMPX'):
    for dtype in ('I16', 'U16', 'I32', 'U32', 'I64', 'U64'):
        bits = int(dtype[1:])
        for predicate in ('LT', 'EQ', 'LE', 'GT', 'NE', 'GE'):
            INTEGER_COMPARISONS[f'{prefix}_{predicate}_{dtype}'] = (bits, False)
        if bits != 16:
            for predicate in ('F', 'T'):
                INTEGER_COMPARISONS[f'{prefix}_{predicate}_{dtype}'] = (bits, True)


def mask_result_writer(mnemonic, dst, result_writer):
    if result_writer is None:
        if mnemonic.startswith('V_CMPX_'):
            write = 'wf.set_exec(result);'
        elif dst:
            write = f'amdgpu::write_wave_mask_scalar({dst[0]}, wf, result);'
        else:
            write = 'wf.set_vcc_mask(result);'
        result_writer = f'[&](uint64_t result) {{ {write} }}'
    return result_writer


def gen_integer_compare(ctx: ExecuteContext) -> str | None:
    arch, mnemonic = ctx.arch_name, ctx.inst.name
    dst, src = ctx.dst_ops, ctx.src_ops
    is_vop3, result_writer = ctx.is_vop3, ctx.result_writer
    if arch not in ('rdna3', 'rdna4') or mnemonic not in INTEGER_COMPARISONS:
        return None
    if len(src) != 2 or len(dst) > 1:
        return None
    bits, constant = INTEGER_COMPARISONS[mnemonic]
    lines = []
    modifiers = '0'
    if bits == 16 and is_vop3:
        lines += [
            '  const uint32_t opsel = amdgpu::vop3_opsel(inst_);',
            '  const uint64_t modifiers = ((opsel & 1) ? RJ_ARITH_ALU_HIGH_A : 0u) |',
            '                             ((opsel & 2) ? RJ_ARITH_ALU_HIGH_B : 0u);',
        ]
        modifiers = 'modifiers'
    result_writer = mask_result_writer(mnemonic, dst, result_writer)
    api = 'rj_arith_' + mnemonic.lower()
    lines += [
        f'  amdgpu::arith::execute_compare<{bits}>(wf, {wave_api(api, 32)}, {wave_api(api, 64)},',
        f'      wf.cu().arithmetic_flags(), {modifiers},',
        f'      {src[0]}, {src[1]}, {str(constant).lower()}, {result_writer});',
    ]
    return '\n'.join(lines)


FLOAT_COMPARISONS = {
    f'{prefix}_{predicate}_F{bits}': (bits, predicate in ('F', 'T'))
    for prefix in ('V_CMP', 'V_CMPX')
    for bits in (16, 32, 64)
    for predicate in (
        'F',
        'LT',
        'EQ',
        'LE',
        'GT',
        'LG',
        'GE',
        'O',
        'U',
        'NGE',
        'NLG',
        'NGT',
        'NLE',
        'NEQ',
        'NLT',
        'T',
    )
}


CLASS_COMPARISONS = {
    f'{prefix}_CLASS_F{bits}': bits
    for prefix in ('V_CMP', 'V_CMPX')
    for bits in (16, 32, 64)
}


def gen_float_compare(ctx: ExecuteContext) -> str | None:
    arch, mnemonic = ctx.arch_name, ctx.inst.name
    dst, src = ctx.dst_ops, ctx.src_ops
    is_vop3, result_writer = ctx.is_vop3, ctx.result_writer
    if arch not in ('rdna3', 'rdna4') or mnemonic not in FLOAT_COMPARISONS:
        return None
    if len(src) != 2 or len(dst) > 1:
        return None
    bits, constant = FLOAT_COMPARISONS[mnemonic]
    # GFX12 removed the constant predicates; their rj-arith APIs describe RDNA3.
    if arch == 'rdna4' and constant:
        return None
    lines = []
    modifiers = '0'
    if is_vop3:
        lines += [
            '  uint64_t modifiers = ((inst_.abs & 1) ? RJ_ARITH_ALU_ABS_A : 0u) |',
            '                       ((inst_.abs & 2) ? RJ_ARITH_ALU_ABS_B : 0u) |',
            '                       ((inst_.neg & 1) ? RJ_ARITH_ALU_NEG_A : 0u) |',
            '                       ((inst_.neg & 2) ? RJ_ARITH_ALU_NEG_B : 0u) |',
            '                       (inst_.clamp ? RJ_ARITH_ALU_CLAMP : 0u);',
        ]
        if bits == 16:
            lines += [
                '  const uint32_t opsel = amdgpu::vop3_opsel(inst_);',
                '  modifiers |= ((opsel & 1) ? RJ_ARITH_ALU_HIGH_A : 0u) |',
                '               ((opsel & 2) ? RJ_ARITH_ALU_HIGH_B : 0u);',
            ]
        modifiers = 'modifiers'
    result_writer = mask_result_writer(mnemonic, dst, result_writer)
    api = 'rj_arith_' + mnemonic.lower() + ('_gfx12' if arch == 'rdna4' else '')
    lines += [
        f'  amdgpu::arith::execute_float_compare<{bits}>(wf, {wave_api(api, 32)}, {wave_api(api, 64)},',
        f'      wf.cu().arithmetic_flags(), {modifiers},',
        f'      {src[0]}, {src[1]}, {str(constant).lower()},',
        f'      {result_writer}, exec_mask);',
    ]
    return '\n'.join(lines)


def gen_class_compare(ctx: ExecuteContext) -> str | None:
    arch, mnemonic = ctx.arch_name, ctx.inst.name
    dst, src = ctx.dst_ops, ctx.src_ops
    is_vop3, result_writer = ctx.is_vop3, ctx.result_writer
    if arch not in ('rdna3', 'rdna4') or mnemonic not in CLASS_COMPARISONS:
        return None
    if len(src) != 2 or len(dst) > 1:
        return None
    bits = CLASS_COMPARISONS[mnemonic]
    lines = []
    modifiers = '0'
    if is_vop3:
        lines += [
            '  uint64_t modifiers = ((inst_.abs & 1) ? RJ_ARITH_ALU_ABS_A : 0u) |',
            '                       ((inst_.neg & 1) ? RJ_ARITH_ALU_NEG_A : 0u) |',
            '                       (inst_.clamp ? RJ_ARITH_ALU_CLAMP : 0u);',
        ]
        if bits == 16:
            lines += [
                '  const uint32_t opsel = amdgpu::vop3_opsel(inst_);',
                '  modifiers |= ((opsel & 1) ? RJ_ARITH_ALU_HIGH_A : 0u) |',
                '               ((opsel & 2) ? RJ_ARITH_ALU_HIGH_B : 0u);',
            ]
        modifiers = 'modifiers'
    api = 'rj_arith_' + mnemonic.lower()
    result_writer = mask_result_writer(mnemonic, dst, result_writer)
    lines += [
        f'  amdgpu::arith::execute_compare<{bits}, 32>(wf, {wave_api(api, 32)}, {wave_api(api, 64)},',
        f'      wf.cu().arithmetic_flags(), {modifiers},',
        f'      {src[0]}, {src[1]}, false, {result_writer});',
    ]
    return '\n'.join(lines)


COMPARE_NAMES = (
    set(INTEGER_COMPARISONS) | set(FLOAT_COMPARISONS) | set(CLASS_COMPARISONS)
)
