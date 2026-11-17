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
        for predicate in ('F', 'T'):
            INTEGER_COMPARISONS[f'{prefix}_{predicate}_{dtype}'] = (bits, True)


def mask_result_writer(mnemonic, dst, result_writer, arch):
    if arch in ('cdna3', 'cdna4') and mnemonic.startswith('V_CMPX_'):
        # CDNA3/4 CMPX writes the scalar mask destination as well as EXEC.
        # Keep the decoder's EXEC writer, which also applies modifier masks.
        mask_write = (
            f'amdgpu::write_wave_mask_scalar({dst[0]}, wf, result);'
            if dst
            else 'wf.set_vcc_mask(result);'
        )
        exec_write = (
            f'{result_writer}(result);' if result_writer else 'wf.set_exec(result);'
        )
        result_writer = f'[&](uint64_t result) {{ {mask_write} {exec_write} }}'
        return result_writer
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
    if (
        arch not in ('rdna3', 'rdna4', 'cdna3', 'cdna4', 'cdna5')
        or mnemonic not in INTEGER_COMPARISONS
    ):
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
    result_writer = mask_result_writer(mnemonic, dst, result_writer, arch)
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


# CDNA spells the floating always-true predicate TRU rather than T.
for prefix in ('V_CMP', 'V_CMPX'):
    for bits in (16, 32, 64):
        FLOAT_COMPARISONS[f'{prefix}_TRU_F{bits}'] = (bits, True)


CLASS_COMPARISONS = {
    f'{prefix}_CLASS_F{bits}': bits
    for prefix in ('V_CMP', 'V_CMPX')
    for bits in (16, 32, 64)
}


def gen_float_compare(ctx: ExecuteContext) -> str | None:
    arch, mnemonic = ctx.arch_name, ctx.inst.name
    dst, src = ctx.dst_ops, ctx.src_ops
    is_vop3, result_writer = ctx.is_vop3, ctx.result_writer
    if (
        arch not in ('rdna2', 'rdna3', 'rdna3_5', 'rdna4', 'cdna3', 'cdna4', 'cdna5')
        or mnemonic not in FLOAT_COMPARISONS
    ):
        return None
    # TRU aliases are CDNA-only; RDNA2 retains its legacy constant predicate
    # handler, including strict semantics and CLAMP's signaling NaN status.
    if '_TRU_' in mnemonic and not arch.startswith('cdna'):
        return None
    if arch == 'rdna2' and not is_vop3:
        return None
    if len(src) != 2 or len(dst) > 1:
        return None
    bits, constant = FLOAT_COMPARISONS[mnemonic]
    if arch == 'rdna3_5':
        relation = mnemonic.split('_')[2].title()
        # Compact operands already select and normalize their encoded half.
        selectors = (
            'amdgpu::vop3_opsel(inst_), inst_.abs, inst_.neg' if is_vop3 else '0, 0, 0'
        )
        signaling = 'inst_.clamp' if is_vop3 else 'false'
        writer = mask_result_writer(mnemonic, dst, result_writer, arch)
        return (
            f'  amdgpu::execute_compare_with_exceptions<amdgpu::fp_format::F{bits}, '
            f'amdgpu::comparison::{relation}>(wf, {src[0]}, {src[1]}, {selectors}, '
            f'exec_mask, !wf.dx10_clamp(), {signaling}, {writer});'
        )
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
        if bits == 16 and arch != 'rdna2':
            lines += [
                '  const uint32_t opsel = amdgpu::vop3_opsel(inst_);',
                '  modifiers |= ((opsel & 1) ? RJ_ARITH_ALU_HIGH_A : 0u) |',
                '               ((opsel & 2) ? RJ_ARITH_ALU_HIGH_B : 0u);',
            ]
        modifiers = 'modifiers'
    result_writer = mask_result_writer(mnemonic, dst, result_writer, arch)
    # CDNA retains the inherited numerical model without claiming gfx12 FP status.
    api = (
        'rj_arith_'
        + mnemonic.lower().replace('_tru_', '_t_')
        + ('_gfx12' if arch == 'rdna4' else '')
    )
    signaling = ', true' if arch == 'rdna2' else ''
    lines += [
        f'  amdgpu::arith::execute_float_compare<{bits}{signaling}>(wf, {wave_api(api, 32)}, {wave_api(api, 64)},',
        f'      wf.cu().arithmetic_flags(), {modifiers},',
        f'      {src[0]}, {src[1]}, {str(constant).lower()},',
        f'      {result_writer}, exec_mask);',
    ]
    return '\n'.join(lines)


def gen_class_compare(ctx: ExecuteContext) -> str | None:
    arch, mnemonic = ctx.arch_name, ctx.inst.name
    dst, src = ctx.dst_ops, ctx.src_ops
    is_vop3, result_writer = ctx.is_vop3, ctx.result_writer
    if (
        arch not in ('rdna3', 'rdna4', 'cdna3', 'cdna4', 'cdna5')
        or mnemonic not in CLASS_COMPARISONS
    ):
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
    api = 'rj_arith_' + mnemonic.lower() + ('_gfx12' if arch == 'rdna4' else '')
    result_writer = mask_result_writer(mnemonic, dst, result_writer, arch)
    lines += [
        f'  amdgpu::arith::execute_compare<{bits}, {min(bits, 32)}>(wf, {wave_api(api, 32)}, {wave_api(api, 64)},',
        f'      wf.cu().arithmetic_flags(), {modifiers},',
        f'      {src[0]}, {src[1]}, false, {result_writer});',
    ]
    return '\n'.join(lines)


COMPARE_NAMES = (
    set(INTEGER_COMPARISONS) | set(FLOAT_COMPARISONS) | set(CLASS_COMPARISONS)
)
