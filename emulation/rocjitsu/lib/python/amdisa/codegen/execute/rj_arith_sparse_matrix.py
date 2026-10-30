# Copyright (c) 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""rj-arith calls for RDNA4 structured-sparse matrix instructions."""

from __future__ import annotations


from .rj_arith_names import wave_api


import re


from . import ExecuteContext


def gen_sparse_wmma(ctx: ExecuteContext) -> str | None:
    """Return a whole-wave SWMMAC call, including its tied accumulator read."""
    if not ctx.profile.uses_rj_arith_matrix or ctx.arch_name != 'rdna4':
        return None
    match = re.fullmatch(
        r'V_SWMMAC_(F32|F16|BF16|I32)_16X16X(32|64)_(F16|BF16|IU8|IU4|FP8_FP8|FP8_BF8|BF8_FP8|BF8_BF8)',
        ctx.inst.name,
    )
    if match is None:
        return None
    result, depth, factors = match.groups()
    depth = int(depth)
    valid = {
        ('F32', 32, 'F16'),
        ('F32', 32, 'BF16'),
        ('F16', 32, 'F16'),
        ('BF16', 32, 'BF16'),
        ('F32', 32, 'FP8_FP8'),
        ('F32', 32, 'FP8_BF8'),
        ('F32', 32, 'BF8_FP8'),
        ('F32', 32, 'BF8_BF8'),
        ('I32', 32, 'IU8'),
        ('I32', 32, 'IU4'),
        ('I32', 64, 'IU4'),
    }
    if (result, depth, factors) not in valid:
        return None
    bits = 16 if factors in ('F16', 'BF16') else 4 if factors == 'IU4' else 8
    packed = result in ('F16', 'BF16')
    api = 'rj_arith_' + ctx.inst.name.lower()
    lines = [
        '  const uint32_t lanes = wf.wf_size();',
        '  const uint32_t vb = wf.vgpr_alloc().base;',
    ]
    operands = [
        ('dst', ctx.dst_ops[0]),
        ('src0_base', ctx.src_ops[0]),
        ('src1_base', ctx.src_ops[1]),
        ('index_base', ctx.src_ops[2]),
    ]
    for variable, operand in operands:
        if variable == 'dst':
            expression = f'vb + {operand}.encoding_value_'
        else:
            expression = f'amdgpu::src_base(vb, {operand}.encoding_value_)'
        lines.append(f'  uint32_t {variable} = {expression};')
    # The existing gfx1201 model ignores selector bits above the metadata set
    # count. rj-arith validates those bits, so normalize the encoded aliases here.
    key_mask = (
        'lanes == 32 ? 0u : 1u'
        if depth == 64
        else '1u' if bits == 4 else 'lanes == 32 ? 1u : 3u'
    )
    lines += [
        f'  const uint32_t index_key = inst_.opsel & ({key_mask});',
        '  uint64_t modifiers = uint64_t(index_key) << 7;',
        '  uint64_t flags = wf.cu().arithmetic_flags();',
    ]
    if result == 'I32':
        lines.append(
            '  modifiers |= (inst_.neg & 3u) | (inst_.clamp ? RJ_ARITH_WMMA_CLAMP : 0u);'
        )
    elif bits == 16:
        lines.append(
            '  modifiers |= (inst_.neg & 7u) | (uint64_t(inst_.neg_hi & 7u) << 3);'
        )
        if packed:
            lines.append('  if (wf.fp16_ovfl()) flags |= RJ_ARITH_FP16_OVFL;')
    # A contains half as many elements as dense B. Round up for Wave64 IU4,
    # whose one physical A register uses only lanes 0..31 for K=32.
    a_words = 16 * (depth // 2) * bits // 32
    b_words = 16 * depth * bits // 32
    d_words = 128 if packed else 256
    a_lanes = '32' if bits == 4 and depth == 32 else 'lanes'
    lines += [
        f'  amdgpu::arith::exec_swmmac(wf, lanes == 32 ? {wave_api(api, 32)} : {wave_api(api, 64)},',
        '      flags, modifiers, dst, src0_base, src1_base, index_base,',
        f'      ({a_words} + lanes - 1) / lanes, {b_words} / lanes, {d_words} / lanes, {a_lanes});',
    ]
    return '\n'.join(lines)
