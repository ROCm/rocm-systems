# Copyright (c) 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""rj-arith calls for decoded RDNA dense matrix instructions."""

from __future__ import annotations

from .rj_arith_names import wave_api

import re

from . import ExecuteContext


def gen_dense_wmma(ctx: ExecuteContext) -> str | None:
    """Return a whole-wave call for a supported RDNA3/RDNA4 dense WMMA."""
    if ctx.arch_name not in ('rdna3', 'rdna4'):
        return None
    match = re.fullmatch(
        r'V_WMMA_(F32|F16|BF16|I32)_16X16X(16|32)_(F16|BF16|IU8|IU4|FP8_FP8|FP8_BF8|BF8_FP8|BF8_BF8)',
        ctx.inst.name,
    )
    if match is None:
        return None
    result, depth, factors = match.groups()
    depth = int(depth)
    old_layout = ctx.arch_name == 'rdna3'
    common_formats = {
        ('F32', 'F16'),
        ('F32', 'BF16'),
        ('F16', 'F16'),
        ('BF16', 'BF16'),
        ('I32', 'IU8'),
        ('I32', 'IU4'),
    }
    native_formats = {
        ('F32', 'FP8_FP8'),
        ('F32', 'FP8_BF8'),
        ('F32', 'BF8_FP8'),
        ('F32', 'BF8_BF8'),
    }
    if (result, factors) not in common_formats and (
        old_layout or (result, factors) not in native_formats
    ):
        return None
    if depth != 16 and (old_layout or result != 'I32' or factors != 'IU4'):
        return None
    bits = 16 if factors in ('F16', 'BF16') else 4 if factors == 'IU4' else 8
    packed = result in ('F16', 'BF16')
    api = 'rj_arith_' + ctx.inst.name.lower() + ('_gfx12' if not old_layout else '')
    dst = ctx.dst_ops[0]
    src0_base, src1_base, s2 = ctx.src_ops[:3]
    factor_count = (
        str(bits // 2)
        if old_layout
        else f'({16 * depth * bits // 32} + lanes - 1) / lanes'
    )
    result_count = f'{128 if packed and not old_layout else 256} / lanes'
    lines = [
        '  const uint32_t lanes = wf.wf_size();',
        '  const uint32_t vb = wf.vgpr_alloc().base;',
    ]
    if ctx.profile.uses_vgpr_msb_indexing:
        for variable, operand in [
            ('dst', dst),
            ('src0_base', src0_base),
            ('src1_base', src1_base),
        ]:
            lines.append(
                f'  uint32_t {variable} = vb + *Isa::resolved_vgpr_offset(wf, '
                f'{operand}.opr_type_, {operand}.encoding_value_, {operand}.vgpr_msb_role());'
            )
        lines += [
            '  uint32_t const_acc = 0;',
            f'  auto src2_off = Isa::resolved_vgpr_offset(wf, {s2}.opr_type_, '
            f'{s2}.encoding_value_, {s2}.vgpr_msb_role());',
            '  uint32_t s2 = src2_off ? vb + *src2_off : dst;',
            f'  if (!src2_off) const_acc = amdgpu::RegisterAccess(wf).read_scalar({s2});',
        ]
    else:
        lines += [
            f'  const uint32_t dst = vb + {dst}.encoding_value_;',
            f'  const uint32_t src0_base = amdgpu::src_base(vb, {src0_base}.encoding_value_);',
            f'  const uint32_t src1_base = amdgpu::src_base(vb, {src1_base}.encoding_value_);',
            '  uint32_t const_acc = 0;',
            f'  const uint32_t s2 = {ctx.arch_name}::resolve_acc(vb, dst, {s2}.encoding_value_,',
            f'      const_acc, [&] {{ return amdgpu::RegisterAccess(wf).read_scalar({s2}); }});',
        ]
    lines += [
        '  uint64_t flags = wf.cu().arithmetic_flags();',
        '  if (wf.fp16_ovfl()) flags |= RJ_ARITH_FP16_OVFL;',
    ]
    if packed:
        bf16 = str(factors == 'BF16').lower()
        lines += [
            f'  if ({s2}.encoding_value_ < 256)',
            f'    const_acc = amdgpu::dot_packed16::inline_word<{bf16}>(const_acc, {s2}.encoding_value_);',
        ]
    if result == 'I32':
        lines.append(
            '  uint64_t modifiers = (inst_.neg & 3u) | (inst_.clamp ? RJ_ARITH_WMMA_CLAMP : 0u);'
        )
    elif bits == 8:
        # Raw C sign aliases are ignored by the native matrix instruction.
        lines.append('  uint64_t modifiers = 0;')
    else:
        lines.append(
            '  uint64_t modifiers = inst_.neg | (uint64_t(inst_.neg_hi) << 3);'
        )
    if packed and old_layout:
        lines += [
            '  const bool high = (inst_.op_sel & 4u) != 0;',
            '  if (high) modifiers |= RJ_ARITH_WMMA_HIGH_C_D;',
        ]
    write_bytes = ', high ? 0xc : 0x3' if packed and old_layout else ''
    if not old_layout and factors == 'IU4' and depth == 16:
        write_bytes = ', 0xf, 32'
    lines += [
        f'  amdgpu::arith::execute_wmma(wf, lanes == 32 ? {wave_api(api, 32)} : {wave_api(api, 64)},',
        '      flags, modifiers, dst, src0_base, src1_base, s2,',
        f'      {factor_count}, {result_count},',
        f'      {s2}.encoding_value_ < 256 ? std::optional<uint32_t>(const_acc) : std::nullopt{write_bytes});',
    ]
    return '\n'.join(lines)
