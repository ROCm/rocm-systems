# Copyright (c) 2025-2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Packed math execute body generators.

Free functions that emit C++ execute_impl bodies for packed 16-bit
instructions (V_PK_ADD_F16, etc.), packed F32 instructions,
and V_PK_MOV_B32.

The opsel_exprs parameter carries the C++ expressions for op_sel and
op_sel_hi fields, derived from the ISA profile at call time.
"""

from __future__ import annotations


from .vop3_modifiers import vop3_dst_mod, vop3_src_mod


def _append_pk_f32_pair_read(
    L: list[str], var: str, src: str, use_cdna5_helpers: bool = False
) -> None:
    """Read a packed F32 operand as a pair of 32-bit words."""
    if use_cdna5_helpers:
        L.append(
            f'    const auto {var} = amdgpu::RegisterAccess(wf).read_lane_pair32('
            f'{src}, lane, amdgpu::ScalarPairMode::Replicate32);'
        )
        return
    L.append(
        f'    const auto {var}_pair_w = amdgpu::RegisterAccess(wf).read_lane_pair32({src}, lane);'
    )
    L.append(f'    const uint32_t {var}_lo_w = {var}_pair_w.lo;')
    L.append(f'    const uint32_t {var}_hi_w = {var}_pair_w.hi;')


def _pk_f32_word_expr(var: str, half: str, use_cdna5_helpers: bool = False) -> str:
    if use_cdna5_helpers:
        return f'{var}.{half}'
    return f'{var}_{half}_w'


def _append_pk16_src_reads(
    L: list[str], srcs: list[str], narrow_inline_to: str | None
) -> None:
    """Read the packed 16-bit sources of one lane.

    read_lane() returns the FP32 pattern of a 32-bit inline float constant.
    When narrow_inline_to is set, convert that pattern to the requested
    16-bit type in the low half. Packed BF16 leaves it unset: CDNA5 ISA
    section 7.7.2 requires OPSEL to select the upper half of the FP32
    constant. F16 and DOT operands retain their separate narrowing rules.
    Narrowing is keyed on the source-selector field rather than the operand,
    so a 32-bit literal
    (selector 255) keeps its value even when that value lands in 240..248.
    The narrowed inline pattern occupies the low half of the packed pair.

    pk16_src_needs_narrowing also declines a source that is declared 16 bits
    wide, because Operand::read_lane already resolved that one through the
    half-precision inline table. CDNA2 builds every packed f16 source that
    way, CDNA3 does for v_pk_min_f16 / v_pk_max_f16.
    """
    narrow = {'f16': 'util::f32_to_f16', 'bf16': 'util::f32_to_bf16'}.get(
        narrow_inline_to or ''
    )
    for i, src in enumerate(srcs):
        L.append(
            f'    uint32_t raw{i} = amdgpu::RegisterAccess(wf).read_lane({src}, lane);'
        )
    if narrow is None:
        return
    for i, src in enumerate(srcs):
        L.append(
            f'    if (amdgpu::pk16_src_needs_narrowing(inst_.src{i}, '
            f'{src}.size_bits()))'
        )
        L.append(f'      raw{i} = {narrow}(std::bit_cast<float>(raw{i}));')


def _packed_inline_narrow_to(dtype: str | None) -> str | None:
    """Packed BF16 keeps the FP32 inline pattern for OPSEL to select its upper half."""
    return 'f16' if dtype == 'f16' else None


def gen_pk_binop(
    dst: list[str],
    src: list[str],
    op: str | None,
    dtype: str | None,
    opsel_exprs: tuple[str, str] = ('', ''),
) -> str:
    """Generate the remaining BF16 packed binary fallback bodies."""
    if dtype != 'bf16':
        raise ValueError(f'packed binary fallback does not support {dtype}')
    d, s0, s1 = dst[0], src[0], src[1]
    L = []
    L.append('  uint64_t exec = wf.exec();')
    L.append('  for (uint32_t lane = 0; lane < wf.wf_size(); ++lane) {')
    L.append('    if (!(exec & (1ULL << lane))) continue;')
    _append_pk16_src_reads(L, [s0, s1], _packed_inline_narrow_to(dtype))

    # op_sel: which half of each src for LO result
    # op_sel_hi: which half for HI result (default = hi)
    opsel, opsel_hi = opsel_exprs
    L.append(f'    bool sel0_lo = ({opsel} >> 0) & 1;')
    L.append(f'    bool sel1_lo = ({opsel} >> 1) & 1;')
    L.append(f'    bool sel0_hi = ({opsel_hi} >> 0) & 1;')
    L.append(f'    bool sel1_hi = ({opsel_hi} >> 1) & 1;')

    if dtype == 'bf16':
        L.append(
            '    float a_lo = util::bf16_to_f32(static_cast<uint16_t>(sel0_lo ? (raw0 >> 16) : raw0));'
        )
        L.append(
            '    float b_lo = util::bf16_to_f32(static_cast<uint16_t>(sel1_lo ? (raw1 >> 16) : raw1));'
        )
        L.append(
            '    float a_hi = util::bf16_to_f32(static_cast<uint16_t>(sel0_hi ? (raw0 >> 16) : raw0));'
        )
        L.append(
            '    float b_hi = util::bf16_to_f32(static_cast<uint16_t>(sel1_hi ? (raw1 >> 16) : raw1));'
        )
        L.append('    if (inst_.neg & 1) { a_lo = -a_lo; }')
        L.append('    if (inst_.neg & 2) { b_lo = -b_lo; }')
        L.append('    if (inst_.neg_hi & 1) { a_hi = -a_hi; }')
        L.append('    if (inst_.neg_hi & 2) { b_hi = -b_hi; }')
        if op in ('add', 'mul'):
            for half, result in (('lo', 'rlo'), ('hi', 'rhi')):
                if op == 'add':
                    expr = f'amdgpu::fp_mode::packed_add_bf16(a_{half}, b_{half}, wf.fp16_ovfl())'
                else:
                    expr = f'amdgpu::fp_mode::packed_mul_bf16(a_{half}, b_{half}, wf.fp16_ovfl())'
                L.append(f'    uint16_t {result} = {expr};')
        else:
            minimum = 'true' if op == 'min' else 'false'
            L.append(
                f'    uint16_t rlo = amdgpu::fp_mode::packed_select_bf16(a_lo, b_lo, {minimum});'
            )
            L.append(
                f'    uint16_t rhi = amdgpu::fp_mode::packed_select_bf16(a_hi, b_hi, {minimum});'
            )
        for result in ('rlo', 'rhi'):
            L.append(
                f'    {result} = amdgpu::fp_mode::clamp_bf16({result}, inst_.clamp, amdgpu::floating_clamp_nan_to_zero(wf));'
            )
        L.append(
            f'    amdgpu::RegisterAccess(wf).write_lane({d}, lane, rlo | (static_cast<uint32_t>(rhi) << 16));'
        )
    L.append('  }')
    return '\n'.join(L)


def gen_pk_ternary(
    dst: list[str],
    src: list[str],
    op: str | None,
    dtype: str | None,
    op_sel_hi_2_expr: str = '',
    opsel_exprs: tuple[str, str] = ('', ''),
) -> str:
    """Generate packed FP16/BF16 ternary arithmetic."""
    d, s0, s1, s2 = dst[0], src[0], src[1], src[2]
    L = []
    L.append('  uint64_t exec = wf.exec();')
    L.append('  for (uint32_t lane = 0; lane < wf.wf_size(); ++lane) {')
    L.append('    if (!(exec & (1ULL << lane))) continue;')
    _append_pk16_src_reads(L, [s0, s1, s2], _packed_inline_narrow_to(dtype))
    opsel, opsel_hi = opsel_exprs
    L.append(f'    bool sel0_lo = ({opsel} >> 0) & 1;')
    L.append(f'    bool sel1_lo = ({opsel} >> 1) & 1;')
    L.append(f'    bool sel2_lo = ({opsel} >> 2) & 1;')
    L.append(f'    bool sel0_hi = ({opsel_hi} >> 0) & 1;')
    L.append(f'    bool sel1_hi = ({opsel_hi} >> 1) & 1;')
    L.append(f'    bool sel2_hi = {op_sel_hi_2_expr};')

    if dtype == 'f16':
        if op in ('min3', 'max3', 'minimum3', 'maximum3'):
            selection = op.removesuffix('3').upper()
            for half, neg in (('lo', 'neg'), ('hi', 'neg_hi')):
                for index, name in enumerate(('first', 'second', 'third')):
                    L.append(
                        f'    uint16_t {name}_{half} = static_cast<uint16_t>(sel{index}_{half} ? (raw{index} >> 16) : raw{index});'
                    )
                    L.append(
                        f'    if (inst_.{neg} & {1 << index}u) {name}_{half} ^= 0x8000u;'
                    )
                L.append(
                    f'    const uint16_t r{half} = amdgpu::fp_mode::packed_select3_f16(amdgpu::fp_mode::PackedBinaryOp::{selection}, first_{half}, second_{half}, third_{half}, wf.fp_denorm_mode_f16_f64(), inst_.clamp, amdgpu::floating_clamp_nan_to_zero(wf), wf.cu().arch(), wf.ieee_mode());'
                )
            L.append(
                f'    amdgpu::RegisterAccess(wf).write_lane({d}, lane, rlo | (static_cast<uint32_t>(rhi) << 16));'
            )
            L.append('  }')
            return '\n'.join(L)
        raise ValueError(f'packed FP16 ternary fallback does not support {op}')
    elif dtype == 'bf16':
        if op != 'fma':
            raise ValueError(f'Unsupported packed BF16 ternary operation: {op}')
        L.append(
            '    float a_lo = util::bf16_to_f32(static_cast<uint16_t>(sel0_lo ? (raw0 >> 16) : raw0));'
        )
        L.append(
            '    float b_lo = util::bf16_to_f32(static_cast<uint16_t>(sel1_lo ? (raw1 >> 16) : raw1));'
        )
        L.append(
            '    float c_lo = util::bf16_to_f32(static_cast<uint16_t>(sel2_lo ? (raw2 >> 16) : raw2));'
        )
        L.append(
            '    float a_hi = util::bf16_to_f32(static_cast<uint16_t>(sel0_hi ? (raw0 >> 16) : raw0));'
        )
        L.append(
            '    float b_hi = util::bf16_to_f32(static_cast<uint16_t>(sel1_hi ? (raw1 >> 16) : raw1));'
        )
        L.append(
            '    float c_hi = util::bf16_to_f32(static_cast<uint16_t>(sel2_hi ? (raw2 >> 16) : raw2));'
        )
        L.append('    if (inst_.neg & 1) { a_lo = -a_lo; }')
        L.append('    if (inst_.neg & 2) { b_lo = -b_lo; }')
        L.append('    if (inst_.neg & 4) { c_lo = -c_lo; }')
        L.append('    if (inst_.neg_hi & 1) { a_hi = -a_hi; }')
        L.append('    if (inst_.neg_hi & 2) { b_hi = -b_hi; }')
        L.append('    if (inst_.neg_hi & 4) { c_hi = -c_hi; }')
        L.append(
            '    uint16_t rlo = amdgpu::fp_mode::packed_fma_bf16(a_lo, b_lo, c_lo, wf.fp16_ovfl());'
        )
        L.append(
            '    uint16_t rhi = amdgpu::fp_mode::packed_fma_bf16(a_hi, b_hi, c_hi, wf.fp16_ovfl());'
        )
        for result in ('rlo', 'rhi'):
            L.append(
                f'    {result} = amdgpu::fp_mode::clamp_bf16({result}, inst_.clamp, amdgpu::floating_clamp_nan_to_zero(wf));'
            )
        L.append(
            f'    amdgpu::RegisterAccess(wf).write_lane({d}, lane, rlo | (static_cast<uint32_t>(rhi) << 16));'
        )
    L.append('  }')
    return '\n'.join(L)


def gen_pk_binop_u64(dst: list[str], src: list[str], op: str) -> str:
    """Forward packed U64 source negation and saturation to rj-arith."""
    if op not in ('add', 'sub'):
        raise ValueError(f'unsupported packed U64 binary operation: {op}')
    return '\n'.join(
        [
            '  const uint64_t modifiers = inst_.neg | (uint64_t(inst_.neg_hi) << 3) |',
            '      (inst_.clamp ? RJ_ARITH_PK_CLAMP : 0);',
            '  amdgpu::arith::execute_packed_u64<false>(',
            f'      wf, rj_arith_v_pk_{op}_nc_u64_w32, modifiers, {dst[0]}, {src[0]}, {src[1]});',
        ]
    )


def gen_pk_binop_f64(dst: list[str], src: list[str], op: str) -> str:
    """Forward both FP64 elements and their independent signs to rj-arith."""
    if op not in ('add', 'mul', 'min_num', 'max_num'):
        raise ValueError(f'unsupported packed F64 binary operation: {op}')
    return '\n'.join(
        [
            '  const uint64_t modifiers = inst_.neg | (uint64_t(inst_.neg_hi) << 3) |',
            '      (inst_.clamp ? RJ_ARITH_PK_CLAMP : 0);',
            '  amdgpu::arith::execute_packed_f64<false>(',
            f'      wf, rj_arith_v_pk_{op}_f64_w32, modifiers, {dst[0]}, {src[0]}, {src[1]});',
        ]
    )


def gen_pk_ternary_f64(dst: list[str], src: list[str], op: str) -> str:
    """Forward packed FP64 FMA through the carrier-aware guest-FP adapter."""
    if op != 'fma':
        raise ValueError(f'unsupported packed F64 ternary operation: {op}')
    return '\n'.join(
        [
            '  const uint64_t modifiers = inst_.neg | (uint64_t(inst_.neg_hi) << 3) |',
            '      (inst_.clamp ? RJ_ARITH_PK_CLAMP : 0);',
            '  amdgpu::arith::execute_packed_f64<true>(',
            f'      wf, rj_arith_v_pk_fma_f64_w32, modifiers, {dst[0]}, {src[0]}, {src[1]}, &{src[2]});',
        ]
    )


def gen_pk_lshl_add_u64(dst: list[str], src: list[str]) -> str:
    """Forward shift-add through the adapter that validates active counts before stores."""
    return '\n'.join(
        [
            '  amdgpu::arith::execute_packed_u64<true>(',
            f'      wf, rj_arith_v_pk_lshl_add_u64_w32, 0, {dst[0]}, {src[0]}, {src[1]}, &{src[2]});',
        ]
    )


# FP8 DOT4 semantic operation -> (src0 is FP8, src1 is FP8), as for FP8 WMMA.
_FP8_DOT4_FP8_SOURCES = {
    'fp8_fp8': ('true', 'true'),
    'fp8_bf8': ('true', 'false'),
    'bf8_fp8': ('false', 'true'),
    'bf8_bf8': ('false', 'false'),
}
