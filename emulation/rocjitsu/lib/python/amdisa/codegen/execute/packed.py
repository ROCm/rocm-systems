# Copyright (c) 2025-2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Packed math execute body generators.

Free functions that emit C++ execute_impl bodies for packed 16-bit
instructions (V_PK_ADD_F16, etc.), paired 64-bit instructions,
dot product instructions, and V_PK_MOV_B32.

The opsel_exprs parameter carries the C++ expressions for op_sel and
op_sel_hi fields, derived from the ISA profile at call time.
"""

from __future__ import annotations

from amdisa.isa_profile import FloatDotAccumulation

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


def gen_pk_binop(
    dst: list[str],
    src: list[str],
    op: str | None,
    dtype: str | None,
    opsel_exprs: tuple[str, str] = ('', ''),
    multiply_clamp: bool = False,
) -> str:
    """Generate the remaining integer packed binary fallback bodies."""
    if dtype not in ('i16', 'u16'):
        raise ValueError(f'packed binary fallback does not support {dtype}')
    d, s0, s1 = dst[0], src[0], src[1]
    L = []
    L.append(
        '  const uint32_t negations = amdgpu::packed_integer_negations(wf.cu().arch(), inst_);'
    )
    L.append('  uint64_t exec = wf.exec();')
    L.append('  for (uint32_t lane = 0; lane < wf.wf_size(); ++lane) {')
    L.append('    if (!(exec & (1ULL << lane))) continue;')
    _append_pk16_src_reads(L, [s0, s1], None)

    # op_sel: which half of each src for LO result
    # op_sel_hi: which half for HI result (default = hi)
    opsel, opsel_hi = opsel_exprs
    _append_pk16_integer_operands(L, dtype, 2, opsel, opsel_hi, 'negations')

    if dtype == 'i16':
        i_op_map = {
            'add': ('a_lo + b_lo', 'a_hi + b_hi'),
            'sub': ('a_lo - b_lo', 'a_hi - b_hi'),
            'max': ('a_lo > b_lo ? a_lo : b_lo', 'a_hi > b_hi ? a_hi : b_hi'),
            'min': ('a_lo < b_lo ? a_lo : b_lo', 'a_hi < b_hi ? a_hi : b_hi'),
            'ashr': (
                'static_cast<int16_t>(b_lo >> (a_lo & 15))',
                'static_cast<int16_t>(b_hi >> (a_hi & 15))',
            ),
        }
        lo_expr, hi_expr = i_op_map[op]
        if op in ('add', 'sub'):
            L.append(
                f'    uint16_t rlo = static_cast<uint16_t>(amdgpu::vop3_integer_{op}<int16_t>(a_lo, b_lo, inst_.clamp));'
            )
            L.append(
                f'    uint16_t rhi = static_cast<uint16_t>(amdgpu::vop3_integer_{op}<int16_t>(a_hi, b_hi, inst_.clamp));'
            )
        else:
            L.append(f'    uint16_t rlo = static_cast<uint16_t>({lo_expr});')
            L.append(f'    uint16_t rhi = static_cast<uint16_t>({hi_expr});')
        L.append(
            f'    amdgpu::RegisterAccess(wf).write_lane({d}, lane, static_cast<uint32_t>(rlo) | (static_cast<uint32_t>(rhi) << 16));'
        )
    else:  # u16
        clamp = 'inst_.clamp' if multiply_clamp else 'false'
        u_op_map = {
            'add': ('a_lo + b_lo', 'a_hi + b_hi'),
            'sub': ('a_lo - b_lo', 'a_hi - b_hi'),
            'mul': (
                f'amdgpu::vop3_integer_mul<uint16_t, 16>(a_lo, b_lo, {clamp})',
                f'amdgpu::vop3_integer_mul<uint16_t, 16>(a_hi, b_hi, {clamp})',
            ),
            'max': ('a_lo > b_lo ? a_lo : b_lo', 'a_hi > b_hi ? a_hi : b_hi'),
            'min': ('a_lo < b_lo ? a_lo : b_lo', 'a_hi < b_hi ? a_hi : b_hi'),
            'shl': (
                'static_cast<uint16_t>(b_lo << (a_lo & 15u))',
                'static_cast<uint16_t>(b_hi << (a_hi & 15u))',
            ),
            'shr': (
                'static_cast<uint16_t>(b_lo >> (a_lo & 15u))',
                'static_cast<uint16_t>(b_hi >> (a_hi & 15u))',
            ),
        }
        lo_expr, hi_expr = u_op_map[op]
        if op in ('add', 'sub'):
            L.append(
                f'    uint16_t rlo = static_cast<uint16_t>(amdgpu::vop3_integer_{op}<uint16_t>(a_lo, b_lo, inst_.clamp));'
            )
            L.append(
                f'    uint16_t rhi = static_cast<uint16_t>(amdgpu::vop3_integer_{op}<uint16_t>(a_hi, b_hi, inst_.clamp));'
            )
        else:
            L.append(f'    uint16_t rlo = static_cast<uint16_t>({lo_expr});')
            L.append(f'    uint16_t rhi = static_cast<uint16_t>({hi_expr});')
        L.append(
            f'    amdgpu::RegisterAccess(wf).write_lane({d}, lane, static_cast<uint32_t>(rlo) | (static_cast<uint32_t>(rhi) << 16));'
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
    integer_clamp: bool = False,
) -> str:
    """Generate the remaining packed integer ternary fallback bodies."""
    if dtype not in ('i16', 'u16'):
        raise ValueError(f'packed ternary fallback does not support {dtype}')
    d, s0, s1, s2 = dst[0], src[0], src[1], src[2]
    L = []
    if op == 'mad':
        L.append(
            '  const uint32_t negations = amdgpu::packed_integer_negations(wf.cu().arch(), inst_);'
        )
    L.append('  uint64_t exec = wf.exec();')
    L.append('  for (uint32_t lane = 0; lane < wf.wf_size(); ++lane) {')
    L.append('    if (!(exec & (1ULL << lane))) continue;')
    _append_pk16_src_reads(L, [s0, s1, s2], None)
    opsel, opsel_hi = opsel_exprs
    high = f'({opsel_hi} | ({op_sel_hi_2_expr} << 2))'
    _append_pk16_integer_operands(
        L, dtype, 3, opsel, high, ('negations' if op == 'mad' else '0')
    )

    if dtype == 'i16':
        if op == 'add_max_sat':
            L.append(
                '    int16_t sum_lo = static_cast<int16_t>(std::clamp(static_cast<int32_t>(a_lo) + b_lo, -32768, 32767));'
            )
            L.append(
                '    int16_t sum_hi = static_cast<int16_t>(std::clamp(static_cast<int32_t>(a_hi) + b_hi, -32768, 32767));'
            )
            L.append(
                '    uint16_t rlo = static_cast<uint16_t>(std::max(sum_lo, c_lo));'
            )
            L.append(
                '    uint16_t rhi = static_cast<uint16_t>(std::max(sum_hi, c_hi));'
            )
        elif op == 'add_min_sat':
            L.append(
                '    int16_t sum_lo = static_cast<int16_t>(std::clamp(static_cast<int32_t>(a_lo) + b_lo, -32768, 32767));'
            )
            L.append(
                '    int16_t sum_hi = static_cast<int16_t>(std::clamp(static_cast<int32_t>(a_hi) + b_hi, -32768, 32767));'
            )
            L.append(
                '    uint16_t rlo = static_cast<uint16_t>(std::min(sum_lo, c_lo));'
            )
            L.append(
                '    uint16_t rhi = static_cast<uint16_t>(std::min(sum_hi, c_hi));'
            )
        elif op in ('min3', 'minimum3'):
            L.append(
                '    uint16_t rlo = static_cast<uint16_t>(std::min(std::min(a_lo, b_lo), c_lo));'
            )
            L.append(
                '    uint16_t rhi = static_cast<uint16_t>(std::min(std::min(a_hi, b_hi), c_hi));'
            )
        elif op in ('max3', 'maximum3'):
            L.append(
                '    uint16_t rlo = static_cast<uint16_t>(std::max(std::max(a_lo, b_lo), c_lo));'
            )
            L.append(
                '    uint16_t rhi = static_cast<uint16_t>(std::max(std::max(a_hi, b_hi), c_hi));'
            )
        elif op == 'mad' and integer_clamp:
            L.append(
                '    uint16_t rlo = amdgpu::vop3_integer_mad<int16_t, 16>('
                'static_cast<uint16_t>(a_lo), static_cast<uint16_t>(b_lo), '
                'static_cast<uint16_t>(c_lo), inst_.clamp);'
            )
            L.append(
                '    uint16_t rhi = amdgpu::vop3_integer_mad<int16_t, 16>('
                'static_cast<uint16_t>(a_hi), static_cast<uint16_t>(b_hi), '
                'static_cast<uint16_t>(c_hi), inst_.clamp);'
            )
        else:
            L.append(
                '    uint16_t rlo = static_cast<uint16_t>(static_cast<uint32_t>(a_lo) * b_lo + c_lo);'
            )
            L.append(
                '    uint16_t rhi = static_cast<uint16_t>(static_cast<uint32_t>(a_hi) * b_hi + c_hi);'
            )
        if op in ('add_max_sat', 'add_min_sat', 'min3', 'max3'):
            L.extend(
                [
                    '    if (inst_.clamp) {',
                    '      rlo = static_cast<uint16_t>(std::max<int16_t>(static_cast<int16_t>(rlo), 0));',
                    '      rhi = static_cast<uint16_t>(std::max<int16_t>(static_cast<int16_t>(rhi), 0));',
                    '    }',
                ]
            )
        L.append(
            f'    amdgpu::RegisterAccess(wf).write_lane({d}, lane, static_cast<uint32_t>(rlo) | (static_cast<uint32_t>(rhi) << 16));'
        )
    else:  # u16
        if op == 'add_max_sat':
            L.append(
                '    uint16_t sum_lo = static_cast<uint16_t>(std::min(static_cast<uint32_t>(a_lo) + b_lo, 65535u));'
            )
            L.append(
                '    uint16_t sum_hi = static_cast<uint16_t>(std::min(static_cast<uint32_t>(a_hi) + b_hi, 65535u));'
            )
            L.append('    uint16_t rlo = std::max(sum_lo, c_lo);')
            L.append('    uint16_t rhi = std::max(sum_hi, c_hi);')
        elif op == 'add_min_sat':
            L.append(
                '    uint16_t sum_lo = static_cast<uint16_t>(std::min(static_cast<uint32_t>(a_lo) + b_lo, 65535u));'
            )
            L.append(
                '    uint16_t sum_hi = static_cast<uint16_t>(std::min(static_cast<uint32_t>(a_hi) + b_hi, 65535u));'
            )
            L.append('    uint16_t rlo = std::min(sum_lo, c_lo);')
            L.append('    uint16_t rhi = std::min(sum_hi, c_hi);')
        elif op in ('min3', 'minimum3'):
            L.append('    uint16_t rlo = std::min(std::min(a_lo, b_lo), c_lo);')
            L.append('    uint16_t rhi = std::min(std::min(a_hi, b_hi), c_hi);')
        elif op in ('max3', 'maximum3'):
            L.append('    uint16_t rlo = std::max(std::max(a_lo, b_lo), c_lo);')
            L.append('    uint16_t rhi = std::max(std::max(a_hi, b_hi), c_hi);')
        elif op == 'mad' and integer_clamp:
            L.append(
                '    uint16_t rlo = amdgpu::vop3_integer_mad<uint16_t, 16>('
                'a_lo, b_lo, c_lo, inst_.clamp);'
            )
            L.append(
                '    uint16_t rhi = amdgpu::vop3_integer_mad<uint16_t, 16>('
                'a_hi, b_hi, c_hi, inst_.clamp);'
            )
        else:
            L.append(
                '    uint16_t rlo = static_cast<uint16_t>(static_cast<uint32_t>(a_lo) * b_lo + c_lo);'
            )
            L.append(
                '    uint16_t rhi = static_cast<uint16_t>(static_cast<uint32_t>(a_hi) * b_hi + c_hi);'
            )
        L.append(
            f'    amdgpu::RegisterAccess(wf).write_lane({d}, lane, static_cast<uint32_t>(rlo) | (static_cast<uint32_t>(rhi) << 16));'
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
