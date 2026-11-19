# Copyright (c) 2025-2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Packed math execute body generators.

Free functions that emit C++ execute_impl bodies for packed 16-bit
instructions (V_PK_ADD_F16, etc.), paired 64-bit instructions,
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


def _append_pk16_integer_operands(
    lines: list[str], dtype: str, count: int, opsel: str, opsel_hi: str, negations: str
) -> None:
    ctype = 'int16_t' if dtype == 'i16' else 'uint16_t'
    for index, name in enumerate(('a', 'b', 'c')[:count]):
        lines.append(
            f'    const uint32_t {name} = amdgpu::select_packed_halves('
            f'raw{index}, {opsel}, {opsel_hi}, {index}, {negations});'
        )
        lines.append(f'    const auto {name}_lo = static_cast<{ctype}>({name});')
        lines.append(f'    const auto {name}_hi = static_cast<{ctype}>({name} >> 16);')


def gen_pk_binop(
    dst: list[str],
    src: list[str],
    op: str | None,
    dtype: str | None,
    opsel_exprs: tuple[str, str] = ('', ''),
    multiply_clamp: bool = False,
) -> str:
    """Generate the remaining integer/BF16 packed binary fallback bodies."""
    if dtype not in ('i16', 'u16', 'bf16'):
        raise ValueError(f'packed binary fallback does not support {dtype}')
    d, s0, s1 = dst[0], src[0], src[1]
    L = []
    integer = dtype in ('i16', 'u16')
    if integer:
        L.append(
            '  const uint32_t negations = amdgpu::packed_integer_negations(wf.cu().arch(), inst_);'
        )
    L.append('  uint64_t exec = wf.exec();')
    L.append('  for (uint32_t lane = 0; lane < wf.wf_size(); ++lane) {')
    L.append('    if (!(exec & (1ULL << lane))) continue;')
    _append_pk16_src_reads(L, [s0, s1], _packed_inline_narrow_to(dtype))

    # op_sel: which half of each src for LO result
    # op_sel_hi: which half for HI result (default = hi)
    opsel, opsel_hi = opsel_exprs
    if integer:
        _append_pk16_integer_operands(L, dtype, 2, opsel, opsel_hi, 'negations')
    else:
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
    elif dtype == 'i16':
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
    """Generate packed 16-bit ternary op (V_PK_FMA_F16, V_PK_MAD_I16, etc.)."""
    d, s0, s1, s2 = dst[0], src[0], src[1], src[2]
    L = []
    integer = dtype in ('i16', 'u16')
    if integer and op == 'mad':
        L.append(
            '  const uint32_t negations = amdgpu::packed_integer_negations(wf.cu().arch(), inst_);'
        )
    L.append('  uint64_t exec = wf.exec();')
    L.append('  for (uint32_t lane = 0; lane < wf.wf_size(); ++lane) {')
    L.append('    if (!(exec & (1ULL << lane))) continue;')
    _append_pk16_src_reads(L, [s0, s1, s2], _packed_inline_narrow_to(dtype))
    opsel, opsel_hi = opsel_exprs
    if integer:
        high = f'({opsel_hi} | ({op_sel_hi_2_expr} << 2))'
        _append_pk16_integer_operands(
            L, dtype, 3, opsel, high, ('negations' if op == 'mad' else '0')
        )
    else:
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
    elif dtype == 'i16':
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
    """Generate packed U64 add/sub with VOP3P source modifiers and clamp.

    The packed pair contains two independent U64 elements.  ``neg`` applies
    to the low element and ``neg_hi`` to the high element before the binary
    operation.  A signed 128-bit intermediate preserves the mathematical
    result until integer clamp has saturated it to the U64 range; without
    clamp, conversion back to U64 provides the instruction's modulo-2^64
    result.  Source negation is applied before the arithmetic, and clamp is
    applied to the resulting mathematical value.
    """
    if op not in ('add', 'sub'):
        raise ValueError(f'unsupported packed U64 binary operation: {op}')

    d, s0, s1 = dst[0], src[0], src[1]
    operator = '+' if op == 'add' else '-'
    L = [
        '  uint64_t exec = wf.exec();',
        '  for (uint32_t lane = 0; lane < wf.wf_size(); ++lane) {',
        '    if (!(exec & (1ULL << lane))) continue;',
        f'    const auto lhs = read_pk_u64_pair({s0}, wf, lane);',
        f'    const auto rhs = read_pk_u64_pair({s1}, wf, lane);',
        '    const auto apply = [&](uint64_t lhs_value, uint64_t rhs_value,',
        '                           bool negate_lhs, bool negate_rhs) -> uint64_t {',
        '      using Wide = util::int128_t;',
        '      Wide lhs_wide = static_cast<Wide>(lhs_value);',
        '      Wide rhs_wide = static_cast<Wide>(rhs_value);',
        '      if (negate_lhs) lhs_wide = -lhs_wide;',
        '      if (negate_rhs) rhs_wide = -rhs_wide;',
        f'      const Wide result = lhs_wide {operator} rhs_wide;',
        '      if (inst_.clamp) {',
        '        if (result < Wide{}) return 0;',
        '        constexpr Wide kMax =',
        '            static_cast<Wide>(std::numeric_limits<uint64_t>::max());',
        '        if (result > kMax) return std::numeric_limits<uint64_t>::max();',
        '      }',
        '      return static_cast<uint64_t>(result);',
        '    };',
        '    const uint64_t result_lo =',
        '        apply(lhs.lo, rhs.lo, (inst_.neg & 1u) != 0, (inst_.neg & 2u) != 0);',
        '    const uint64_t result_hi = apply(lhs.hi, rhs.hi,',
        '                                     (inst_.neg_hi & 1u) != 0,',
        '                                     (inst_.neg_hi & 2u) != 0);',
        f'    write_pk_u64_pair({d}, wf, lane, {{result_lo, result_hi}});',
        '  }',
    ]
    return '\n'.join(L)


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
    """Generate packed U64 shift-left-add over two independent elements.

    V_PK_LSHL_ADD_U64 has two 64-bit values in each 128-bit value/addend
    operand and two 32-bit shift counts in its 64-bit shift operand. VGPR U64
    sources supply both elements, while scalar-backed U64 sources broadcast the
    low 64 bits. Its public LLVM profile explicitly disables clamp and source
    modifiers, so the body intentionally consumes neither field. The currently
    proven shift-count range is 0..4; execution fails closed before any
    destination write if either count of any active lane is outside that range.
    """
    d, s0, s1, s2 = dst[0], src[0], src[1], src[2]
    return '\n'.join(
        [
            '  uint64_t exec = wf.exec();',
            '  std::array<PkU64Pair, 64> results{};',
            '  for (uint32_t lane = 0; lane < wf.wf_size(); ++lane) {',
            '    if (!(exec & (1ULL << lane))) continue;',
            f'    const auto values = read_pk_u64_pair({s0}, wf, lane);',
            f'    const auto shifts = read_pk_u32_pair({s1}, wf, lane);',
            f'    const auto addends = read_pk_u64_pair({s2}, wf, lane);',
            '    if (shifts.lo > 4u || shifts.hi > 4u) {',
            '      wf.report_instruction_execution_error(',
            '          amdgpu::InstructionExecutionError::UnsupportedOperandValue);',
            '      return;',
            '    }',
            '    const uint64_t result_lo = amdgpu::lshl_masked(values.lo, static_cast<uint64_t>(shifts.lo)) + addends.lo;',
            '    const uint64_t result_hi = amdgpu::lshl_masked(values.hi, static_cast<uint64_t>(shifts.hi)) + addends.hi;',
            '    results[lane] = {result_lo, result_hi};',
            '  }',
            '  for (uint32_t lane = 0; lane < wf.wf_size(); ++lane) {',
            '    if (!(exec & (1ULL << lane))) continue;',
            f'    write_pk_u64_pair({d}, wf, lane, results[lane]);',
            '  }',
        ]
    )


def gen_pk_mov_b32(
    dst: list[str],
    src: list[str],
    opsel_exprs: tuple[str, str] = ('', ''),
) -> str:
    """Generate V_PK_MOV_B32: move two 32-bit values based on op_sel."""
    d, s0, s1 = dst[0], src[0], src[1]
    L = []
    L.append('  uint64_t exec = wf.exec();')
    L.append('  for (uint32_t lane = 0; lane < wf.wf_size(); ++lane) {')
    L.append('    if (!(exec & (1ULL << lane))) continue;')
    for var, src in [('s0', s0), ('s1', s1)]:
        L.append(
            f'    uint64_t {var}_pair_w = amdgpu::RegisterAccess(wf).read_lane64({src}, lane);'
        )
        L.append(f'    uint32_t {var}_lo_w = static_cast<uint32_t>({var}_pair_w);')
        L.append(
            f'    uint32_t {var}_hi_w = static_cast<uint32_t>({var}_pair_w >> 32);'
        )
    opsel, opsel_hi = opsel_exprs
    s0_lo = _pk_f32_word_expr('s0', 'lo')
    s0_hi = _pk_f32_word_expr('s0', 'hi')
    s1_lo = _pk_f32_word_expr('s1', 'lo')
    s1_hi = _pk_f32_word_expr('s1', 'hi')
    L.append(f'    uint32_t lo = ({opsel} & 1) ? {s0_hi} : {s0_lo};')
    L.append(f'    uint32_t hi = ({opsel} & 2) ? {s1_hi} : {s1_lo};')
    L.append(
        f'    amdgpu::RegisterAccess(wf).write_lane64({d}, lane, static_cast<uint64_t>(lo) | (static_cast<uint64_t>(hi) << 32));'
    )
    L.append('  }')
    return '\n'.join(L)


# FP8 DOT4 semantic operation -> (src0 is FP8, src1 is FP8), as for FP8 WMMA.
_FP8_DOT4_FP8_SOURCES = {
    'fp8_fp8': ('true', 'true'),
    'fp8_bf8': ('true', 'false'),
    'bf8_fp8': ('false', 'true'),
    'bf8_bf8': ('false', 'false'),
}
