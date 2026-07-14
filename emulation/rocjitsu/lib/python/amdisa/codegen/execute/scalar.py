# Copyright (c) 2025-2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Scalar ALU execute body generators.

Free functions that emit C++ execute_impl bodies for scalar ALU
instructions: unary, binary, bit field extract, compare, compare-with-immediate,
bit compare, and saveexec.
"""

from __future__ import annotations


def gen_scalar_unary(
    dst: list[str], src: list[str], op: str | None, dtype: str | None, scc: str | None
) -> str:
    """Generate C++ execute_impl body for scalar unary ALU instructions (SOP1).

    Covers bitwise (not, brev, bcnt, ff0/ff1, flbit, wqm, quadmask),
    bit manipulation (bitset0/1), sign extension (sext8/16), count
    leading/trailing (clz, ctz, cls), FP rounding (ceil, floor, trunc,
    rndne), and type conversion (cvt_f32_i32, cvt_f16_f32, etc.).

    Args:
        dst: C++ destination operand expressions (e.g. ``['sdst']``).
        src: C++ source operand expressions (e.g. ``['ssrc0']``).
        op: Operation name derived from the mnemonic (e.g. ``'not'``,
            ``'brev'``, ``'cvt_f32_i32'``). None for unrecognized ops.
        dtype: Canonical data type (e.g. ``'b32'``, ``'b64'``, ``'f32'``).
            Determines register width (32 vs 64-bit read/write).
        scc: SCC update mode (``'nonzero'``, ``'none'``, or None).
            When ``'nonzero'``, emits ``wf.write_scc(result != 0)``.

    Returns:
        Multi-line C++ string for the execute_impl body.
    """
    L = []
    is_64 = dtype in ('b64', 'i64')

    # Special cases that don't follow the is_64 read/write pattern.
    if op == 'flbit_i32_i64':
        # 64-bit signed input → 32-bit output (find first bit of sign).
        L.append(f'  int64_t sval = static_cast<int64_t>({src[0]}.read_scalar64(wf));')
        L.append(
            '  uint64_t uval = sval < 0 ? ~static_cast<uint64_t>(sval) : static_cast<uint64_t>(sval);'
        )
        L.append(
            '  uint32_t result = uval == 0 ? static_cast<uint32_t>(-1) : static_cast<uint32_t>(std::countl_zero(uval));'
        )
        L.append(f'  {dst[0]}.write_scalar(wf, result);')
        if scc and scc != 'none':
            L.append('  wf.write_scc(result != 0);')
        return '\n'.join(L)

    if op == 'clz64':
        # 64-bit unsigned input → 32-bit output (count leading zeros).
        L.append(f'  uint64_t val = {src[0]}.read_scalar64(wf);')
        L.append(
            '  uint32_t result = val == 0 ? static_cast<uint32_t>(-1) : static_cast<uint32_t>(std::countl_zero(val));'
        )
        L.append(f'  {dst[0]}.write_scalar(wf, result);')
        if scc and scc != 'none':
            L.append('  wf.write_scc(result != 0);')
        return '\n'.join(L)

    if op == 'cls64':
        # 64-bit signed input → 32-bit output (count leading sign bits).
        L.append(f'  int64_t sval = static_cast<int64_t>({src[0]}.read_scalar64(wf));')
        L.append(
            '  uint64_t uval = sval < 0 ? ~static_cast<uint64_t>(sval) : static_cast<uint64_t>(sval);'
        )
        L.append(
            '  uint32_t result = uval == 0 ? 63u : static_cast<uint32_t>(std::countl_zero(uval)) - 1;'
        )
        L.append(f'  {dst[0]}.write_scalar(wf, result);')
        if scc and scc != 'none':
            L.append('  wf.write_scc(result != 0);')
        return '\n'.join(L)

    if op == 'ctz' and is_64:
        L.append(f'  uint64_t val = {src[0]}.read_scalar64(wf);')
        L.append(
            '  uint32_t result = val == 0 ? static_cast<uint32_t>(-1) : static_cast<uint32_t>(std::countr_zero(val));'
        )
        L.append(f'  {dst[0]}.write_scalar(wf, result);')
        if scc and scc != 'none':
            L.append('  wf.write_scc(result != 0);')
        return '\n'.join(L)

    if op == 'clz' and is_64:
        L.append(f'  uint64_t val = {src[0]}.read_scalar64(wf);')
        L.append(
            '  uint32_t result = val == 0 ? static_cast<uint32_t>(-1) : static_cast<uint32_t>(std::countl_zero(val));'
        )
        L.append(f'  {dst[0]}.write_scalar(wf, result);')
        if scc and scc != 'none':
            L.append('  wf.write_scc(result != 0);')
        return '\n'.join(L)

    if op == 'cls' and is_64:
        L.append(f'  int64_t sval = static_cast<int64_t>({src[0]}.read_scalar64(wf));')
        L.append(
            '  uint64_t uval = sval < 0 ? ~static_cast<uint64_t>(sval) : static_cast<uint64_t>(sval);'
        )
        L.append(
            '  uint32_t result = uval == 0 ? 63u : static_cast<uint32_t>(std::countl_zero(uval)) - 1;'
        )
        L.append(f'  {dst[0]}.write_scalar(wf, result);')
        if scc and scc != 'none':
            L.append('  wf.write_scc(result != 0);')
        return '\n'.join(L)

    if op in ('bitset0', 'bitset1') and is_64:
        # 32-bit input (bit index), 64-bit read-modify-write destination.
        L.append(f'  uint32_t bit = {src[0]}.read_scalar(wf);')
        if op == 'bitset0':
            L.append(
                f'  uint64_t result = {dst[0]}.read_scalar64(wf) & ~(1ULL << (bit & 63));'
            )
        else:
            L.append(
                f'  uint64_t result = {dst[0]}.read_scalar64(wf) | (1ULL << (bit & 63));'
            )
        L.append(f'  {dst[0]}.write_scalar64(wf, result);')
        return '\n'.join(L)

    if is_64:
        L.append(f'  uint64_t val = {src[0]}.read_scalar64(wf);')
        op_map = {
            'not': '~val',
            'wqm': '0; for (int q = 0; q < 16; ++q) if (val & (0xFULL << (q * 4))) result |= (0xFULL << (q * 4))',
            'bcnt0': 'static_cast<uint64_t>(std::popcount(~val))',
            'bcnt1': 'static_cast<uint64_t>(std::popcount(val))',
            'ff0': 'static_cast<uint64_t>(val == ~0ULL ? -1 : std::countr_zero(~val))',
            'ff1': 'static_cast<uint64_t>(val == 0 ? -1 : std::countr_zero(val))',
            'flbit': 'static_cast<uint64_t>(val == 0 ? -1 : std::countl_zero(val))',
            'bitrepl': 'val',
        }
        if op == 'brev':
            L.append('  uint64_t result = 0;')
            L.append(
                '  for (int i = 0; i < 64; ++i) result |= ((val >> i) & 1) << (63 - i);'
            )
        elif op == 'quadmask':
            L.append('  uint64_t result = 0;')
            L.append(
                '  for (int q = 0; q < 16; ++q) if (val & (0xFULL << (q * 4))) result |= (1ULL << q);'
            )
        elif op in op_map:
            L.append(f'  uint64_t result = {op_map[op]};')
        else:
            L.append(f'  uint64_t result = val; // unhandled: {op}')
        L.append(f'  {dst[0]}.write_scalar64(wf, result);')
        L.append('  wf.write_scc(result != 0);')
    else:
        if op == 'abs':
            # Use unsigned negation to avoid UB when val == INT_MIN.
            L.append(f'  int32_t val = static_cast<int32_t>({src[0]}.read_scalar(wf));')
            L.append('  uint32_t uval = static_cast<uint32_t>(val);')
            L.append('  uint32_t result = val < 0 ? (0u - uval) : uval;')
        elif op == 'sext8':
            L.append(f'  uint32_t val = {src[0]}.read_scalar(wf);')
            L.append(
                '  uint32_t result = static_cast<uint32_t>(static_cast<int32_t>(static_cast<int8_t>(val & 0xFF)));'
            )
        elif op == 'sext16':
            L.append(f'  uint32_t val = {src[0]}.read_scalar(wf);')
            L.append(
                '  uint32_t result = static_cast<uint32_t>(static_cast<int32_t>(static_cast<int16_t>(val & 0xFFFF)));'
            )
        elif op == 'flbit_i32':
            L.append(
                f'  int32_t sval = static_cast<int32_t>({src[0]}.read_scalar(wf));'
            )
            L.append(
                '  uint32_t val = sval < 0 ? ~static_cast<uint32_t>(sval) : static_cast<uint32_t>(sval);'
            )
            L.append(
                '  uint32_t result = val == 0 ? static_cast<uint32_t>(-1) : static_cast<uint32_t>(std::countl_zero(val));'
            )
        elif op == 'bitset0':
            # b64 case handled by early return above (is_64 branch).
            L.append(f'  uint32_t bit = {src[0]}.read_scalar(wf);')
            L.append(
                f'  uint32_t result = {dst[0]}.read_scalar(wf) & ~(1u << (bit & 31));'
            )
        elif op == 'bitset1':
            L.append(f'  uint32_t bit = {src[0]}.read_scalar(wf);')
            L.append(
                f'  uint32_t result = {dst[0]}.read_scalar(wf) | (1u << (bit & 31));'
            )
        else:
            L.append(f'  uint32_t val = {src[0]}.read_scalar(wf);')
            op_map = {
                'not': '~val',
                'wqm': '0; for (int q = 0; q < 8; ++q) if (val & (0xFu << (q * 4))) result |= (0xFu << (q * 4))',
                'bcnt0': 'static_cast<uint32_t>(std::popcount(~val))',
                'bcnt1': 'static_cast<uint32_t>(std::popcount(val))',
                'ff0': 'val == ~0u ? static_cast<uint32_t>(-1) : static_cast<uint32_t>(std::countr_zero(~val))',
                'ff1': 'val == 0 ? static_cast<uint32_t>(-1) : static_cast<uint32_t>(std::countr_zero(val))',
                'flbit': 'val == 0 ? static_cast<uint32_t>(-1) : static_cast<uint32_t>(std::countl_zero(val))',
                'quadmask': '0; for (int q = 0; q < 8; ++q) if (val & (0xFu << (q * 4))) result |= (1u << q)',
            }
            if op == 'brev':
                L.append('  uint32_t result = 0;')
                L.append(
                    '  for (int i = 0; i < 32; ++i) result |= ((val >> i) & 1) << (31 - i);'
                )
            elif op == 'ceil' and dtype == 'f32':
                L.append(
                    '  uint32_t result = std::bit_cast<uint32_t>(std::ceil(std::bit_cast<float>(val)));'
                )
            elif op == 'ceil' and dtype == 'f16':
                L.append(
                    '  float f = util::f16_to_f32(static_cast<uint16_t>(val & 0xFFFF));'
                )
                L.append(
                    '  uint32_t result = static_cast<uint32_t>(util::f32_to_f16(std::ceil(f)));'
                )
            elif op == 'floor' and dtype == 'f32':
                L.append(
                    '  uint32_t result = std::bit_cast<uint32_t>(std::floor(std::bit_cast<float>(val)));'
                )
            elif op == 'floor' and dtype == 'f16':
                L.append(
                    '  float f = util::f16_to_f32(static_cast<uint16_t>(val & 0xFFFF));'
                )
                L.append(
                    '  uint32_t result = static_cast<uint32_t>(util::f32_to_f16(std::floor(f)));'
                )
            elif op == 'trunc' and dtype == 'f32':
                L.append(
                    '  uint32_t result = std::bit_cast<uint32_t>(std::trunc(std::bit_cast<float>(val)));'
                )
            elif op == 'trunc' and dtype == 'f16':
                L.append(
                    '  float f = util::f16_to_f32(static_cast<uint16_t>(val & 0xFFFF));'
                )
                L.append(
                    '  uint32_t result = static_cast<uint32_t>(util::f32_to_f16(std::trunc(f)));'
                )
            elif op == 'rndne' and dtype == 'f32':
                L.append(
                    '  uint32_t result = std::bit_cast<uint32_t>(std::nearbyint(std::bit_cast<float>(val)));'
                )
            elif op == 'rndne' and dtype == 'f16':
                L.append(
                    '  float f = util::f16_to_f32(static_cast<uint16_t>(val & 0xFFFF));'
                )
                L.append(
                    '  uint32_t result = static_cast<uint32_t>(util::f32_to_f16(std::nearbyint(f)));'
                )
            elif op == 'cvt_f32_i32':
                L.append(
                    '  uint32_t result = std::bit_cast<uint32_t>(static_cast<float>(static_cast<int32_t>(val)));'
                )
            elif op == 'cvt_f32_u32':
                L.append(
                    '  uint32_t result = std::bit_cast<uint32_t>(static_cast<float>(val));'
                )
            elif op == 'cvt_i32_f32':
                L.append('  float f = std::bit_cast<float>(val);')
                L.append(
                    '  int32_t r = std::isnan(f) ? 0 : (f >= 2147483648.0f ? INT32_MAX : (f < -2147483648.0f ? INT32_MIN : static_cast<int32_t>(f)));'
                )
                L.append('  uint32_t result = static_cast<uint32_t>(r);')
            elif op == 'cvt_u32_f32':
                L.append('  float f = std::bit_cast<float>(val);')
                L.append(
                    '  uint32_t result = (std::isnan(f) || f < 0.0f) ? 0u : (f >= 4294967296.0f ? UINT32_MAX : static_cast<uint32_t>(f));'
                )
            elif op == 'cvt_f16_f32':
                L.append(
                    '  uint32_t result = static_cast<uint32_t>(util::f32_to_f16(std::bit_cast<float>(val)));'
                )
            elif op == 'cvt_f32_f16':
                L.append(
                    '  uint32_t result = std::bit_cast<uint32_t>(util::f16_to_f32(static_cast<uint16_t>(val & 0xFFFF)));'
                )
            elif op == 'cvt_hi_f32_f16':
                L.append(
                    '  uint32_t result = std::bit_cast<uint32_t>(util::f16_to_f32(static_cast<uint16_t>((val >> 16) & 0xFFFF)));'
                )
            elif op == 'ctz':
                L.append(
                    '  uint32_t result = val == 0 ? static_cast<uint32_t>(-1) : static_cast<uint32_t>(std::countr_zero(val));'
                )
            elif op == 'clz':
                L.append(
                    '  uint32_t result = val == 0 ? static_cast<uint32_t>(-1) : static_cast<uint32_t>(std::countl_zero(val));'
                )
            elif op == 'cls':
                # Count leading sign bits: number of consecutive bits matching the sign bit.
                L.append('  int32_t sval = static_cast<int32_t>(val);')
                L.append(
                    '  uint32_t uval = sval < 0 ? ~static_cast<uint32_t>(sval) : static_cast<uint32_t>(sval);'
                )
                L.append(
                    '  uint32_t result = uval == 0 ? 31u : static_cast<uint32_t>(std::countl_zero(uval)) - 1;'
                )
            elif op in op_map:
                L.append(f'  uint32_t result = {op_map[op]};')
            else:
                L.append(f'  uint32_t result = val; // TODO: {op}')
        L.append(f'  {dst[0]}.write_scalar(wf, result);')
        if scc and scc != 'none':
            L.append('  wf.write_scc(result != 0);')

    return '\n'.join(L)


def gen_scalar_binop(
    dst: list[str], src: list[str], op: str | None, dtype: str | None, scc: str | None
) -> str:
    """Generate C++ execute_impl body for scalar binary ALU instructions (SOP2).

    Covers arithmetic (add, sub, addc, subb, mul, mulhi, absdiff),
    bitwise (and, or, xor, nand, nor, xnor, andn2, orn2), shifts
    (shl, shr, ashr), min/max, bit field (bfm, bfe via gen_scalar_bfe),
    shift-add (lshl1_add through lshl4_add), pack (pack_ll/lh/hh),
    and scalar FP (f32/f16 add/sub/mul/min/max/fma).

    Args:
        dst: C++ destination operand expressions (e.g. ``['sdst']``).
        src: C++ source operand expressions (e.g. ``['ssrc0', 'ssrc1']``).
        op: Operation name (e.g. ``'add'``, ``'and'``, ``'shl'``).
        dtype: Canonical data type (e.g. ``'u32'``, ``'i32'``, ``'b64'``,
            ``'f32'``). Controls signedness, width, and FP dispatch.
        scc: SCC update mode (``'carry'``, ``'borrow'``, ``'overflow'``,
            ``'nonzero'``, ``'compare'``, ``'none'``, or None).

    Returns:
        Multi-line C++ string for the execute_impl body.
    """
    L = []
    is_64 = dtype in ('b64', 'i64', 'u64')

    if is_64:
        if dtype == 'i64':
            L.append(
                f'  int64_t s0 = static_cast<int64_t>({src[0]}.read_scalar64(wf));'
            )
            L.append(
                f'  int64_t s1 = static_cast<int64_t>({src[1]}.read_scalar64(wf));'
            )
        else:
            L.append(f'  uint64_t s0 = {src[0]}.read_scalar64(wf);')
            L.append(f'  uint64_t s1 = {src[1]}.read_scalar64(wf);')
    elif dtype in ('i32',):
        L.append(f'  int32_t s0 = static_cast<int32_t>({src[0]}.read_scalar(wf));')
        L.append(f'  int32_t s1 = static_cast<int32_t>({src[1]}.read_scalar(wf));')
    else:
        L.append(f'  uint32_t s0 = {src[0]}.read_scalar(wf);')
        L.append(f'  uint32_t s1 = {src[1]}.read_scalar(wf);')

    # Compute result
    if is_64 and op == 'mul':
        L.append(f'  {dst[0]}.write_scalar64(wf, static_cast<uint64_t>(s0 * s1));')
    elif dtype in ('i32',) and op in ('add', 'sub'):
        sign = '+' if op == 'add' else '-'
        L.append(
            f'  int64_t wide = static_cast<int64_t>(s0) {sign} static_cast<int64_t>(s1);'
        )
        L.append('  int32_t result = static_cast<int32_t>(wide);')
        L.append(f'  {dst[0]}.write_scalar(wf, static_cast<uint32_t>(result));')
        L.append('  wf.write_scc(wide != static_cast<int64_t>(result));')
    elif dtype in ('u32',) and op == 'add':
        L.append(
            '  uint64_t wide = static_cast<uint64_t>(s0) + static_cast<uint64_t>(s1);'
        )
        L.append(f'  {dst[0]}.write_scalar(wf, static_cast<uint32_t>(wide));')
        L.append('  wf.write_scc(wide > 0xFFFFFFFFULL);')
    elif dtype in ('u32',) and op == 'sub':
        L.append(f'  {dst[0]}.write_scalar(wf, s0 - s1);')
        L.append('  wf.write_scc(s0 < s1);')
    elif dtype in ('u32',) and op == 'addc':
        L.append(
            '  uint64_t wide = static_cast<uint64_t>(s0) + static_cast<uint64_t>(s1) + (wf.read_scc() ? 1u : 0u);'
        )
        L.append(f'  {dst[0]}.write_scalar(wf, static_cast<uint32_t>(wide));')
        L.append('  wf.write_scc(wide > 0xFFFFFFFFULL);')
    elif dtype in ('u32',) and op == 'subb':
        L.append('  uint32_t bin = wf.read_scc() ? 1u : 0u;')
        L.append(
            '  uint64_t wide = static_cast<uint64_t>(s0) - static_cast<uint64_t>(s1) - bin;'
        )
        L.append(f'  {dst[0]}.write_scalar(wf, static_cast<uint32_t>(wide));')
        L.append(
            '  wf.write_scc(static_cast<uint64_t>(s0) < static_cast<uint64_t>(s1) + bin);'
        )
    elif dtype in ('i32',) and op == 'mul':
        # Use unsigned multiply to avoid signed overflow UB. The lower 32
        # bits are identical for signed and unsigned multiplication.
        L.append(
            f'  {dst[0]}.write_scalar(wf, static_cast<uint32_t>(static_cast<uint32_t>(s0) * static_cast<uint32_t>(s1)));'
        )
    elif dtype in ('u32',) and op == 'mul':
        L.append(f'  {dst[0]}.write_scalar(wf, s0 * s1);')
    elif op == 'mulhi':
        if dtype in ('u32',):
            L.append(
                '  uint64_t wide = static_cast<uint64_t>(s0) * static_cast<uint64_t>(s1);'
            )
            L.append(f'  {dst[0]}.write_scalar(wf, static_cast<uint32_t>(wide >> 32));')
        else:
            L.append(
                '  int64_t wide = static_cast<int64_t>(s0) * static_cast<int64_t>(s1);'
            )
            L.append(
                f'  {dst[0]}.write_scalar(wf, static_cast<uint32_t>(static_cast<uint64_t>(wide) >> 32));'
            )
    elif op == 'min':
        if dtype in ('i32',):
            L.append(f'  int32_t result = s0 < s1 ? s0 : s1;')
            L.append(f'  {dst[0]}.write_scalar(wf, static_cast<uint32_t>(result));')
            L.append('  wf.write_scc(s0 < s1);')
        else:
            L.append(f'  uint32_t result = s0 < s1 ? s0 : s1;')
            L.append(f'  {dst[0]}.write_scalar(wf, result);')
            L.append('  wf.write_scc(s0 < s1);')
    elif op == 'max':
        if dtype in ('i32',):
            L.append(f'  int32_t result = s0 > s1 ? s0 : s1;')
            L.append(f'  {dst[0]}.write_scalar(wf, static_cast<uint32_t>(result));')
            L.append('  wf.write_scc(s0 > s1);')
        else:
            L.append(f'  uint32_t result = s0 > s1 ? s0 : s1;')
            L.append(f'  {dst[0]}.write_scalar(wf, result);')
            L.append('  wf.write_scc(s0 > s1);')
    elif op == 'absdiff':
        # Use int64_t to avoid signed overflow UB when s0 and s1 are
        # near INT32_MIN/INT32_MAX (e.g., INT32_MAX - INT32_MIN).
        L.append('  int64_t wide_s0 = static_cast<int64_t>(s0);')
        L.append('  int64_t wide_s1 = static_cast<int64_t>(s1);')
        L.append(
            '  uint32_t result = static_cast<uint32_t>(wide_s0 > wide_s1 ? (wide_s0 - wide_s1) : (wide_s1 - wide_s0));'
        )
        L.append(f'  {dst[0]}.write_scalar(wf, result);')
        L.append('  wf.write_scc(result != 0);')
    elif op == 'bfm':
        if is_64:
            L.append('  uint64_t count = s0 & 63u;')
            L.append('  uint64_t offset = s1 & 63u;')
            L.append(
                '  uint64_t result = count == 0 ? 0 : ((1ULL << count) - 1) << offset;'
            )
            L.append(f'  {dst[0]}.write_scalar64(wf, result);')
        else:
            L.append('  uint32_t result = ::rocjitsu::amdgpu::bfm_b32(s0, s1);')
            L.append(f'  {dst[0]}.write_scalar(wf, result);')
    elif op == 'bfe':
        return gen_scalar_bfe(dst, src, dtype)
    elif op in ('lshl1_add', 'lshl2_add', 'lshl3_add', 'lshl4_add'):
        shift = op[4]  # extract digit
        L.append(
            f'  uint64_t wide = (static_cast<uint64_t>(s0) << {shift}u) + static_cast<uint64_t>(s1);'
        )
        L.append(f'  {dst[0]}.write_scalar(wf, static_cast<uint32_t>(wide));')
        L.append('  wf.write_scc(wide > 0xFFFFFFFFULL);')
    elif dtype == 'f32' and op in (
        'add',
        'sub',
        'mul',
        'min',
        'max',
        'min_num',
        'max_num',
        'minimum',
        'maximum',
        'fma',
    ):
        fp_op = {
            'add': 'f0 + f1',
            'sub': 'f0 - f1',
            'mul': 'f0 * f1',
            'min': 'std::fmin(f0, f1)',
            'max': 'std::fmax(f0, f1)',
            'min_num': 'std::fmin(f0, f1)',
            'max_num': 'std::fmax(f0, f1)',
            'minimum': '((std::isnan(f0) || std::isnan(f1)) ? std::numeric_limits<float>::quiet_NaN() : (f0 == f1 ? (std::signbit(f0) ? f0 : f1) : (f0 < f1 ? f0 : f1)))',
            'maximum': '((std::isnan(f0) || std::isnan(f1)) ? std::numeric_limits<float>::quiet_NaN() : (f0 == f1 ? (std::signbit(f0) ? f1 : f0) : (f0 > f1 ? f0 : f1)))',
            'fma': 'std::fma(f0, f1, std::bit_cast<float>(static_cast<uint32_t>(wf.read_scc())))',
        }
        L.append('  float f0 = std::bit_cast<float>(s0);')
        L.append('  float f1 = std::bit_cast<float>(s1);')
        L.append(f'  float fr = {fp_op[op]};')
        L.append(f'  {dst[0]}.write_scalar(wf, std::bit_cast<uint32_t>(fr));')
    elif dtype == 'f16' and op in (
        'add',
        'sub',
        'mul',
        'min',
        'max',
        'min_num',
        'max_num',
        'minimum',
        'maximum',
    ):
        fp_op = {
            'add': 'f0 + f1',
            'sub': 'f0 - f1',
            'mul': 'f0 * f1',
            'min': 'std::fmin(f0, f1)',
            'max': 'std::fmax(f0, f1)',
            'min_num': 'std::fmin(f0, f1)',
            'max_num': 'std::fmax(f0, f1)',
            'minimum': '((std::isnan(f0) || std::isnan(f1)) ? std::numeric_limits<float>::quiet_NaN() : (f0 == f1 ? (std::signbit(f0) ? f0 : f1) : (f0 < f1 ? f0 : f1)))',
            'maximum': '((std::isnan(f0) || std::isnan(f1)) ? std::numeric_limits<float>::quiet_NaN() : (f0 == f1 ? (std::signbit(f0) ? f1 : f0) : (f0 > f1 ? f0 : f1)))',
        }
        L.append('  float f0 = util::f16_to_f32(static_cast<uint16_t>(s0 & 0xFFFF));')
        L.append('  float f1 = util::f16_to_f32(static_cast<uint16_t>(s1 & 0xFFFF));')
        L.append(f'  float fr = {fp_op[op]};')
        L.append(
            f'  {dst[0]}.write_scalar(wf, static_cast<uint32_t>(util::f32_to_f16(fr)));'
        )
    elif op == 'pack_ll':
        L.append(
            f'  {dst[0]}.write_scalar(wf, (s0 & 0xFFFFu) | ((s1 & 0xFFFFu) << 16));'
        )
    elif op == 'pack_lh':
        L.append(f'  {dst[0]}.write_scalar(wf, (s0 & 0xFFFFu) | (s1 & 0xFFFF0000u));')
    elif op == 'pack_hh':
        L.append(
            f'  {dst[0]}.write_scalar(wf, ((s0 >> 16) & 0xFFFFu) | (s1 & 0xFFFF0000u));'
        )
    elif op == 'pack_hl':
        L.append(
            f'  {dst[0]}.write_scalar(wf, ((s0 >> 16) & 0xFFFFu) | ((s1 & 0xFFFFu) << 16));'
        )
    elif op == 'cvt_pkrtz_f16_f32':
        L.append('  float f0 = std::bit_cast<float>(s0);')
        L.append('  float f1 = std::bit_cast<float>(s1);')
        L.append(
            f'  {dst[0]}.write_scalar(wf, static_cast<uint32_t>(util::f32_to_f16_rtz(f0)) | (static_cast<uint32_t>(util::f32_to_f16_rtz(f1)) << 16));'
        )
    else:
        # Bitwise / shift ops
        utype = 'uint64_t' if is_64 else 'uint32_t'
        mask = 63 if is_64 else 31
        op_map = {
            'and': 's0 & s1',
            'or': 's0 | s1',
            'xor': 's0 ^ s1',
            'nand': '~(s0 & s1)',
            'nor': '~(s0 | s1)',
            'xnor': '~(s0 ^ s1)',
            'andn2': 's0 & ~s1',
            'orn2': 's0 | ~s1',
            'shl': f's0 << (s1 & {mask}u)',
            'shr': f's0 >> (s1 & {mask}u)',
        }
        if dtype == 'i32' and op == 'ashr':
            L.append(f'  int32_t result = s0 >> (s1 & 31);')
            L.append(f'  {dst[0]}.write_scalar(wf, static_cast<uint32_t>(result));')
        elif dtype == 'i64' and op == 'ashr':
            L.append(f'  int64_t result = s0 >> (s1 & 63);')
            L.append(f'  {dst[0]}.write_scalar64(wf, static_cast<uint64_t>(result));')
        elif op in op_map:
            L.append(f'  {utype} result = {op_map[op]};')
            if is_64:
                L.append(f'  {dst[0]}.write_scalar64(wf, result);')
            else:
                L.append(f'  {dst[0]}.write_scalar(wf, result);')
        else:
            L.append('  (void)s1;')
            L.append(f'  {utype} result = s0; // TODO: op={op}')
            if is_64:
                L.append(f'  {dst[0]}.write_scalar64(wf, result);')
            else:
                L.append(f'  {dst[0]}.write_scalar(wf, result);')

        # SCC
        if scc == 'nonzero':
            L.append('  wf.write_scc(result != 0);')

    return '\n'.join(L)


def gen_scalar_bfe(dst: list[str], src: list[str], dtype: str | None) -> str:
    """Generate C++ execute_impl body for scalar bit field extract (S_BFE_*).

    Extracts a bitfield from src[0] using offset and width from src[1].
    For signed types (i32/i64), sign-extends the extracted field.
    Always sets SCC = (result != 0).

    Args:
        dst: C++ destination operand expressions.
        src: C++ source operand expressions. src[0] is the base value,
            src[1] encodes offset (bits[4:0]) and width (bits[22:16]).
        dtype: Data type (``'u32'``, ``'i32'``, ``'u64'``, ``'i64'``).
            Signed types trigger sign extension of the extracted field.

    Returns:
        Multi-line C++ string for the execute_impl body.
    """
    L = []
    if dtype in ('u64', 'i64'):
        L.append(f'  uint64_t base = {src[0]}.read_scalar64(wf);')
        L.append(f'  uint32_t field = {src[1]}.read_scalar(wf);')
        L.append('  uint32_t offset = field & 63u;')
        L.append('  uint32_t width = (field >> 16) & 127u;')
        L.append('  if (width == 0) {')
        L.append(f'    {dst[0]}.write_scalar64(wf, 0);')
        L.append('    wf.write_scc(false);')
        L.append('  } else {')
        L.append('    if (offset + width > 64) width = 64 - offset;')
        L.append('    uint64_t mask = width >= 64 ? ~0ULL : ((1ULL << width) - 1);')
        L.append('    uint64_t extracted = (base >> offset) & mask;')
        if dtype == 'i64':
            L.append('    if (width < 64 && (extracted & (1ULL << (width - 1))))')
            L.append('      extracted |= ~mask;')
        L.append(f'    {dst[0]}.write_scalar64(wf, extracted);')
        L.append('    wf.write_scc(extracted != 0);')
        L.append('  }')
    else:
        L.append(f'  uint32_t base = {src[0]}.read_scalar(wf);')
        L.append(f'  uint32_t field = {src[1]}.read_scalar(wf);')
        L.append('  uint32_t offset = field & 31u;')
        L.append('  uint32_t width = (field >> 16) & 127u;')
        L.append('  if (width == 0) {')
        L.append(f'    {dst[0]}.write_scalar(wf, 0);')
        L.append('    wf.write_scc(false);')
        L.append('  } else {')
        L.append('    if (offset + width > 32) width = 32 - offset;')
        L.append('    uint32_t mask = width >= 32 ? ~0u : ((1u << width) - 1);')
        L.append('    uint32_t extracted = (base >> offset) & mask;')
        if dtype == 'i32':
            L.append('    if (width < 32 && (extracted & (1u << (width - 1))))')
            L.append('      extracted |= ~mask;')
        L.append(f'    {dst[0]}.write_scalar(wf, extracted);')
        L.append('    wf.write_scc(extracted != 0);')
        L.append('  }')
    return '\n'.join(L)


def gen_scalar_cmp(src: list[str], op: str | None, dtype: str | None) -> str:
    """Generate C++ execute_impl body for scalar compare instructions (SOPC).

    Compares two scalar operands and writes the result to SCC.
    No destination register — the only output is SCC.

    Args:
        src: C++ source operand expressions (e.g. ``['ssrc0', 'ssrc1']``).
        op: Comparison operation (``'eq'``, ``'ne'``, ``'lt'``, ``'le'``,
            ``'gt'``, ``'ge'``, ``'lg'``).
        dtype: Data type controlling signedness and width
            (``'u32'``, ``'i32'``, ``'u64'``, ``'i64'``).

    Returns:
        Multi-line C++ string for the execute_impl body.
    """
    L = []
    cmp_map = {
        'eq': '==',
        'ne': '!=',
        'lg': '!=',
        'gt': '>',
        'ge': '>=',
        'lt': '<',
        'le': '<=',
    }
    if dtype in ('i32',):
        L.append(f'  int32_t s0 = static_cast<int32_t>({src[0]}.read_scalar(wf));')
        L.append(f'  int32_t s1 = static_cast<int32_t>({src[1]}.read_scalar(wf));')
    elif dtype in ('i64',):
        L.append(f'  int64_t s0 = static_cast<int64_t>({src[0]}.read_scalar64(wf));')
        L.append(f'  int64_t s1 = static_cast<int64_t>({src[1]}.read_scalar64(wf));')
    elif dtype in ('u64',):
        L.append(f'  uint64_t s0 = {src[0]}.read_scalar64(wf);')
        L.append(f'  uint64_t s1 = {src[1]}.read_scalar64(wf);')
    elif dtype == 'f64':
        L.append(f'  double s0 = std::bit_cast<double>({src[0]}.read_scalar64(wf));')
        L.append(f'  double s1 = std::bit_cast<double>({src[1]}.read_scalar64(wf));')
    elif dtype == 'f32':
        L.append(f'  float s0 = std::bit_cast<float>({src[0]}.read_scalar(wf));')
        L.append(f'  float s1 = std::bit_cast<float>({src[1]}.read_scalar(wf));')
    elif dtype == 'f16':
        L.append(
            f'  float s0 = util::f16_to_f32(static_cast<uint16_t>({src[0]}.read_scalar(wf) & 0xFFFF));'
        )
        L.append(
            f'  float s1 = util::f16_to_f32(static_cast<uint16_t>({src[1]}.read_scalar(wf) & 0xFFFF));'
        )
    else:
        L.append(f'  uint32_t s0 = {src[0]}.read_scalar(wf);')
        L.append(f'  uint32_t s1 = {src[1]}.read_scalar(wf);')
    L.append(f'  wf.write_scc(s0 {cmp_map[op]} s1);')
    return '\n'.join(L)


def gen_scalar_cmpk(
    dst: list[str], src: list[str], op: str | None, dtype: str | None
) -> str:
    """Generate C++ execute_impl body for scalar compare-with-immediate (SOPK).

    Compares an SGPR against a 16-bit inline immediate and sets SCC.
    The SOPK encoding uses ``sdst`` as the register operand (read-only
    for compares) and ``simm16`` as the immediate.

    Args:
        dst: C++ destination operand expressions. For SOPK compares,
            dst[0] is actually the register being compared (read-only).
        src: C++ source operand expressions. src[0] is the immediate.
        op: Comparison operation (``'eq'``, ``'ne'``, ``'lt'``, etc.).
        dtype: Data type (``'u32'`` or ``'i32'``). Signed types
            sign-extend the 16-bit immediate before comparison.

    Returns:
        Multi-line C++ string for the execute_impl body.
    """
    L = []
    cmp_map = {
        'eq': '==',
        'ne': '!=',
        'lg': '!=',
        'gt': '>',
        'ge': '>=',
        'lt': '<',
        'le': '<=',
    }
    if dtype in ('i32',):
        L.append(f'  int32_t s0 = static_cast<int32_t>({dst[0]}.read_scalar(wf));')
        L.append(f'  int32_t imm = static_cast<int16_t>({src[0]}.encoding_value_);')
    else:
        L.append(f'  uint32_t s0 = {dst[0]}.read_scalar(wf);')
        L.append(
            f'  uint32_t imm = static_cast<uint32_t>(static_cast<uint16_t>({src[0]}.encoding_value_));'
        )
    L.append(f'  wf.write_scc(s0 {cmp_map[op]} imm);')
    return '\n'.join(L)


def gen_scalar_bitcmp(src: list[str], op: str | None, dtype: str | None) -> str:
    """Generate C++ execute_impl body for scalar bit compare (S_BITCMP0/1_*).

    Tests a single bit in src[0] at the position given by src[1] and
    sets SCC. BITCMP0 sets SCC if the bit is 0; BITCMP1 sets SCC if
    the bit is 1.

    Args:
        src: C++ source operand expressions. src[0] is the value to test,
            src[1] is the bit index.
        op: ``'bitcmp0'`` (test for zero) or ``'bitcmp1'`` (test for one).
        dtype: Data type (``'b32'`` or ``'b64'``). Controls the bit
            index mask (31 vs 63).

    Returns:
        Multi-line C++ string for the execute_impl body.
    """
    L = []
    if dtype in ('b64',):
        L.append(f'  uint64_t val = {src[0]}.read_scalar64(wf);')
        L.append(f'  uint32_t bit = {src[1]}.read_scalar(wf) & 63u;')
    else:
        L.append(f'  uint32_t val = {src[0]}.read_scalar(wf);')
        L.append(f'  uint32_t bit = {src[1]}.read_scalar(wf) & 31u;')
    if op == 'bitcmp0':
        L.append('  wf.write_scc(!(val & (1ULL << bit)));')
    else:
        L.append('  wf.write_scc((val & (1ULL << bit)) != 0);')
    return '\n'.join(L)


def gen_scalar_saveexec(dst: list[str], src: list[str], op: str | None) -> str:
    """Generate C++ execute_impl body for saveexec instructions (S_*_SAVEEXEC_B64).

    Saves the current EXEC mask to sdst, computes a new EXEC from the
    bitwise operation between old EXEC and ssrc0, and sets SCC = (new != 0).

    Per the ISA spec, all sources are read before any destination is written.
    Reading ssrc0 before writing sdst prevents aliasing bugs when sdst == ssrc0.

    Args:
        dst: C++ destination operand expressions (e.g. ``['sdst']``).
            Receives the old EXEC value.
        src: C++ source operand expressions (e.g. ``['ssrc0']``).
            The 64-bit mask combined with old EXEC via the bitwise op.
        op: Bitwise operation (``'and'``, ``'or'``, ``'xor'``, ``'nand'``,
            ``'nor'``, ``'xnor'``, ``'andn1'``, ``'andn2'``, ``'orn1'``,
            ``'orn2'``, and RDNA3/4 ``'and_not0'``, ``'or_not0'``, etc.).

    Returns:
        Multi-line C++ string for the execute_impl body.
    """
    L = []
    L.append('  uint64_t old_exec = wf.exec();')
    L.append(f'  uint64_t src = {src[0]}.read_scalar64(wf);')
    L.append(f'  {dst[0]}.write_scalar64(wf, old_exec);')
    saveexec_map = {
        'and': 'old_exec & src',
        'or': 'old_exec | src',
        'xor': 'old_exec ^ src',
        'nand': '~(old_exec & src)',
        'nor': '~(old_exec | src)',
        'xnor': '~(old_exec ^ src)',
        'andn1': '~src & old_exec',
        'andn2': 'src & ~old_exec',
        'orn1': '~src | old_exec',
        'orn2': 'src | ~old_exec',
        # RDNA3/4 not0/not1 variants
        'and_not0': 'old_exec & ~src',
        'or_not0': 'old_exec | ~src',
        'and_not1': 'src & ~old_exec',
        'or_not1': 'src | ~old_exec',
    }
    if op not in saveexec_map:
        L.append('  (void)src;')
    expr = saveexec_map.get(op, f'old_exec /* TODO: {op} */')
    L.append(f'  uint64_t result = {expr};')
    L.append(f'  util::Logger::vm([&](auto &os) {{')
    L.append(
        f'    os << std::format("saveexec ssrc0_ev={{}} src={{:#x}} exec={{:#x}}->{{:#x}}",'
    )
    L.append(
        f'                      {src[0]}.encoding_value(), src, old_exec, result);'
    )
    L.append(f'  }});')
    L.append('  wf.set_exec(result);')
    L.append('  wf.write_scc(result != 0);')
    return '\n'.join(L)
