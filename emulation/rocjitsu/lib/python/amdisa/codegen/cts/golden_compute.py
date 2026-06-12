# Copyright (c) 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Golden reference computation for CTS test cases.

Ground truth comes from the MATHEMATICAL DEFINITION of each operation,
computed via independent libraries (Python ints, numpy, mpmath).
We never reimplement ISA pseudocode.
"""

from __future__ import annotations

import struct
from dataclasses import dataclass

import numpy as np

U32 = 0xFFFF_FFFF
U64 = 0xFFFF_FFFF_FFFF_FFFF


@dataclass
class GoldenResult:
    output_bits: int
    scc: bool | None = None
    tolerance_ulps: int = 0


def _to_i32(u: int) -> int:
    u &= U32
    return u - (1 << 32) if u >= (1 << 31) else u


def _to_u32(i: int) -> int:
    return i & U32


def _to_i64(u: int) -> int:
    u &= U64
    return u - (1 << 64) if u >= (1 << 63) else u


def _popcount(x: int, bits: int = 32) -> int:
    return bin(x & ((1 << bits) - 1)).count('1')


def _ff0(x: int, bits: int = 32) -> int:
    """Find first zero bit from LSB. Returns -1 if all ones."""
    mask = (1 << bits) - 1
    x &= mask
    for i in range(bits):
        if not (x & (1 << i)):
            return i
    return -1 & U32


def _ff1(x: int, bits: int = 32) -> int:
    """Find first one bit from LSB. Returns -1 if zero."""
    mask = (1 << bits) - 1
    x &= mask
    for i in range(bits):
        if x & (1 << i):
            return i
    return -1 & U32


def _flbit_unsigned(x: int, bits: int = 32) -> int:
    """Find position of highest set bit from MSB (unsigned). Returns -1 if 0."""
    mask = (1 << bits) - 1
    x &= mask
    if x == 0:
        return -1 & U32
    for i in range(bits - 1, -1, -1):
        if x & (1 << i):
            return (bits - 1) - i
    return -1 & U32


def _flbit_signed(x: int) -> int:
    """Find position of highest bit different from sign bit (signed)."""
    x = _to_i32(x & U32)
    if x == 0 or x == -1:
        return -1 & U32
    if x < 0:
        x = ~x & U32
    else:
        x = x & U32
    for i in range(31, -1, -1):
        if x & (1 << i):
            return 31 - i
    return -1 & U32


def _brev(x: int, bits: int = 32) -> int:
    """Bit reverse."""
    result = 0
    for i in range(bits):
        if x & (1 << i):
            result |= 1 << (bits - 1 - i)
    return result


def _sext(x: int, from_bits: int) -> int:
    """Sign-extend from from_bits to 32 bits."""
    mask = (1 << from_bits) - 1
    x &= mask
    sign = x & (1 << (from_bits - 1))
    if sign:
        x |= U32 & ~mask
    return x & U32


def _wqm(x: int, bits: int = 32) -> int:
    """Whole quad mode — set all bits in each nibble if any bit is set."""
    result = 0
    for i in range(0, bits, 4):
        nibble = (x >> i) & 0xF
        if nibble:
            result |= 0xF << i
    return result & ((1 << bits) - 1)


def _bfe_u32(base: int, offset_width: int) -> int:
    offset = offset_width & 0x1F
    width = (offset_width >> 16) & 0x7F
    if width == 0:
        return 0
    if offset + width > 32:
        width = 32 - offset
    return (base >> offset) & ((1 << width) - 1)


def _bfe_i32(base: int, offset_width: int) -> int:
    offset = offset_width & 0x1F
    width = (offset_width >> 16) & 0x7F
    if width == 0:
        return 0
    if offset + width > 32:
        width = 32 - offset
    extracted = (base >> offset) & ((1 << width) - 1)
    if extracted & (1 << (width - 1)):
        extracted |= U32 & ~((1 << width) - 1)
    return extracted & U32


def _bfm(width: int, offset: int) -> int:
    width &= 0x1F
    offset &= 0x1F
    if width == 0:
        return 0
    return (((1 << width) - 1) << offset) & U32


def _scc_nonzero(result: int, bits: int = 32) -> bool:
    return (result & ((1 << bits) - 1)) != 0


def _scc_carry_u32(a: int, b: int) -> bool:
    return (a & U32) + (b & U32) > U32


def _scc_borrow_u32(a: int, b: int) -> bool:
    return (a & U32) < (b & U32)


def _scc_overflow_i32(a: int, b: int, result: int) -> bool:
    sa = (a >> 31) & 1
    sb = (b >> 31) & 1
    sr = (result >> 31) & 1
    return sa == sb and sa != sr


# ---------------------------------------------------------------------------
#  Scalar unary operations
# ---------------------------------------------------------------------------


def _scalar_unary(op: str, dtype: str | None, src: int) -> GoldenResult:
    """Compute golden for a scalar unary instruction."""
    s = src & U32
    result: int

    if op == 'not':
        result = (~s) & U32
    elif op == 'brev':
        result = _brev(s)
    elif op == 'bcnt0':
        result = 32 - _popcount(s)
    elif op == 'bcnt1':
        result = _popcount(s)
    elif op == 'ff0':
        result = _ff0(s)
    elif op == 'ff1':
        result = _ff1(s)
    elif op == 'flbit':
        result = _flbit_unsigned(s)
    elif op == 'flbit_i32':
        result = _flbit_signed(s)
    elif op == 'abs':
        signed_val = _to_i32(s)
        result = _to_u32(abs(signed_val))
    elif op == 'sext8':
        result = _sext(s, 8)
    elif op == 'sext16':
        result = _sext(s, 16)
    elif op == 'wqm':
        result = _wqm(s)
    elif op == 'bitset0':
        bit_pos = s & 0x1F
        result = s & ~(1 << bit_pos)
        return GoldenResult(result & U32, scc=None)
    elif op == 'bitset1':
        bit_pos = s & 0x1F
        result = s | (1 << bit_pos)
        return GoldenResult(result & U32, scc=None)
    elif op == 'ctz':
        result = _ff1(s)
    elif op == 'clz':
        if s == 0:
            result = 32
        else:
            result = 0
            for i in range(31, -1, -1):
                if s & (1 << i):
                    result = 31 - i
                    break
    elif op == 'cls':
        signed_val = _to_i32(s)
        if signed_val == 0 or signed_val == -1:
            result = 31
        else:
            x = (~s if signed_val < 0 else s) & U32
            result = 0
            for i in range(30, -1, -1):
                if x & (1 << i):
                    result = 30 - i
                    break
    elif op == 'quadmask':
        result = 0
        for i in range(0, 32, 4):
            nibble = (s >> i) & 0xF
            if nibble:
                result |= 1 << (i // 4)
    elif op == 'cvt_f32_i32':
        result = struct.unpack('<I', struct.pack('<f', np.float32(_to_i32(s))))[0]
    elif op == 'cvt_f32_u32':
        result = struct.unpack('<I', struct.pack('<f', np.float32(s)))[0]
    elif op == 'cvt_i32_f32':
        f = struct.unpack('<f', struct.pack('<I', s))[0]
        if np.isnan(f):
            result = 0
        elif f >= 2147483648.0:
            result = 0x7FFF_FFFF
        elif f <= -2147483649.0:
            result = 0x8000_0000
        else:
            result = _to_u32(int(np.trunc(np.float32(f))))
    elif op == 'cvt_u32_f32':
        f = struct.unpack('<f', struct.pack('<I', s))[0]
        if np.isnan(f) or f < 0:
            result = 0
        elif f >= 4294967296.0:
            result = U32
        else:
            result = int(np.trunc(np.float32(f))) & U32
    else:
        return GoldenResult(0, scc=None)

    result &= U32
    scc = (
        _scc_nonzero(result)
        if op not in ('bitset0', 'bitset1', 'sext8', 'sext16')
        else None
    )
    return GoldenResult(result, scc=scc)


# ---------------------------------------------------------------------------
#  Scalar binary operations
# ---------------------------------------------------------------------------


def _scalar_binop(op: str, dtype: str | None, src0: int, src1: int) -> GoldenResult:
    """Compute golden for a scalar binary instruction."""
    a = src0 & U32
    b = src1 & U32
    result: int
    scc: bool | None = None

    if op == 'add':
        result = (a + b) & U32
        if dtype in ('u32',):
            scc = _scc_carry_u32(a, b)
        elif dtype in ('i32',):
            scc = _scc_overflow_i32(a, b, result)
        else:
            scc = _scc_nonzero(result)
    elif op == 'sub':
        result = (a - b) & U32
        if dtype in ('u32',):
            scc = _scc_borrow_u32(a, b)
        elif dtype in ('i32',):
            sa = (a >> 31) & 1
            sb = (b >> 31) & 1
            sr = (result >> 31) & 1
            scc = (sa != sb) and (sr != sa)
        else:
            scc = _scc_nonzero(result)
    elif op == 'and':
        result = a & b
        scc = _scc_nonzero(result)
    elif op == 'or':
        result = a | b
        scc = _scc_nonzero(result)
    elif op == 'xor':
        result = a ^ b
        scc = _scc_nonzero(result)
    elif op == 'nand':
        result = (~(a & b)) & U32
        scc = _scc_nonzero(result)
    elif op == 'nor':
        result = (~(a | b)) & U32
        scc = _scc_nonzero(result)
    elif op == 'xnor':
        result = (~(a ^ b)) & U32
        scc = _scc_nonzero(result)
    elif op == 'andn2':
        result = a & (~b & U32)
        scc = _scc_nonzero(result)
    elif op == 'orn2':
        result = a | (~b & U32)
        scc = _scc_nonzero(result)
    elif op == 'shl':
        shift = b & 0x1F
        result = (a << shift) & U32
        scc = _scc_nonzero(result)
    elif op == 'shr':
        shift = b & 0x1F
        result = a >> shift
        scc = _scc_nonzero(result)
    elif op == 'ashr':
        shift = b & 0x1F
        signed_a = _to_i32(a)
        result = _to_u32(signed_a >> shift)
        scc = _scc_nonzero(result)
    elif op == 'mul':
        if dtype in ('i32', 'i16'):
            result = (_to_i32(a) * _to_i32(b)) & U32
        else:
            result = (a * b) & U32
        scc = None
    elif op == 'mulhi':
        if dtype in ('i32',):
            full = _to_i32(a) * _to_i32(b)
            result = (full >> 32) & U32
        else:
            full = a * b
            result = (full >> 32) & U32
        scc = None
    elif op == 'min':
        if dtype in ('i32',):
            result = _to_u32(min(_to_i32(a), _to_i32(b)))
        else:
            result = min(a, b)
        scc = result == a
    elif op == 'max':
        if dtype in ('i32',):
            result = _to_u32(max(_to_i32(a), _to_i32(b)))
        else:
            result = max(a, b)
        scc = result == a
    elif op == 'absdiff':
        if dtype in ('i32',):
            result = _to_u32(abs(_to_i32(a) - _to_i32(b)))
        else:
            result = abs(a - b) & U32
        scc = _scc_nonzero(result)
    elif op == 'bfe':
        if dtype in ('i32',):
            result = _bfe_i32(a, b)
        else:
            result = _bfe_u32(a, b)
        scc = _scc_nonzero(result)
    elif op == 'bfm':
        result = _bfm(a, b)
        scc = None
    elif op == 'pack_ll':
        result = (a & 0xFFFF) | ((b & 0xFFFF) << 16)
        scc = None
    elif op == 'pack_lh':
        result = (a & 0xFFFF) | (b & 0xFFFF_0000)
        scc = None
    elif op == 'pack_hh':
        result = ((a >> 16) & 0xFFFF) | (b & 0xFFFF_0000)
        scc = None
    elif op == 'pack_hl':
        result = ((a >> 16) & 0xFFFF) | ((b & 0xFFFF) << 16)
        scc = None
    elif op in ('lshl1_add', 'lshl2_add', 'lshl3_add', 'lshl4_add'):
        shift = int(op[4])  # lshl{N}_add
        full_sum = (a << shift) + b
        result = full_sum & U32
        scc = full_sum > U32
    else:
        result = 0
        scc = None

    result &= U32
    return GoldenResult(result, scc=scc)


# ---------------------------------------------------------------------------
#  Scalar mov / cselect
# ---------------------------------------------------------------------------


def _scalar_mov(src: int) -> GoldenResult:
    return GoldenResult(src & U32, scc=None)


def _scalar_cselect(src0: int, src1: int, scc_in: bool) -> GoldenResult:
    result = src0 if scc_in else src1
    return GoldenResult(result & U32, scc=None)


# ---------------------------------------------------------------------------
#  Vector unary operations (integer only for Phase 2)
# ---------------------------------------------------------------------------


def _vector_unary_int(op: str, dtype: str | None, src: int) -> GoldenResult | None:
    """Compute golden for an integer vector unary instruction."""
    s = src & U32

    if op == 'not':
        return GoldenResult(~s & U32)
    if op == 'bfrev':
        return GoldenResult(_brev(s))
    if op == 'bcnt':
        return GoldenResult(_popcount(s))
    if op == 'ffbl':
        return GoldenResult(_ff1(s))
    if op == 'ffbh_u32':
        return GoldenResult(_flbit_unsigned(s))
    if op == 'ffbh_i32':
        return GoldenResult(_flbit_signed(s))
    if op == 'cls_i32':
        signed_val = _to_i32(s)
        if signed_val == 0 or signed_val == -1:
            return GoldenResult(31)
        x = (~s if signed_val < 0 else s) & U32
        for i in range(30, -1, -1):
            if x & (1 << i):
                return GoldenResult(30 - i)
        return GoldenResult(31)
    return None


# ---------------------------------------------------------------------------
#  Vector binary operations (integer only for Phase 2)
# ---------------------------------------------------------------------------


def _vector_binop_int(
    op: str, dtype: str | None, src0: int, src1: int
) -> GoldenResult | None:
    """Compute golden for an integer vector binary instruction."""
    a = src0 & U32
    b = src1 & U32

    if op == 'add':
        return GoldenResult((a + b) & U32)
    if op == 'sub':
        return GoldenResult((a - b) & U32)
    if op == 'rsub':
        return GoldenResult((b - a) & U32)
    if op == 'and':
        return GoldenResult(a & b)
    if op == 'or':
        return GoldenResult(a | b)
    if op == 'xor':
        return GoldenResult(a ^ b)
    if op == 'xnor':
        return GoldenResult(~(a ^ b) & U32)
    if op == 'shl':
        return GoldenResult((b << (a & 0x1F)) & U32)
    if op == 'shr':
        return GoldenResult(b >> (a & 0x1F))
    if op == 'ashr':
        shift = a & 0x1F
        return GoldenResult(_to_u32(_to_i32(b) >> shift))
    if op == 'mul':
        if dtype == 'i24':
            sa = _to_i32(a << 8) >> 8  # sign-extend 24-bit
            sb = _to_i32(b << 8) >> 8
            return GoldenResult(_to_u32(sa * sb))
        if dtype == 'u24':
            return GoldenResult(((a & 0xFFFFFF) * (b & 0xFFFFFF)) & U32)
        return GoldenResult((a * b) & U32)
    if op == 'mulhi':
        if dtype == 'i24':
            sa = _to_i32(a << 8) >> 8
            sb = _to_i32(b << 8) >> 8
            return GoldenResult(_to_u32((sa * sb) >> 32))
        if dtype == 'u24':
            return GoldenResult(((a & 0xFFFFFF) * (b & 0xFFFFFF)) >> 32)
        if dtype == 'i32':
            return GoldenResult(_to_u32((_to_i32(a) * _to_i32(b)) >> 32))
        return GoldenResult((a * b) >> 32)
    if op == 'min':
        if dtype in ('i32', 'i16'):
            return GoldenResult(_to_u32(min(_to_i32(a), _to_i32(b))))
        return GoldenResult(min(a, b))
    if op == 'max':
        if dtype in ('i32', 'i16'):
            return GoldenResult(_to_u32(max(_to_i32(a), _to_i32(b))))
        return GoldenResult(max(a, b))
    if op == 'bfm':
        return GoldenResult(_bfm(a, b))
    return None


# ---------------------------------------------------------------------------
#  Vector mov
# ---------------------------------------------------------------------------


def _vector_mov(src: int) -> GoldenResult:
    return GoldenResult(src & U32)


# ---------------------------------------------------------------------------
#  Scalar comparison operations
# ---------------------------------------------------------------------------


def _f32_from_bits(bits: int) -> float:
    return struct.unpack('<f', struct.pack('<I', bits & U32))[0]


def _int_cmp(op: str, a: int, b: int, signed: bool) -> bool | None:
    """Evaluate an integer comparison. Returns None if op is unknown."""
    if signed:
        a, b = _to_i32(a), _to_i32(b)
    cmp_map = {
        'eq': a == b,
        'ne': a != b,
        'lg': a != b,
        'lt': a < b,
        'le': a <= b,
        'gt': a > b,
        'ge': a >= b,
        'f': False,
        't': True,
    }
    return cmp_map.get(op)


def _fp_cmp(op: str, fa: float, fb: float) -> bool | None:
    """Evaluate an IEEE 754 FP comparison with NaN semantics."""
    unord = bool(np.isnan(fa) or np.isnan(fb))
    cmp_map = {
        'f': False,
        't': True,
        'eq': (not unord) and fa == fb,
        'ne': unord or fa != fb,
        'lg': (not unord) and fa != fb,
        'lt': (not unord) and fa < fb,
        'le': (not unord) and fa <= fb,
        'gt': (not unord) and fa > fb,
        'ge': (not unord) and fa >= fb,
        'o': not unord,
        'u': unord,
        'nge': unord or not (fa >= fb),
        'ngt': unord or not (fa > fb),
        'nle': unord or not (fa <= fb),
        'nlt': unord or not (fa < fb),
        'nlg': unord or fa == fb,
        'neq': unord or fa != fb,
    }
    return cmp_map.get(op)


def _scalar_cmp(op: str, dtype: str | None, src0: int, src1: int) -> GoldenResult:
    """Compute golden for a scalar comparison (SOPC). Result is SCC only."""
    a = src0 & U32
    b = src1 & U32

    if dtype in ('f32',):
        fa, fb = _f32_from_bits(a), _f32_from_bits(b)
        result = _fp_cmp(op, fa, fb)
        if result is None:
            return GoldenResult(0, scc=False)
        return GoldenResult(0, scc=result)

    if dtype in ('i32',):
        result = _int_cmp(op, a, b, signed=True)
        if result is None:
            return GoldenResult(0, scc=False)
        return GoldenResult(0, scc=result)

    if dtype in ('u32',):
        result = _int_cmp(op, a, b, signed=False)
        if result is None:
            return GoldenResult(0, scc=False)
        return GoldenResult(0, scc=result)

    return GoldenResult(0, scc=False)


def _scalar_bitcmp(op: str, src0: int, src1: int) -> GoldenResult:
    """Compute golden for S_BITCMP0_B32 / S_BITCMP1_B32."""
    bit = src1 & 0x1F
    bit_val = (src0 >> bit) & 1
    if op == 'bitcmp0':
        return GoldenResult(0, scc=(bit_val == 0))
    return GoldenResult(0, scc=(bit_val == 1))


# ---------------------------------------------------------------------------
#  FP helpers
# ---------------------------------------------------------------------------


def _f32_to_bits(f: float) -> int:
    return struct.unpack('<I', struct.pack('<f', f))[0]


def _apply_src_modifiers_f32(bits: int, src_idx: int, neg: int, abs_: int) -> float:
    """Apply VOP3 abs then neg to an f32 source, return as Python float."""
    val = _f32_from_bits(bits)
    if abs_ & (1 << src_idx):
        val = abs(val)
    if neg & (1 << src_idx):
        val = -val
    return np.float32(val)


def _apply_dst_modifiers_f32(val: float, omod: int, clamp: int) -> float:
    """Apply VOP3 omod then clamp to an f32 result."""
    v = np.float32(val)
    if omod == 1:
        v = np.float32(v * np.float32(2.0))
    elif omod == 2:
        v = np.float32(v * np.float32(4.0))
    elif omod == 3:
        v = np.float32(v * np.float32(0.5))
    if clamp:
        v = np.float32(np.clip(v, np.float32(0.0), np.float32(1.0)))
    return v


# ---------------------------------------------------------------------------
#  Vector unary FP operations
# ---------------------------------------------------------------------------


def _vector_unary_fp(
    op: str, src_bits: int, neg: int = 0, abs_: int = 0, omod: int = 0, clamp: int = 0
) -> GoldenResult | None:
    """Compute golden for an FP vector unary instruction with modifiers."""
    a = _apply_src_modifiers_f32(src_bits, 0, neg, abs_)

    if op == 'floor':
        r = np.float32(np.floor(a))
    elif op == 'ceil':
        r = np.float32(np.ceil(a))
    elif op == 'trunc':
        r = np.float32(np.trunc(a))
    elif op == 'rndne':
        r = np.float32(np.rint(a))
    elif op == 'fract':
        r = np.float32(a - np.floor(a))
    elif op == 'rcp':
        if a == 0.0:
            r = np.float32(np.copysign(np.inf, a))
        elif np.isinf(a):
            r = np.float32(np.copysign(0.0, a))
        elif np.isnan(a):
            r = np.float32(np.nan)
        else:
            r = np.float32(np.float32(1.0) / a)
    elif op == 'rsq':
        if np.isnan(a):
            r = np.float32(np.nan)
        elif a == 0.0:
            r = np.float32(np.copysign(np.inf, a))
        elif a < 0.0:
            r = np.float32(np.nan)
        elif np.isinf(a):
            r = np.float32(0.0)
        else:
            r = np.float32(np.float32(1.0) / np.float32(np.sqrt(a)))
    elif op == 'sqrt':
        if np.isnan(a):
            r = np.float32(np.nan)
        elif a < 0.0:
            r = np.float32(np.nan)
        elif a == 0.0:
            r = np.float32(np.copysign(0.0, a))
        elif np.isinf(a):
            r = np.float32(np.inf)
        else:
            r = np.float32(np.sqrt(a))
    else:
        return None

    r = _apply_dst_modifiers_f32(r, omod, clamp)
    return GoldenResult(_f32_to_bits(float(r)))


# ---------------------------------------------------------------------------
#  Vector binary FP operations
# ---------------------------------------------------------------------------


def _vector_binop_fp(
    op: str,
    src0_bits: int,
    src1_bits: int,
    neg: int = 0,
    abs_: int = 0,
    omod: int = 0,
    clamp: int = 0,
) -> GoldenResult | None:
    """Compute golden for an FP vector binary instruction with modifiers."""
    a = _apply_src_modifiers_f32(src0_bits, 0, neg, abs_)
    b = _apply_src_modifiers_f32(src1_bits, 1, neg, abs_)

    if op == 'add':
        r = np.float32(a + b)
    elif op == 'sub':
        r = np.float32(a - b)
    elif op == 'rsub':
        r = np.float32(b - a)
    elif op == 'mul':
        r = np.float32(a * b)
    elif op == 'max':
        if np.isnan(a):
            r = b
        elif np.isnan(b):
            r = a
        else:
            r = np.float32(max(float(a), float(b)))
    elif op == 'min':
        if np.isnan(a):
            r = b
        elif np.isnan(b):
            r = a
        else:
            r = np.float32(min(float(a), float(b)))
    else:
        return None

    r = _apply_dst_modifiers_f32(r, omod, clamp)
    return GoldenResult(_f32_to_bits(float(r)))


# ---------------------------------------------------------------------------
#  Vector ternary operations (integer only for Phase 3)
# ---------------------------------------------------------------------------


def _vector_ternary_int(
    op: str, dtype: str | None, src0: int, src1: int, src2: int
) -> GoldenResult | None:
    """Compute golden for an integer vector ternary instruction."""
    a, b, c = src0 & U32, src1 & U32, src2 & U32

    if op == 'min3':
        if dtype in ('i32',):
            result = _to_u32(min(_to_i32(a), _to_i32(b), _to_i32(c)))
        else:
            result = min(a, b, c)
        return GoldenResult(result & U32)
    if op == 'max3':
        if dtype in ('i32',):
            result = _to_u32(max(_to_i32(a), _to_i32(b), _to_i32(c)))
        else:
            result = max(a, b, c)
        return GoldenResult(result & U32)
    if op == 'med3':
        if dtype in ('i32',):
            vals = sorted([_to_i32(a), _to_i32(b), _to_i32(c)])
            result = _to_u32(vals[1])
        else:
            vals = sorted([a, b, c])
            result = vals[1]
        return GoldenResult(result & U32)
    if op == 'bfe_u':
        offset = b & 0x1F
        width = c & 0x1F
        if width == 0:
            return GoldenResult(0)
        if offset + width > 32:
            width = 32 - offset
        return GoldenResult((a >> offset) & ((1 << width) - 1))
    if op == 'bfe_i':
        offset = b & 0x1F
        width = c & 0x1F
        if width == 0:
            return GoldenResult(0)
        if offset + width > 32:
            width = 32 - offset
        extracted = (a >> offset) & ((1 << width) - 1)
        if extracted & (1 << (width - 1)):
            extracted |= U32 & ~((1 << width) - 1)
        return GoldenResult(extracted & U32)
    if op == 'bfi':
        return GoldenResult(((a & b) | ((~a & U32) & c)) & U32)
    if op == 'alignbit':
        shift = c & 0x1F
        combined = (a << 32) | b
        return GoldenResult((combined >> shift) & U32)
    if op == 'alignbyte':
        shift = (c & 0x3) * 8
        combined = (a << 32) | b
        return GoldenResult((combined >> shift) & U32)
    if op == 'add3':
        return GoldenResult((a + b + c) & U32)
    if op == 'xor3':
        return GoldenResult(a ^ b ^ c)
    if op == 'and_or':
        return GoldenResult(((a & b) | c) & U32)
    if op == 'or3':
        return GoldenResult((a | b | c) & U32)
    if op == 'lshl_or':
        return GoldenResult(((a << (b & 0x1F)) | c) & U32)
    if op == 'lshl_add':
        return GoldenResult(((a << (b & 0x1F)) + c) & U32)
    if op == 'add_lshl':
        return GoldenResult(((a + b) << (c & 0x1F)) & U32)
    if op == 'xad':
        return GoldenResult(((a ^ b) + c) & U32)
    if op == 'maxmin':
        if dtype in ('i32',):
            result = _to_u32(min(max(_to_i32(a), _to_i32(b)), _to_i32(c)))
        else:
            result = min(max(a, b), c)
        return GoldenResult(result & U32)
    if op == 'minmax':
        if dtype in ('i32',):
            result = _to_u32(max(min(_to_i32(a), _to_i32(b)), _to_i32(c)))
        else:
            result = max(min(a, b), c)
        return GoldenResult(result & U32)
    if op == 'add_max':
        if dtype in ('i32',):
            result = _to_u32(max(_to_i32(a) + _to_i32(b), _to_i32(c)))
            return GoldenResult(result & U32)
        return GoldenResult(max((a + b) & U32, c))
    if op == 'add_min':
        if dtype in ('i32',):
            result = _to_u32(min(_to_i32(a) + _to_i32(b), _to_i32(c)))
            return GoldenResult(result & U32)
        return GoldenResult(min((a + b) & U32, c))
    if op == 'lerp_u8':
        result = 0
        for byte_idx in range(4):
            ab = (a >> (byte_idx * 8)) & 0xFF
            bb = (b >> (byte_idx * 8)) & 0xFF
            cb = (c >> (byte_idx * 8)) & 0xFF
            numer = (bb - ab) * cb + 128
            val = ab + (numer // 256 if numer >= 0 else -(-numer // 256))
            result |= (val & 0xFF) << (byte_idx * 8)
        return GoldenResult(result)
    if op == 'sad_u8':
        sad = 0
        for byte_idx in range(4):
            sad += abs(((a >> (byte_idx * 8)) & 0xFF) - ((b >> (byte_idx * 8)) & 0xFF))
        return GoldenResult((sad + c) & U32)
    if op == 'sad_u16':
        sad = abs((a & 0xFFFF) - (b & 0xFFFF))
        sad += abs(((a >> 16) & 0xFFFF) - ((b >> 16) & 0xFFFF))
        return GoldenResult((sad + c) & U32)
    if op == 'sad_u32':
        sad = abs(int(a) - int(b))
        return GoldenResult((sad + c) & U32)
    if op == 'perm':
        src_64 = (a << 32) | b
        result = 0
        for i in range(4):
            sel = (c >> (i * 8)) & 0xFF
            if sel < 8:
                byte_val = (src_64 >> (sel * 8)) & 0xFF
            elif sel == 0xC:
                byte_val = 0
            elif sel == 0xD:
                byte_val = 0xFF
            else:
                byte_val = 0
            result |= byte_val << (i * 8)
        return GoldenResult(result)
    return None


# ---------------------------------------------------------------------------
#  Public dispatch
# ---------------------------------------------------------------------------


def compute_scalar_alu_golden(
    semantic_class: str,
    operation: str | None,
    dtype: str | None,
    inputs: list[int],
    scc_in: bool = False,
) -> GoldenResult | None:
    """Compute the golden reference for a scalar ALU instruction.

    Args:
        semantic_class: e.g. 'scalar_unary', 'scalar_binop'
        operation: e.g. 'add', 'not', 'bfe'
        dtype: e.g. 'u32', 'i32', 'b32'
        inputs: raw uint32 input values [src0] or [src0, src1]
        scc_in: current SCC value (for cselect, addc, subb)

    Returns:
        GoldenResult with output_bits and optional scc, or None if unsupported.
    """
    if semantic_class == 'scalar_unary' and operation and len(inputs) >= 1:
        return _scalar_unary(operation, dtype, inputs[0])
    if semantic_class == 'scalar_binop' and operation and len(inputs) >= 2:
        return _scalar_binop(operation, dtype, inputs[0], inputs[1])
    if semantic_class == 'scalar_mov' and len(inputs) >= 1:
        return _scalar_mov(inputs[0])
    if semantic_class == 'scalar_cselect' and len(inputs) >= 2:
        return _scalar_cselect(inputs[0], inputs[1], scc_in)
    return None


def compute_vector_alu_golden(
    semantic_class: str,
    operation: str | None,
    dtype: str | None,
    inputs: list[int],
) -> GoldenResult | None:
    """Compute golden for a vector ALU instruction (integer ops only).

    Returns GoldenResult or None if the operation is unsupported (e.g. FP).
    """
    if semantic_class == 'vector_mov' and len(inputs) >= 1:
        return _vector_mov(inputs[0])
    if semantic_class == 'vector_unary' and operation and len(inputs) >= 1:
        return _vector_unary_int(operation, dtype, inputs[0])
    if semantic_class == 'vector_binop' and operation and len(inputs) >= 2:
        return _vector_binop_int(operation, dtype, inputs[0], inputs[1])
    return None


def compute_scalar_cmp_golden(
    semantic_class: str,
    operation: str | None,
    dtype: str | None,
    inputs: list[int],
) -> GoldenResult | None:
    """Compute golden for a scalar comparison instruction (SOPC).

    Returns GoldenResult with scc set, output_bits=0.
    """
    if semantic_class == 'scalar_cmp' and operation and len(inputs) >= 2:
        return _scalar_cmp(operation, dtype, inputs[0], inputs[1])
    if semantic_class == 'scalar_bitcmp' and operation and len(inputs) >= 2:
        return _scalar_bitcmp(operation, inputs[0], inputs[1])
    return None


def compute_vector_cmp_golden(
    semantic_class: str,
    operation: str | None,
    dtype: str | None,
    inputs: list[int],
) -> GoldenResult | None:
    """Compute golden for a vector comparison (VOPC/VOP3).

    Returns GoldenResult with scc=True/False representing the VCC bit for lane 0.
    """
    if semantic_class not in ('vector_cmp', 'vector_cmpx'):
        return None
    if not operation or len(inputs) < 2:
        return None

    a, b = inputs[0] & U32, inputs[1] & U32

    if dtype in ('f32',):
        fa, fb = _f32_from_bits(a), _f32_from_bits(b)
        result = _fp_cmp(operation, fa, fb)
    elif dtype in ('i32', 'i16'):
        result = _int_cmp(operation, a, b, signed=True)
    elif dtype in ('u32', 'u16'):
        result = _int_cmp(operation, a, b, signed=False)
    else:
        return None

    if result is None:
        return None
    return GoldenResult(0, scc=result)


def compute_vector_ternary_golden(
    semantic_class: str,
    operation: str | None,
    dtype: str | None,
    inputs: list[int],
) -> GoldenResult | None:
    """Compute golden for a vector ternary ALU instruction (integer ops only)."""
    if semantic_class != 'vector_ternary' or not operation or len(inputs) < 3:
        return None
    return _vector_ternary_int(operation, dtype, inputs[0], inputs[1], inputs[2])


# ---------------------------------------------------------------------------
#  Bitop3 (8-bit truth table applied bitwise to 3 inputs)
# ---------------------------------------------------------------------------


def _bitop3(truth_table: int, src0: int, src1: int, src2: int, nbits: int = 32) -> int:
    result = 0
    for i in range(nbits):
        idx = (((src0 >> i) & 1) << 2) | (((src1 >> i) & 1) << 1) | ((src2 >> i) & 1)
        result |= ((truth_table >> idx) & 1) << i
    return result & ((1 << nbits) - 1)


def compute_bitop3_golden(truth_table: int, inputs: list[int]) -> GoldenResult | None:
    if len(inputs) < 3:
        return None
    result = _bitop3(truth_table, inputs[0] & U32, inputs[1] & U32, inputs[2] & U32)
    return GoldenResult(result)


# ---------------------------------------------------------------------------
#  Integer dot products
# ---------------------------------------------------------------------------


def _sext_n(val: int, nbits: int) -> int:
    mask = (1 << nbits) - 1
    val &= mask
    if val & (1 << (nbits - 1)):
        val -= 1 << nbits
    return val


def _dot4_i32_i8(src0: int, src1: int, acc: int) -> int:
    s = _to_i32(acc)
    for i in range(4):
        a = _sext_n((src0 >> (i * 8)) & 0xFF, 8)
        b = _sext_n((src1 >> (i * 8)) & 0xFF, 8)
        s += a * b
    return _to_u32(s)


def _dot4_u32_u8(src0: int, src1: int, acc: int) -> int:
    s = acc & U32
    for i in range(4):
        a = (src0 >> (i * 8)) & 0xFF
        b = (src1 >> (i * 8)) & 0xFF
        s += a * b
    return s & U32


def _dot8_i32_i4(src0: int, src1: int, acc: int) -> int:
    s = _to_i32(acc)
    for i in range(8):
        a = _sext_n((src0 >> (i * 4)) & 0xF, 4)
        b = _sext_n((src1 >> (i * 4)) & 0xF, 4)
        s += a * b
    return _to_u32(s)


def _dot8_u32_u4(src0: int, src1: int, acc: int) -> int:
    s = acc & U32
    for i in range(8):
        a = (src0 >> (i * 4)) & 0xF
        b = (src1 >> (i * 4)) & 0xF
        s += a * b
    return s & U32


def _dot2_i32_i16(src0: int, src1: int, acc: int) -> int:
    s = _to_i32(acc)
    for i in range(2):
        a = _sext_n((src0 >> (i * 16)) & 0xFFFF, 16)
        b = _sext_n((src1 >> (i * 16)) & 0xFFFF, 16)
        s += a * b
    return _to_u32(s)


def _dot2_u32_u16(src0: int, src1: int, acc: int) -> int:
    s = acc & U32
    for i in range(2):
        a = (src0 >> (i * 16)) & 0xFFFF
        b = (src1 >> (i * 16)) & 0xFFFF
        s += a * b
    return s & U32


_DOT_DISPATCH = {
    'dot4_i32_i8': _dot4_i32_i8,
    'dot4_u32_u8': _dot4_u32_u8,
    'dot8_i32_i4': _dot8_i32_i4,
    'dot8_u32_u4': _dot8_u32_u4,
    'dot2_i32_i16': _dot2_i32_i16,
    'dot2_u32_u16': _dot2_u32_u16,
}


def compute_dot_product_golden(
    semantic_class: str,
    inputs: list[int],
) -> GoldenResult | None:
    fn = _DOT_DISPATCH.get(semantic_class)
    if fn is None or len(inputs) < 3:
        return None
    result = fn(inputs[0] & U32, inputs[1] & U32, inputs[2] & U32)
    return GoldenResult(result & U32)


# ---------------------------------------------------------------------------
#  Packed 16-bit integer operations
# ---------------------------------------------------------------------------


def _pk_binop_int(op: str, dtype: str, src0: int, src1: int) -> int:
    lo0, hi0 = src0 & 0xFFFF, (src0 >> 16) & 0xFFFF
    lo1, hi1 = src1 & 0xFFFF, (src1 >> 16) & 0xFFFF
    signed = dtype == 'i16'

    def apply_op(a: int, b: int) -> int:
        if signed:
            a, b = _sext_n(a, 16), _sext_n(b, 16)

        if op == 'add':
            r = a + b
        elif op == 'sub':
            r = a - b
        elif op == 'mul':
            r = a * b
        elif op == 'min':
            r = min(a, b)
        elif op == 'max':
            r = max(a, b)
        elif op == 'shl':
            r = b << (a & 0xF)
        elif op == 'shr':
            r = (b & 0xFFFF) >> (a & 0xF)
        elif op == 'ashr':
            r = _sext_n(b, 16) >> (a & 0xF)
        else:
            return 0
        return r & 0xFFFF

    rlo = apply_op(lo0, lo1)
    rhi = apply_op(hi0, hi1)
    return (rhi << 16) | rlo


def compute_packed_int_golden(
    operation: str,
    dtype: str,
    inputs: list[int],
) -> GoldenResult | None:
    if len(inputs) < 2:
        return None
    result = _pk_binop_int(operation, dtype, inputs[0] & U32, inputs[1] & U32)
    return GoldenResult(result & U32)


# ---------------------------------------------------------------------------
#  64-bit multiply-accumulate (mad_u64_u32, mad_i64_i32)
# ---------------------------------------------------------------------------


def compute_mad_64_32_golden(
    dtype: str,
    inputs: list[int],
) -> tuple[int, int] | None:
    if len(inputs) < 4:
        return None
    src0, src1 = inputs[0] & U32, inputs[1] & U32
    src2 = ((inputs[3] & U32) << 32) | (inputs[2] & U32)

    if dtype == 'i64':
        result = _to_i32(src0) * _to_i32(src1) + _to_i64(src2)
        result &= U64
    else:
        result = (src0 * src1 + src2) & U64

    return result & U32, (result >> 32) & U32


# ---------------------------------------------------------------------------
#  Packed FP16 operations
# ---------------------------------------------------------------------------


def _f16_bits_to_f32(bits: int) -> np.float32:
    f16_val = np.frombuffer(
        np.array([bits & 0xFFFF], dtype=np.uint16).tobytes(),
        dtype=np.float16,
    )[0]
    return np.float32(f16_val)


def _f32_to_f16_bits(val: float) -> int:
    f16_val = np.float16(val)
    return int(
        np.frombuffer(
            np.array([f16_val], dtype=np.float16).tobytes(),
            dtype=np.uint16,
        )[0]
    )


def _pk_binop_f16(op: str, src0: int, src1: int) -> int:
    lo0, hi0 = src0 & 0xFFFF, (src0 >> 16) & 0xFFFF
    lo1, hi1 = src1 & 0xFFFF, (src1 >> 16) & 0xFFFF

    def apply_op(a_bits: int, b_bits: int) -> int:
        a = _f16_bits_to_f32(a_bits)
        b = _f16_bits_to_f32(b_bits)
        if op == 'add':
            r = np.float32(a + b)
        elif op == 'mul':
            r = np.float32(a * b)
        elif op == 'min':
            r = np.float32(np.fmin(a, b))
        elif op == 'max':
            r = np.float32(np.fmax(a, b))
        else:
            return 0
        return _f32_to_f16_bits(float(r))

    rlo = apply_op(lo0, lo1)
    rhi = apply_op(hi0, hi1)
    return (rhi << 16) | rlo


def _pk_ternary_f16(op: str, src0: int, src1: int, src2: int) -> int:
    lo0, hi0 = src0 & 0xFFFF, (src0 >> 16) & 0xFFFF
    lo1, hi1 = src1 & 0xFFFF, (src1 >> 16) & 0xFFFF
    lo2, hi2 = src2 & 0xFFFF, (src2 >> 16) & 0xFFFF

    def apply_op(a_bits: int, b_bits: int, c_bits: int) -> int:
        a = _f16_bits_to_f32(a_bits)
        b = _f16_bits_to_f32(b_bits)
        c = _f16_bits_to_f32(c_bits)
        if op == 'fma':
            r = np.float32(float(a) * float(b) + float(c))
        elif op == 'min3':
            r = np.float32(np.fmin(np.fmin(a, b), c))
        elif op == 'max3':
            r = np.float32(np.fmax(np.fmax(a, b), c))
        else:
            return 0
        return _f32_to_f16_bits(float(r))

    rlo = apply_op(lo0, lo1, lo2)
    rhi = apply_op(hi0, hi1, hi2)
    return (rhi << 16) | rlo


def compute_packed_fp16_golden(
    semantic_class: str,
    operation: str,
    inputs: list[int],
) -> GoldenResult | None:
    if semantic_class == 'pk_binop' and len(inputs) >= 2:
        result = _pk_binop_f16(operation, inputs[0] & U32, inputs[1] & U32)
        return GoldenResult(result & U32)
    if semantic_class == 'pk_ternary' and len(inputs) >= 3:
        result = _pk_ternary_f16(
            operation,
            inputs[0] & U32,
            inputs[1] & U32,
            inputs[2] & U32,
        )
        return GoldenResult(result & U32)
    return None


def compute_vector_modifier_golden(
    semantic_class: str,
    operation: str | None,
    dtype: str | None,
    inputs: list[int],
    neg: int = 0,
    abs_: int = 0,
    omod: int = 0,
    clamp: int = 0,
) -> GoldenResult | None:
    """Compute golden for a VOP3 FP instruction with modifiers."""
    if dtype != 'f32' or not operation:
        return None
    if semantic_class == 'vector_unary' and len(inputs) >= 1:
        return _vector_unary_fp(operation, inputs[0], neg, abs_, omod, clamp)
    if semantic_class == 'vector_binop' and len(inputs) >= 2:
        return _vector_binop_fp(operation, inputs[0], inputs[1], neg, abs_, omod, clamp)
    return None


# ---------------------------------------------------------------------------
# DS (LDS) atomic operations
# ---------------------------------------------------------------------------


def compute_ds_atomic_golden(
    operation: str, initial: int, operand: int
) -> tuple[int, int]:
    """Compute golden (old_value, new_lds_value) for a DS atomic RTN op."""
    old = initial & U32
    op = operand & U32
    if operation == 'add':
        new = (old + op) & U32
    elif operation == 'sub':
        new = (old - op) & U32
    elif operation == 'rsub':
        new = (op - old) & U32
    elif operation == 'swap':
        new = op
    elif operation == 'smin':
        new = min(_to_i32(old), _to_i32(op)) & U32
    elif operation == 'smax':
        new = max(_to_i32(old), _to_i32(op)) & U32
    elif operation == 'umin':
        new = min(old, op)
    elif operation == 'umax':
        new = max(old, op)
    elif operation == 'and':
        new = old & op
    elif operation == 'or':
        new = old | op
    elif operation == 'xor':
        new = old ^ op
    elif operation == 'inc':
        new = 0 if old >= op else (old + 1) & U32
    elif operation == 'dec':
        new = op if (old == 0 or old > op) else (old - 1) & U32
    else:
        raise ValueError(f"Unknown DS atomic op: {operation}")
    return (old, new)
