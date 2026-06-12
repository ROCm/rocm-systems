# Copyright (c) 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Deterministic input vector generation for CTS test cases.

Generates edge-case + pseudo-random inputs per data type.  All values are
returned as unsigned 32-bit or 64-bit integers (the raw bit representation).
"""

from __future__ import annotations

import struct

U32_MAX = 0xFFFF_FFFF
U64_MAX = 0xFFFF_FFFF_FFFF_FFFF
I32_MAX = 0x7FFF_FFFF
I32_MIN = 0x8000_0000
I64_MAX = 0x7FFF_FFFF_FFFF_FFFF
I64_MIN = 0x8000_0000_0000_0000


def _simple_hash(seed: int, idx: int) -> int:
    """Fast deterministic hash — NOT cryptographic, just reproducible."""
    x = (seed * 2654435761 + idx * 2246822519) & U32_MAX
    x ^= x >> 16
    x = (x * 0x45D9F3B) & U32_MAX
    x ^= x >> 16
    return x


def int_u32_inputs(seed: int = 0) -> list[int]:
    """32 edge-case + random unsigned 32-bit values."""
    vals = [
        0,
        1,
        2,
        0xF,
        0xFF,
        0xFFFF,
        I32_MAX,
        I32_MIN,
        U32_MAX,
        U32_MAX - 1,
        0x5555_5555,
        0xAAAA_AAAA,
        0x0000_0001,
        0x0000_0002,
        0x0000_0004,
        0x0000_0008,
        0x0000_0010,
        0x0000_0100,
        0x0001_0000,
        0x0100_0000,
        0x8000_0000,
        0x4000_0000,
        0x2000_0000,
        0x1000_0000,
    ]
    while len(vals) < 32:
        vals.append(_simple_hash(seed, len(vals)))
    return vals[:32]


def int_i32_inputs(seed: int = 0) -> list[int]:
    """32 edge-case + random signed 32-bit values (as unsigned bits)."""
    vals = [
        0,
        1,
        U32_MAX,  # 0, 1, -1
        I32_MAX,
        I32_MIN,  # INT_MAX, INT_MIN
        I32_MIN + 1,  # INT_MIN + 1
        2,
        0xFFFF_FFFE,  # 2, -2
        0x7F,
        0xFFFF_FF80,  # 127, -128
        0x7FFF,
        0xFFFF_8000,  # 32767, -32768
        0x5555_5555,
        0xAAAA_AAAA,
        0x0000_0010,
        0xFFFF_FFF0,
    ]
    while len(vals) < 32:
        vals.append(_simple_hash(seed, len(vals)))
    return vals[:32]


def int_u64_inputs(seed: int = 0) -> list[tuple[int, int]]:
    """32 edge-case + random 64-bit values as (lo32, hi32) pairs."""

    def split(v: int) -> tuple[int, int]:
        return (v & U32_MAX, (v >> 32) & U32_MAX)

    vals64 = [
        0,
        1,
        2,
        U64_MAX,
        U64_MAX - 1,
        I64_MAX,
        I64_MIN,
        I64_MIN + 1,
        0xFFFF_FFFF,
        0x1_0000_0000,
        0x5555_5555_5555_5555,
        0xAAAA_AAAA_AAAA_AAAA,
        0x0000_0000_8000_0000,
        0x8000_0000_0000_0000,
    ]
    while len(vals64) < 32:
        lo = _simple_hash(seed, len(vals64) * 2)
        hi = _simple_hash(seed, len(vals64) * 2 + 1)
        vals64.append((hi << 32) | lo)
    return [split(v) for v in vals64[:32]]


def float_f32_bits(seed: int = 0) -> list[int]:
    """32 edge-case + random float32 values as raw uint32 bit patterns."""

    def f2b(f: float) -> int:
        return struct.unpack('<I', struct.pack('<f', f))[0]

    vals = [
        f2b(0.0),  # +0
        f2b(-0.0),  # -0
        0x7F80_0000,  # +Inf
        0xFF80_0000,  # -Inf
        0x7FC0_0000,  # qNaN
        0x7F80_0001,  # sNaN
        f2b(1.0),
        f2b(-1.0),
        f2b(0.5),
        f2b(2.0),
        0x0080_0000,  # smallest positive normal
        0x8080_0000,  # smallest negative normal
        0x7F7F_FFFF,  # largest finite positive
        0xFF7F_FFFF,  # largest finite negative
        0x0000_0001,  # smallest positive denormal
        0x8000_0001,  # smallest negative denormal
        0x007F_FFFF,  # largest positive denormal
        0x807F_FFFF,  # largest negative denormal
        f2b(3.14159265),
        f2b(-2.71828182),
    ]
    while len(vals) < 32:
        raw = _simple_hash(seed, len(vals))
        # Make it a finite normal: clear exponent all-ones, set exponent nonzero
        exp = (raw >> 23) & 0xFF
        if exp == 0 or exp == 0xFF:
            raw = (raw & 0x807F_FFFF) | (0x40 << 23)
        vals.append(raw)
    return vals[:32]


def packed_i16_inputs(seed: int = 0) -> list[int]:
    """32 packed 16-bit input values — each is (hi16 << 16) | lo16."""
    vals = [
        0x0000_0000,  # (0, 0)
        0x0001_0001,  # (1, 1)
        0xFFFF_FFFF,  # (-1, -1) signed or (65535, 65535) unsigned
        0x7FFF_7FFF,  # (32767, 32767)
        0x8000_8000,  # (-32768, -32768)
        0x0001_FFFF,  # (1, -1)
        0xFFFF_0001,  # (-1, 1)
        0x7FFF_8000,  # (32767, -32768)
        0x8000_7FFF,  # (-32768, 32767)
        0x00FF_00FF,  # (255, 255)
        0xFF00_FF00,  # (-256, -256)
        0x5555_AAAA,  # mixed pattern
        0xAAAA_5555,  # inverted
        0x0010_0010,  # (16, 16)
        0x0100_0100,  # (256, 256)
        0x1234_5678,  # arbitrary
    ]
    while len(vals) < 32:
        vals.append(_simple_hash(seed + 99, len(vals)))
    return vals[:32]


def dot_product_inputs(seed: int = 0) -> list[int]:
    """30 input values (consumed in groups of 3) for dot product tests."""
    vals = [
        0x01020304,
        0x05060708,
        0x00000000,
        0x01010101,
        0x01010101,
        0x00000001,
        0x7F7F7F7F,
        0x7F7F7F7F,
        0x00000000,
        0x80808080,
        0x01010101,
        0x00000064,
        0xFFFFFFFF,
        0xFFFFFFFF,
        0x00000000,
        0x00FF00FF,
        0x00010001,
        0x0000000A,
        0x12345678,
        0x9ABCDEF0,
        0x00001000,
        0x55555555,
        0xAAAAAAAA,
        0xFFFFFFFF,
        0x00010002,
        0x00030004,
        0x00000005,
        0x80008000,
        0x7FFF7FFF,
        0x00000000,
    ]
    return vals[:30]


def packed_f16_inputs(seed: int = 0) -> list[int]:
    """32 packed f16 input values — each is (hi_f16_bits << 16) | lo_f16_bits."""
    import numpy as np

    def f16b(f: float) -> int:
        return int(
            np.frombuffer(
                np.array([np.float16(f)], dtype=np.float16).tobytes(),
                dtype=np.uint16,
            )[0]
        )

    def pack(lo_f: float, hi_f: float) -> int:
        return (f16b(hi_f) << 16) | f16b(lo_f)

    vals = [
        pack(0.0, 0.0),
        pack(1.0, 1.0),
        pack(-1.0, -1.0),
        pack(0.5, 0.5),
        pack(2.0, 2.0),
        pack(1.0, -1.0),
        pack(-1.0, 1.0),
        pack(0.5, 2.0),
        pack(3.0, 0.25),
        pack(10.0, -10.0),
        pack(100.0, 0.01),
        pack(-0.0, 0.0),  # signed zero
        0x0400_0400,  # smallest positive normal (2^-14)
        0x7BFF_7BFF,  # largest finite (65504.0)
        0x0001_0001,  # smallest positive denormal
        pack(1.5, -0.75),
        pack(0.125, 8.0),
        pack(-0.5, 0.5),
        pack(4.0, -4.0),
        pack(0.333251953125, 3.0),
        pack(1.0, 0.0),
        pack(0.0, 1.0),
        pack(256.0, -256.0),
        pack(0.0625, 16.0),
    ]
    while len(vals) < 32:
        raw = _simple_hash(seed + 200, len(vals))
        exp_lo = (raw >> 10) & 0x1F
        exp_hi = (raw >> 26) & 0x1F
        if exp_lo == 0x1F:
            raw = (raw & ~(0x1F << 10)) | (0x0E << 10)
        if exp_hi == 0x1F:
            raw = (raw & ~(0x1F << 26)) | (0x0E << 26)
        vals.append(raw & 0xFFFF_FFFF)
    return vals[:32]


def packed_f16_ternary_inputs(seed: int = 0) -> list[int]:
    """30 packed f16 input values (consumed in groups of 3) for ternary tests."""
    import numpy as np

    def f16b(f: float) -> int:
        return int(
            np.frombuffer(
                np.array([np.float16(f)], dtype=np.float16).tobytes(),
                dtype=np.uint16,
            )[0]
        )

    def pack(lo_f: float, hi_f: float) -> int:
        return (f16b(hi_f) << 16) | f16b(lo_f)

    vals = [
        pack(1.0, 1.0),
        pack(2.0, 2.0),
        pack(0.0, 0.0),
        pack(0.5, -0.5),
        pack(3.0, 3.0),
        pack(1.0, 1.0),
        pack(0.0, 0.0),
        pack(0.0, 0.0),
        pack(0.0, 0.0),
        pack(1.0, -1.0),
        pack(1.0, -1.0),
        pack(0.5, 0.5),
        pack(2.0, 0.5),
        pack(3.0, 4.0),
        pack(-1.0, 2.0),
        pack(10.0, -10.0),
        pack(0.1, 0.1),
        pack(5.0, 5.0),
        pack(0.25, 0.25),
        pack(4.0, 4.0),
        pack(0.5, 0.5),
        pack(-2.0, 3.0),
        pack(0.5, -0.5),
        pack(1.0, -1.0),
        pack(100.0, 0.01),
        pack(0.01, 100.0),
        pack(0.0, 0.0),
        pack(0.5, 0.5),
        pack(0.5, 0.5),
        pack(0.5, 0.5),
    ]
    return vals[:30]


def ds_test_data() -> list[int]:
    """16 data values for DS read/write/atomic tests."""
    return [
        0x0000_0000,
        0xFFFF_FFFF,
        0x8000_0000,
        0x7FFF_FFFF,
        0xDEAD_BEEF,
        0xCAFE_BABE,
        0x1234_5678,
        0x0000_0001,
        0x0000_0002,
        0x0000_000A,
        0x5555_5555,
        0xAAAA_AAAA,
        0x0100_0000,
        0x00FF_00FF,
        0xFF00_FF00,
        0x4000_0000,
    ]


def ds_addr_offset_pairs() -> list[tuple[int, int]]:
    """8 (addr, offset) pairs for DS tests. All effective addrs within 64KB LDS."""
    return [
        (0, 0),
        (0, 128),
        (256, 0),
        (256, 64),
        (1024, 0),
        (0, 1024),
        (4096, 256),
        (8192, 0),
    ]


def ds_atomic_pairs() -> list[tuple[int, int]]:
    """8 (initial_lds, operand) pairs for DS atomic tests."""
    return [
        (0x0000_0000, 0x0000_0001),
        (0x0000_000A, 0x0000_0005),
        (0xFFFF_FFFF, 0x0000_0001),
        (0x8000_0000, 0x7FFF_FFFF),
        (0x0000_0001, 0x0000_0000),
        (0x5555_5555, 0xAAAA_AAAA),
        (0x0000_0000, 0xFFFF_FFFF),
        (0x1234_5678, 0x8765_4321),
    ]


def inputs_for_dtype(dtype: str | None, seed: int = 0) -> list[int]:
    """Return 32 test input values (as raw uint32) for the given data type."""
    if dtype in ('f32', 'f16', 'bf16'):
        return float_f32_bits(seed)
    if dtype in ('b64', 'i64', 'u64', 'f64'):
        pairs = int_u64_inputs(seed)
        flat: list[int] = []
        for lo, hi in pairs:
            flat.extend([lo, hi])
        return flat
    return int_u32_inputs(seed)
