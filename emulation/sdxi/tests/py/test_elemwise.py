"""COMPUTE ELEMWISE descriptor tests.

Exercises the TYPE_COMPUTE / SUBTYPE_ELEMWISE path: elementwise arithmetic
(add, min, max) and reductive operations (reduce_min, reduce_max) over
FP32, FP16, BF16, INT8, INT32, FP8_E4M3, and INT4 element types.

Usage (via CTest):
    ctest -R sdxi-elemwise -V

Or directly:
    SDXI_SERVER_BIN=./build/rocm-sdxi python3 tests/py/test_elemwise.py

Copyright (C) Advanced Micro Devices, Inc.

SPDX-License-Identifier: MIT
"""

import os
import struct

from sdxi.check import Suite
from sdxi.cxt_driver import (
    DTYPE_BF16,
    DTYPE_FP16,
    DTYPE_FP32,
    DTYPE_FP8_E4M3,
    DTYPE_FP8_E5M2,
    DTYPE_INT4,
    DTYPE_INT8,
    DTYPE_INT32,
    ELEMOP_ADD,
    ELEMOP_MAX,
    ELEMOP_MIN,
    ELEMOP_REDUCE_MAX,
    ELEMOP_REDUCE_MIN,
    ELEM_SIZE,
    CxtDevice,
)
from sdxi.server import Server, stop_and_check
from sdxi.vfu import GuestMemory


# ---------------------------------------------------------------------------
# Scalar encoding helpers
# ---------------------------------------------------------------------------

def _fp32_bits(v):
    return struct.unpack("<I", struct.pack("<f", v))[0]


def _fp16_bits(v):
    return struct.unpack("<H", struct.pack("<e", v))[0]


def _bf16_bits(v):
    return _fp32_bits(v) >> 16


def _int_scalar(v, signed_width=64):
    if v < 0:
        return v & ((1 << signed_width) - 1)
    return v


# ---------------------------------------------------------------------------
# Payload packing helpers
# ---------------------------------------------------------------------------

def _pack_int8(values):
    return struct.pack("b" * len(values), *values)


def _unpack_int8(data):
    return list(struct.unpack("b" * len(data), data))


def _pack_int32(values):
    return struct.pack("<" + "i" * len(values), *values)


def _unpack_int32(data):
    n = len(data) // 4
    return list(struct.unpack("<" + "i" * n, data))


def _pack_fp32(values):
    return struct.pack("<" + "f" * len(values), *values)


def _unpack_fp32(data):
    n = len(data) // 4
    return list(struct.unpack("<" + "f" * n, data))


def _pack_fp16(values):
    return struct.pack("<" + "e" * len(values), *values)


def _unpack_fp16(data):
    n = len(data) // 2
    return list(struct.unpack("<" + "e" * n, data))


def _pack_bf16(values):
    out = bytearray()
    for v in values:
        bits = _fp32_bits(v)
        out += struct.pack("<H", bits >> 16)
    return bytes(out)


def _unpack_bf16(data):
    n = len(data) // 2
    raw = struct.unpack("<" + "H" * n, data)
    result = []
    for h in raw:
        bits = h << 16
        result.append(struct.unpack("<f", struct.pack("<I", bits))[0])
    return result


def _pack_fp8_e4m3(raw_bytes):
    """raw_bytes is a list of uint8 OCP E4M3 bit patterns."""
    return bytes(raw_bytes)


def _unpack_fp8_e4m3(data):
    return list(data)


def _pack_fp8_e5m2(raw_bytes):
    """raw_bytes is a list of uint8 OCP E5M2 bit patterns."""
    return bytes(raw_bytes)


def _unpack_fp8_e5m2(data):
    return list(data)


def _pack_int4(values):
    """Pack signed −8..7 values, two per byte, low nibble first."""
    out = bytearray()
    for i in range(0, len(values), 2):
        lo = values[i] & 0xF
        hi = (values[i + 1] & 0xF) if i + 1 < len(values) else 0
        out.append(lo | (hi << 4))
    return bytes(out)


def _unpack_int4(data, n_elems):
    result = []
    for byte in data:
        lo = byte & 0xF
        hi = (byte >> 4) & 0xF
        result.append(lo if lo < 8 else lo - 16)
        result.append(hi if hi < 8 else hi - 16)
        if len(result) >= n_elems:
            break
    return result[:n_elems]


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_elemwise_int8_add(suite, dev, mem):
    """INT8 elementwise add: each element += scalar."""
    cxt = dev.cxt(1)
    src_data = _pack_int8([1, 2, 3, 10, -5])
    scalar = _int_scalar(5)
    expected = _pack_int8([6, 7, 8, 15, 0])

    src = mem.alloc(len(src_data))
    dst = mem.alloc(len(src_data))
    mem.write(src, src_data)

    ok, _ = cxt.elemwise(dev.vfu, src, len(src_data), dst, DTYPE_INT8, ELEMOP_ADD, scalar)
    suite.check(ok, "INT8 ADD CSB reports success")
    result = mem.read(dst, len(expected))
    suite.check_eq(result, expected, "INT8 ADD result correct")


def test_elemwise_int8_min(suite, dev, mem):
    """INT8 elementwise min: clamp elements to scalar."""
    cxt = dev.cxt(1)
    src_data = _pack_int8([1, 2, 3, 10, -5])
    scalar = _int_scalar(3)
    expected = _pack_int8([1, 2, 3, 3, -5])

    src = mem.alloc(len(src_data))
    dst = mem.alloc(len(src_data))
    mem.write(src, src_data)

    ok, _ = cxt.elemwise(dev.vfu, src, len(src_data), dst, DTYPE_INT8, ELEMOP_MIN, scalar)
    suite.check(ok, "INT8 MIN CSB reports success")
    result = mem.read(dst, len(expected))
    suite.check_eq(result, expected, "INT8 MIN result correct")


def test_elemwise_int8_max(suite, dev, mem):
    """INT8 elementwise max: lift elements to scalar."""
    cxt = dev.cxt(1)
    src_data = _pack_int8([1, 2, 3, 10, -5])
    scalar = _int_scalar(3)
    expected = _pack_int8([3, 3, 3, 10, 3])

    src = mem.alloc(len(src_data))
    dst = mem.alloc(len(src_data))
    mem.write(src, src_data)

    ok, _ = cxt.elemwise(dev.vfu, src, len(src_data), dst, DTYPE_INT8, ELEMOP_MAX, scalar)
    suite.check(ok, "INT8 MAX CSB reports success")
    result = mem.read(dst, len(expected))
    suite.check_eq(result, expected, "INT8 MAX result correct")


def test_elemwise_int8_reduce_min(suite, dev, mem):
    """INT8 reduce_min: single output element is minimum of all inputs."""
    cxt = dev.cxt(1)
    src_data = _pack_int8([3, 1, 4, -2, 7])
    expected = _pack_int8([-2])

    src = mem.alloc(len(src_data))
    dst = mem.alloc(ELEM_SIZE[DTYPE_INT8])
    mem.write(src, src_data)

    ok, _ = cxt.elemwise(dev.vfu, src, len(src_data), dst, DTYPE_INT8, ELEMOP_REDUCE_MIN)
    suite.check(ok, "INT8 REDUCE_MIN CSB reports success")
    result = mem.read(dst, 1)
    suite.check_eq(result, expected, "INT8 REDUCE_MIN result correct")


def test_elemwise_int8_reduce_max(suite, dev, mem):
    """INT8 reduce_max: single output element is maximum of all inputs."""
    cxt = dev.cxt(1)
    src_data = _pack_int8([3, 1, 4, -2, 7])
    expected = _pack_int8([7])

    src = mem.alloc(len(src_data))
    dst = mem.alloc(ELEM_SIZE[DTYPE_INT8])
    mem.write(src, src_data)

    ok, _ = cxt.elemwise(dev.vfu, src, len(src_data), dst, DTYPE_INT8, ELEMOP_REDUCE_MAX)
    suite.check(ok, "INT8 REDUCE_MAX CSB reports success")
    result = mem.read(dst, 1)
    suite.check_eq(result, expected, "INT8 REDUCE_MAX result correct")


def test_elemwise_int32_add(suite, dev, mem):
    """INT32 elementwise add."""
    cxt = dev.cxt(1)
    values = [100, 200, -50, 0]
    src_data = _pack_int32(values)
    scalar = _int_scalar(10)
    expected = _pack_int32([v + 10 for v in values])

    src = mem.alloc(len(src_data))
    dst = mem.alloc(len(src_data))
    mem.write(src, src_data)

    ok, _ = cxt.elemwise(dev.vfu, src, len(src_data), dst, DTYPE_INT32, ELEMOP_ADD, scalar)
    suite.check(ok, "INT32 ADD CSB reports success")
    result = mem.read(dst, len(expected))
    suite.check_eq(result, expected, "INT32 ADD result correct")


def test_elemwise_int32_reduce_min(suite, dev, mem):
    """INT32 reduce_min."""
    cxt = dev.cxt(1)
    values = [100, 200, 50, 300]
    src_data = _pack_int32(values)
    expected = _pack_int32([50])

    src = mem.alloc(len(src_data))
    dst = mem.alloc(ELEM_SIZE[DTYPE_INT32])
    mem.write(src, src_data)

    ok, _ = cxt.elemwise(dev.vfu, src, len(src_data), dst, DTYPE_INT32, ELEMOP_REDUCE_MIN)
    suite.check(ok, "INT32 REDUCE_MIN CSB reports success")
    result = mem.read(dst, 4)
    suite.check_eq(result, expected, "INT32 REDUCE_MIN result correct")


def test_elemwise_fp32_add(suite, dev, mem):
    """FP32 elementwise add."""
    cxt = dev.cxt(1)
    values = [1.0, 2.0, 3.0, -1.5]
    src_data = _pack_fp32(values)
    scalar = _fp32_bits(1.5)
    expected = _pack_fp32([v + 1.5 for v in values])

    src = mem.alloc(len(src_data))
    dst = mem.alloc(len(src_data))
    mem.write(src, src_data)

    ok, _ = cxt.elemwise(dev.vfu, src, len(src_data), dst, DTYPE_FP32, ELEMOP_ADD, scalar)
    suite.check(ok, "FP32 ADD CSB reports success")
    result = mem.read(dst, len(expected))
    result_vals = _unpack_fp32(result)
    expected_vals = _unpack_fp32(expected)
    for i, (r, e) in enumerate(zip(result_vals, expected_vals)):
        suite.check(abs(r - e) < 1e-5, f"FP32 ADD element {i}: {r} ~= {e}")


def test_elemwise_fp32_reduce_max(suite, dev, mem):
    """FP32 reduce_max."""
    cxt = dev.cxt(1)
    values = [1.0, 3.5, 2.0, -1.0]
    src_data = _pack_fp32(values)
    expected_val = 3.5

    src = mem.alloc(len(src_data))
    dst = mem.alloc(ELEM_SIZE[DTYPE_FP32])
    mem.write(src, src_data)

    ok, _ = cxt.elemwise(dev.vfu, src, len(src_data), dst, DTYPE_FP32, ELEMOP_REDUCE_MAX)
    suite.check(ok, "FP32 REDUCE_MAX CSB reports success")
    result = mem.read(dst, 4)
    (r,) = _unpack_fp32(result)
    suite.check(abs(r - expected_val) < 1e-5, f"FP32 REDUCE_MAX: {r} ~= {expected_val}")


def test_elemwise_fp16_add(suite, dev, mem):
    """FP16 elementwise add."""
    cxt = dev.cxt(1)
    values = [1.0, 2.0, 3.0]
    src_data = _pack_fp16(values)
    scalar = _fp16_bits(0.5)
    expected_vals = [v + 0.5 for v in values]

    src = mem.alloc(len(src_data))
    dst = mem.alloc(len(src_data))
    mem.write(src, src_data)

    ok, _ = cxt.elemwise(dev.vfu, src, len(src_data), dst, DTYPE_FP16, ELEMOP_ADD, scalar)
    suite.check(ok, "FP16 ADD CSB reports success")
    result = mem.read(dst, len(src_data))
    result_vals = _unpack_fp16(result)
    for i, (r, e) in enumerate(zip(result_vals, expected_vals)):
        suite.check(abs(r - e) < 1e-2, f"FP16 ADD element {i}: {r} ~= {e}")


def test_elemwise_bf16_add(suite, dev, mem):
    """BF16 elementwise add."""
    cxt = dev.cxt(1)
    values = [1.0, 2.0, 4.0]
    src_data = _pack_bf16(values)
    scalar = _bf16_bits(1.0)
    expected_vals = [v + 1.0 for v in values]

    src = mem.alloc(len(src_data))
    dst = mem.alloc(len(src_data))
    mem.write(src, src_data)

    ok, _ = cxt.elemwise(dev.vfu, src, len(src_data), dst, DTYPE_BF16, ELEMOP_ADD, scalar)
    suite.check(ok, "BF16 ADD CSB reports success")
    result = mem.read(dst, len(src_data))
    result_vals = _unpack_bf16(result)
    for i, (r, e) in enumerate(zip(result_vals, expected_vals)):
        suite.check(abs(r - e) < 0.1, f"BF16 ADD element {i}: {r} ~= {e}")


def test_elemwise_fp8_e4m3_add(suite, dev, mem):
    """FP8 E4M3 elementwise add with known bit patterns.

    0x38 encodes 1.0 in OCP E4M3 (sign=0 exp=0111=7-bias7=0 mant=000).
    0x40 encodes 2.0 (sign=0 exp=1000=8-7=1 mant=000).
    Adding 1.0 (0x38) to each: [1.0, 2.0] → [2.0, 3.0] → [0x40, 0x44].
    0x44 = sign=0 exp=1000=1 mant=100 = 1 + 0.5 ... actually:
    E4M3: value = (-1)^s * 2^(exp-7) * (1 + mant/8)
    0x40 = 0 1000 000 = 2^(8-7) * (1+0) = 2.0  ✓
    0x44 = 0 1000 100 = 2^(8-7) * (1+4/8) = 2 * 1.5 = 3.0  ✓
    """
    cxt = dev.cxt(1)
    src_raw = [0x38, 0x40]   # [1.0, 2.0]
    scalar = 0x38             # 1.0 in E4M3
    expected_raw = [0x40, 0x44]  # [2.0, 3.0]

    src_data = _pack_fp8_e4m3(src_raw)
    expected = _pack_fp8_e4m3(expected_raw)

    src = mem.alloc(len(src_data))
    dst = mem.alloc(len(src_data))
    mem.write(src, src_data)

    ok, _ = cxt.elemwise(dev.vfu, src, len(src_data), dst, DTYPE_FP8_E4M3, ELEMOP_ADD, scalar)
    suite.check(ok, "FP8_E4M3 ADD CSB reports success")
    result = mem.read(dst, len(expected))
    result_raw = _unpack_fp8_e4m3(result)
    suite.check_eq(result, expected, f"FP8_E4M3 ADD: got {[hex(b) for b in result_raw]}")


def test_elemwise_int4_add(suite, dev, mem):
    """INT4 elementwise add with nibble-packed input.

    4 elements [1, 2, 3, -1] packed in 2 bytes; scalar = 2.
    Expected: [3, 4, 5, 1].
    """
    cxt = dev.cxt(1)
    values = [1, 2, 3, -1]
    src_data = _pack_int4(values)   # 2 bytes
    scalar = 2
    expected_values = [3, 4, 5, 1]
    expected = _pack_int4(expected_values)

    src = mem.alloc(len(src_data))
    dst = mem.alloc(len(src_data))
    mem.write(src, src_data)

    ok, _ = cxt.elemwise(dev.vfu, src, len(src_data), dst, DTYPE_INT4, ELEMOP_ADD, scalar)
    suite.check(ok, "INT4 ADD CSB reports success")
    result = mem.read(dst, len(expected))
    result_vals = _unpack_int4(result, len(values))
    suite.check_eq(result_vals, expected_values, f"INT4 ADD result: {result_vals}")


def test_elemwise_int4_reduce_min(suite, dev, mem):
    """INT4 reduce_min across nibble-packed elements.

    4 elements [1, 2, 3, -1]; min = -1.
    Result is 1 byte with the minimum in the low nibble: 0x0F.
    """
    cxt = dev.cxt(1)
    values = [1, 2, 3, -1]
    src_data = _pack_int4(values)
    expected = _pack_int4([-1])  # 0x0F

    src = mem.alloc(len(src_data))
    dst = mem.alloc(ELEM_SIZE[DTYPE_INT4])
    mem.write(src, src_data)

    ok, _ = cxt.elemwise(dev.vfu, src, len(src_data), dst, DTYPE_INT4, ELEMOP_REDUCE_MIN)
    suite.check(ok, "INT4 REDUCE_MIN CSB reports success")
    result = mem.read(dst, ELEM_SIZE[DTYPE_INT4])
    result_vals = _unpack_int4(result, 1)
    suite.check_eq(result_vals, [-1], f"INT4 REDUCE_MIN result: {result_vals}")


def test_elemwise_int4_reduce_max(suite, dev, mem):
    """INT4 reduce_max across nibble-packed elements.

    4 elements [1, 2, 3, -1]; max = 3.
    Result is 1 byte with the maximum in the low nibble: 0x03.
    """
    cxt = dev.cxt(1)
    values = [1, 2, 3, -1]
    src_data = _pack_int4(values)
    expected = _pack_int4([3])  # 0x03

    src = mem.alloc(len(src_data))
    dst = mem.alloc(ELEM_SIZE[DTYPE_INT4])
    mem.write(src, src_data)

    ok, _ = cxt.elemwise(dev.vfu, src, len(src_data), dst, DTYPE_INT4, ELEMOP_REDUCE_MAX)
    suite.check(ok, "INT4 REDUCE_MAX CSB reports success")
    result = mem.read(dst, ELEM_SIZE[DTYPE_INT4])
    result_vals = _unpack_int4(result, 1)
    suite.check_eq(result_vals, [3], f"INT4 REDUCE_MAX result: {result_vals}")


def test_elemwise_fp8_e5m2_add(suite, dev, mem):
    """FP8 E5M2 elementwise add with known bit patterns.

    OCP E5M2: bias=15, 5-bit exponent, 2-bit mantissa.
    0x3C = 0 01111 00 = 2^(15-15) * 1.0 = 1.0
    0x40 = 0 10000 00 = 2^(16-15) * 1.0 = 2.0
    0x42 = 0 10000 10 = 2^(16-15) * 1.5 = 3.0
    Adding 1.0 (0x3C): [1.0, 2.0] → [2.0, 3.0] → [0x40, 0x42].
    """
    cxt = dev.cxt(1)
    src_raw = [0x3C, 0x40]   # [1.0, 2.0]
    scalar = 0x3C             # 1.0 in E5M2
    expected_raw = [0x40, 0x42]  # [2.0, 3.0]

    src_data = _pack_fp8_e5m2(src_raw)
    expected = _pack_fp8_e5m2(expected_raw)

    src = mem.alloc(len(src_data))
    dst = mem.alloc(len(src_data))
    mem.write(src, src_data)

    ok, _ = cxt.elemwise(dev.vfu, src, len(src_data), dst, DTYPE_FP8_E5M2, ELEMOP_ADD, scalar)
    suite.check(ok, "FP8_E5M2 ADD CSB reports success")
    result = mem.read(dst, len(expected))
    result_raw = _unpack_fp8_e5m2(result)
    suite.check_eq(result, expected, f"FP8_E5M2 ADD: got {[hex(b) for b in result_raw]}")


def test_elemwise_fp8_e5m2_reduce_max(suite, dev, mem):
    """FP8 E5M2 reduce_max.

    Input: [1.0, 2.0, 3.0] = [0x3C, 0x40, 0x42]; expected max = 3.0 = 0x42.
    """
    cxt = dev.cxt(1)
    src_raw = [0x3C, 0x40, 0x42]   # [1.0, 2.0, 3.0]
    expected_raw = [0x42]           # 3.0

    src_data = _pack_fp8_e5m2(src_raw)
    expected = _pack_fp8_e5m2(expected_raw)

    src = mem.alloc(len(src_data))
    dst = mem.alloc(ELEM_SIZE[DTYPE_FP8_E5M2])
    mem.write(src, src_data)

    ok, _ = cxt.elemwise(dev.vfu, src, len(src_data), dst, DTYPE_FP8_E5M2, ELEMOP_REDUCE_MAX)
    suite.check(ok, "FP8_E5M2 REDUCE_MAX CSB reports success")
    result = mem.read(dst, ELEM_SIZE[DTYPE_FP8_E5M2])
    result_raw = _unpack_fp8_e5m2(result)
    suite.check_eq(result, expected, f"FP8_E5M2 REDUCE_MAX: got {[hex(b) for b in result_raw]}")


def main():
    binary = os.environ.get(
        "SDXI_SERVER_BIN",
        os.path.join(os.path.dirname(__file__), "..", "..", "build", "rocm-sdxi"),
    )
    suite = Suite("SDXI COMPUTE ELEMWISE tests")
    server = Server(binary)
    server.start()
    try:
        with GuestMemory(size=256 << 20, name="sdxi-elemwise-mem") as mem:
            with server.connect() as vfu:
                vfu.dma_map(mem)
                dev = CxtDevice(vfu, mem, num_cxt=4, ring_sz=32)
                suite.run("int8_add",        test_elemwise_int8_add,        dev, mem)
                suite.run("int8_min",        test_elemwise_int8_min,        dev, mem)
                suite.run("int8_max",        test_elemwise_int8_max,        dev, mem)
                suite.run("int8_reduce_min", test_elemwise_int8_reduce_min, dev, mem)
                suite.run("int8_reduce_max", test_elemwise_int8_reduce_max, dev, mem)
                suite.run("int32_add",       test_elemwise_int32_add,       dev, mem)
                suite.run("int32_reduce_min",test_elemwise_int32_reduce_min,dev, mem)
                suite.run("fp32_add",        test_elemwise_fp32_add,        dev, mem)
                suite.run("fp32_reduce_max", test_elemwise_fp32_reduce_max, dev, mem)
                suite.run("fp16_add",        test_elemwise_fp16_add,        dev, mem)
                suite.run("bf16_add",        test_elemwise_bf16_add,        dev, mem)
                suite.run("fp8_e4m3_add",        test_elemwise_fp8_e4m3_add,       dev, mem)
                suite.run("int4_add",            test_elemwise_int4_add,            dev, mem)
                suite.run("int4_reduce_min",     test_elemwise_int4_reduce_min,     dev, mem)
                suite.run("int4_reduce_max",     test_elemwise_int4_reduce_max,     dev, mem)
                suite.run("fp8_e5m2_add",        test_elemwise_fp8_e5m2_add,        dev, mem)
                suite.run("fp8_e5m2_reduce_max", test_elemwise_fp8_e5m2_reduce_max, dev, mem)
    finally:
        stop_and_check(suite, server)
    suite.finish()


if __name__ == "__main__":
    main()
