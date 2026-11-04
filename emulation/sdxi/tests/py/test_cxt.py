"""Context-table DMA, compression, and performance-counter tests.

Exercises the v3 descriptor path: two-level context table, DMAB COPY,
DMAB COMPRESS, DMAB DECOMPRESS, and the CTR_SELECT/CTR_VAL counter
interface.

Usage (via CTest):
    ctest -R sdxi-cxt -V

Or directly:
    SDXI_SERVER_BIN=./build/rocm-sdxi python3 tests/py/test_cxt.py

Copyright (C) Advanced Micro Devices, Inc.

SPDX-License-Identifier: MIT
"""

import os
import zlib

from sdxi.check import Suite
from sdxi.cxt_driver import (
    CTR_BYTES_COMPRESSED,
    CTR_BYTES_DECOMPRESSED,
    CTR_BYTES_MOVED,
    CTR_NUM_COMPRESS,
    CTR_NUM_COPY,
    CTR_NUM_DECOMPRESS,
    CTR_NUM_ERRORS,
    CxtDevice,
)
from sdxi.server import Server, stop_and_check
from sdxi.vfu import GuestMemory


def test_cxt_copy(suite, dev, mem):
    """DMAB COPY via context-table path moves data correctly."""
    cxt = dev.cxt(1)
    src = mem.alloc(4096)
    dst = mem.alloc(4096)
    pattern = bytes(range(256)) * 16
    mem.write(src, pattern)
    ok = cxt.copy(dev.vfu, src, dst, len(pattern))
    suite.check(ok, "cxt COPY CSB reports success")
    result = mem.read(dst, len(pattern))
    suite.check_eq(result, pattern, "cxt COPY data correct")


def test_cxt_copy_counter(suite, dev, mem):
    """bytes_moved and num_copy counters increment after COPY."""
    cxt = dev.cxt(1)
    cid = cxt.cid

    before_bytes = dev.read_counter(cid, CTR_BYTES_MOVED)
    before_copy  = dev.read_counter(cid, CTR_NUM_COPY)

    src = mem.alloc(4096)
    dst = mem.alloc(4096)
    length = 1024
    mem.write(src, bytes(range(256)) * 4)
    ok = cxt.copy(dev.vfu, src, dst, length)
    suite.check(ok, "COPY for counter test succeeded")

    after_bytes = dev.read_counter(cid, CTR_BYTES_MOVED)
    after_copy  = dev.read_counter(cid, CTR_NUM_COPY)

    suite.check_eq(after_bytes - before_bytes, length, "bytes_moved incremented by copy length")
    suite.check_eq(after_copy  - before_copy,  1,      "num_copy incremented by 1")


def test_compress_roundtrip(suite, dev, mem):
    """COMPRESS followed by DECOMPRESS recovers the original data."""
    cxt = dev.cxt(1)

    # Highly compressible input
    original = b"SDXI compress test\n" * 512
    src_len  = len(original)
    dst_cap  = src_len * 2   # generous bound

    src_gpa  = mem.alloc(src_len)
    comp_gpa = mem.alloc(dst_cap)
    dec_gpa  = mem.alloc(dst_cap)

    mem.write(src_gpa, original)

    ok, comp_len = cxt.compress(dev.vfu, src_gpa, src_len, comp_gpa, dst_cap, level=6)
    suite.check(ok, "COMPRESS CSB reports success")
    suite.check(comp_len > 0,       "compressed length > 0")
    suite.check(comp_len < src_len, "compressed length < original (data was compressible)")

    ok2, dec_len = cxt.decompress(dev.vfu, comp_gpa, int(comp_len), dec_gpa, dst_cap)
    suite.check(ok2, "DECOMPRESS CSB reports success")
    suite.check_eq(dec_len, src_len, "decompressed length == original length")

    recovered = mem.read(dec_gpa, src_len)
    suite.check_eq(recovered, original, "decompressed data matches original")


def test_compress_counters(suite, dev, mem):
    """Compression counters (bytes_compressed, num_compress, etc.) increment."""
    cxt = dev.cxt(2)
    cid = cxt.cid

    original = b"counter test payload " * 200
    src_len  = len(original)
    dst_cap  = src_len * 2

    src_gpa  = mem.alloc(src_len)
    comp_gpa = mem.alloc(dst_cap)
    dec_gpa  = mem.alloc(dst_cap)
    mem.write(src_gpa, original)

    b_comp_before   = dev.read_counter(cid, CTR_BYTES_COMPRESSED)
    n_comp_before   = dev.read_counter(cid, CTR_NUM_COMPRESS)
    b_decomp_before = dev.read_counter(cid, CTR_BYTES_DECOMPRESSED)
    n_decomp_before = dev.read_counter(cid, CTR_NUM_DECOMPRESS)
    err_before      = dev.read_counter(cid, CTR_NUM_ERRORS)

    ok, comp_len = cxt.compress(dev.vfu, src_gpa, src_len, comp_gpa, dst_cap)
    suite.check(ok, "COMPRESS for counter test succeeded")

    ok2, dec_len = cxt.decompress(dev.vfu, comp_gpa, int(comp_len), dec_gpa, dst_cap)
    suite.check(ok2, "DECOMPRESS for counter test succeeded")

    suite.check_eq(dev.read_counter(cid, CTR_BYTES_COMPRESSED)   - b_comp_before,   src_len, "bytes_compressed += src_len")
    suite.check_eq(dev.read_counter(cid, CTR_NUM_COMPRESS)       - n_comp_before,   1,       "num_compress += 1")
    suite.check_eq(dev.read_counter(cid, CTR_BYTES_DECOMPRESSED) - b_decomp_before, src_len, "bytes_decompressed += dec_len")
    suite.check_eq(dev.read_counter(cid, CTR_NUM_DECOMPRESS)     - n_decomp_before, 1,       "num_decompress += 1")
    suite.check_eq(dev.read_counter(cid, CTR_NUM_ERRORS)         - err_before,      0,       "no errors")


def test_error_counter(suite, dev, mem):
    """A COPY with an unmapped address increments num_errors."""
    cxt = dev.cxt(1)
    cid = cxt.cid

    err_before = dev.read_counter(cid, CTR_NUM_ERRORS)

    # Use a GPA outside the registered DMA region to force an SGL map failure.
    BAD_GPA = 0xDEAD_0000_0000
    src = mem.alloc(64)
    mem.write(src, b"x" * 64)
    ok = cxt.copy(dev.vfu, src, BAD_GPA, 64)
    suite.check(not ok, "COPY to bad GPA reports failure in CSB")

    err_after = dev.read_counter(cid, CTR_NUM_ERRORS)
    suite.check(err_after > err_before, "num_errors incremented after failed COPY")


def main():
    binary = os.environ.get(
        "SDXI_SERVER_BIN",
        os.path.join(os.path.dirname(__file__), "..", "..", "build", "rocm-sdxi"),
    )
    suite = Suite("SDXI context-table tests")
    server = Server(binary)
    server.start()
    try:
        with GuestMemory(size=256 << 20, name="sdxi-cxt-mem") as mem:
            with server.connect() as vfu:
                vfu.dma_map(mem)
                dev = CxtDevice(vfu, mem, num_cxt=4, ring_sz=32)
                suite.run("cxt_copy",          test_cxt_copy,          dev, mem)
                suite.run("cxt_copy_counter",  test_cxt_copy_counter,  dev, mem)
                suite.run("compress_roundtrip", test_compress_roundtrip, dev, mem)
                suite.run("compress_counters",  test_compress_counters,  dev, mem)
                suite.run("error_counter",      test_error_counter,      dev, mem)
    finally:
        stop_and_check(suite, server)
    suite.finish()


if __name__ == "__main__":
    main()
