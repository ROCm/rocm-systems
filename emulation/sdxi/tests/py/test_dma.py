"""Functional DMA tests for the SDXI vfio-user emulator.

Submits MEMCPY, MEMSET, FENCE, and bad-opcode descriptors and verifies
the CQE status and data integrity. Each test gets its own queue to keep
them independent.

Usage (via CTest):
    ctest -R sdxi-dma -V

Or directly:
    SDXI_SERVER_BIN=./build/rocm-sdxi python3 tests/py/test_dma.py

Copyright (C) Advanced Micro Devices, Inc.

SPDX-License-Identifier: MIT
"""

import os
import struct

from sdxi.check import Suite
from sdxi.sdxi_driver import (
    CQE_STATUS_INVAL,
    CQE_STATUS_OK,
    OP_FENCE,
    OP_MEMCPY,
    OP_MEMSET,
    SdxiDevice,
    pack_sqe,
)
from sdxi.server import Server, stop_and_check
from sdxi.vfu import GuestMemory


def _mkdev(vfu):
    return SdxiDevice(vfu)


def test_memcpy(suite, dev, mem, qid):
    q = dev.queue(mem, qid)
    # Write a recognisable pattern at the source
    src = mem.alloc(4096)
    dst = mem.alloc(4096)
    pattern = bytes(range(256)) * 16
    mem.write(src, pattern)
    sqe = pack_sqe(OP_MEMCPY, 0, qid, 1, src, dst, len(pattern))
    q.submit(sqe)
    status, cid, _sq_head = q.wait_cq()
    suite.check_eq(status, CQE_STATUS_OK, "MEMCPY CQE status == OK")
    suite.check_eq(cid, 1, "MEMCPY CQE cid echoed")
    result = mem.read(dst, len(pattern))
    suite.check_eq(result, pattern, "MEMCPY data matches source")


def test_memset(suite, dev, mem, qid):
    q = dev.queue(mem, qid)
    dst = mem.alloc(4096)
    fill_byte = 0xAB
    length = 256
    sqe = pack_sqe(OP_MEMSET, 0, qid, 2, fill_byte, dst, length)
    q.submit(sqe)
    status, cid, _sq_head = q.wait_cq()
    suite.check_eq(status, CQE_STATUS_OK, "MEMSET CQE status == OK")
    suite.check_eq(cid, 2, "MEMSET CQE cid echoed")
    result = mem.read(dst, length)
    suite.check_eq(result, bytes([fill_byte]) * length, "MEMSET filled correctly")


def test_fence(suite, dev, mem, qid):
    q = dev.queue(mem, qid)
    sqe = pack_sqe(OP_FENCE, 0, qid, 3, 0, 0, 0)
    q.submit(sqe)
    status, cid, _sq_head = q.wait_cq()
    suite.check_eq(status, CQE_STATUS_OK, "FENCE CQE status == OK")
    suite.check_eq(cid, 3, "FENCE CQE cid echoed")


def test_bad_opcode(suite, dev, mem, qid):
    q = dev.queue(mem, qid)
    sqe = pack_sqe(0xFF, 0, qid, 4, 0, 0, 0)
    q.submit(sqe)
    status, cid, _sq_head = q.wait_cq()
    suite.check_eq(status, CQE_STATUS_INVAL, "bad opcode CQE status == INVAL")
    suite.check_eq(cid, 4, "bad opcode CQE cid echoed")


def test_multi_command(suite, dev, mem, qid):
    """Submit several FENCEs in one burst and collect all CQEs."""
    q = dev.queue(mem, qid)
    n = 4
    for i in range(n):
        sqe = pack_sqe(OP_FENCE, 0, qid, 10 + i, 0, 0, 0)
        q.submit(sqe)
    for i in range(n):
        status, cid, _sq_head = q.wait_cq()
        suite.check_eq(status, CQE_STATUS_OK, "multi[%d] status == OK", i)
        suite.check_eq(cid, 10 + i, "multi[%d] cid == %d", i, 10 + i)


def test_zero_length(suite, dev, mem, qid):
    """Zero-length MEMCPY and MEMSET both complete with status OK."""
    q = dev.queue(mem, qid)
    src = mem.alloc(64)
    dst = mem.alloc(64)

    sqe = pack_sqe(OP_MEMCPY, 0, qid, 20, src, dst, 0)
    q.submit(sqe)
    status, cid, _ = q.wait_cq()
    suite.check_eq(status, CQE_STATUS_OK, "zero-len MEMCPY status == OK")
    suite.check_eq(cid, 20, "zero-len MEMCPY cid echoed")

    sqe = pack_sqe(OP_MEMSET, 0, qid, 21, 0xAB, dst, 0)
    q.submit(sqe)
    status, cid, _ = q.wait_cq()
    suite.check_eq(status, CQE_STATUS_OK, "zero-len MEMSET status == OK")
    suite.check_eq(cid, 21, "zero-len MEMSET cid echoed")


def test_boundary_crossing(suite, dev, mem, qid):
    """MEMCPY of a buffer that straddles a 4 KiB page boundary."""
    PAGE = 4096
    q = dev.queue(mem, qid)
    # Advance the allocator so the next byte-aligned alloc straddles a page.
    offset_in_page = mem._next % PAGE
    gap = (PAGE - 128 - offset_in_page) % PAGE
    if gap:
        mem.alloc(gap, align=1)
    xfer_size = 512  # 128 bytes before the boundary, 384 after
    cid = 30
    src = mem.alloc(xfer_size, align=1)
    dst = mem.alloc(xfer_size, align=1)

    pattern = bytes(i & 0xFF for i in range(xfer_size))
    mem.write(src, pattern)
    sqe = pack_sqe(OP_MEMCPY, 0, qid, cid, src, dst, xfer_size)
    q.submit(sqe)
    status, cid_got, _ = q.wait_cq()
    suite.check_eq(status, CQE_STATUS_OK, "boundary MEMCPY status == OK")
    suite.check_eq(cid_got, cid, "boundary MEMCPY cid echoed")
    result = mem.read(dst, xfer_size)
    suite.check_eq(result, pattern, "boundary MEMCPY data correct")


def test_multi_queue_sim(suite, dev, mem, base_qid):
    """Submit to three queues simultaneously then drain all completions."""
    n = 3
    queues = [dev.queue(mem, base_qid + i) for i in range(n)]
    for i, q in enumerate(queues):
        sqe = pack_sqe(OP_FENCE, 0, base_qid + i, 40 + i, 0, 0, 0)
        q.submit(sqe)
    for i, q in enumerate(queues):
        status, cid, _ = q.wait_cq()
        suite.check_eq(status, CQE_STATUS_OK, "multi-q[%d] status == OK", i)
        suite.check_eq(cid, 40 + i, "multi-q[%d] cid == %d", i, 40 + i)


def main():
    binary = os.environ.get(
        "SDXI_SERVER_BIN",
        os.path.join(os.path.dirname(__file__), "..", "..", "build", "rocm-sdxi"),
    )
    suite = Suite("SDXI DMA tests")
    server = Server(binary)
    server.start()
    try:
        with GuestMemory(size=128 << 20, name="sdxi-guest-mem") as mem:
            with server.connect() as vfu:
                vfu.dma_map(mem)
                dev = _mkdev(vfu)
                suite.run("memcpy", test_memcpy, dev, mem, 0)
                suite.run("memset", test_memset, dev, mem, 1)
                suite.run("fence", test_fence, dev, mem, 2)
                suite.run("bad_opcode", test_bad_opcode, dev, mem, 3)
                suite.run("multi_command", test_multi_command, dev, mem, 4)
                suite.run("zero_length", test_zero_length, dev, mem, 5)
                suite.run("boundary_crossing", test_boundary_crossing, dev, mem, 6)
                suite.run("multi_queue_sim", test_multi_queue_sim, dev, mem, 7)
    finally:
        stop_and_check(suite, server)
    suite.finish()


if __name__ == "__main__":
    main()
