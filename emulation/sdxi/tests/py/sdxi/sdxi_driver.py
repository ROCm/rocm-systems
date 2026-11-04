"""High-level driver for the SDXI emulator.

Wraps queue setup, SQE submission, doorbell writes, and CQE polling
so that test scripts work at the level of "submit a MEMCPY and wait"
rather than raw register reads and writes.

Copyright (C) Advanced Micro Devices, Inc.

SPDX-License-Identifier: MIT
"""

import struct
import time

from sdxi.vfu import VFU_PCI_DEV_BAR0_REGION_IDX, VFU_PCI_DEV_BAR2_REGION_IDX

# BAR0 register offsets
REG_DEVID = 0x000
REG_VERSION = 0x004
REG_CAP = 0x008
REG_NUM_QUEUES = 0x00C
REG_STATUS = 0x010
REG_CTRL = 0x014
REG_Q_SELECT = 0x020
REG_Q_CTRL = 0x024
REG_Q_STATUS = 0x028
REG_Q_SIZE = 0x02C
REG_Q_HEAD = 0x030
REG_Q_TAIL = 0x034
REG_Q_BASE_LO = 0x038
REG_Q_BASE_HI = 0x03C
REG_CQ_SIZE = 0x040
REG_CQ_HEAD = 0x044
REG_CQ_TAIL = 0x048
REG_CQ_BASE_LO = 0x04C
REG_CQ_BASE_HI = 0x050
REG_Q_INTR_VEC = 0x054

# Device constants
DEVID_MAGIC = 0x53445849  # "SDXI"
VERSION_VAL = 0x0100

STATUS_READY = 1 << 0
STATUS_ERROR = 1 << 1

CTRL_ENABLE = 1 << 0
CTRL_RESET = 1 << 1

QCTRL_ENABLE = 1 << 0
QCTRL_RESET = 1 << 1
QCTRL_INTR_EN = 1 << 2

CAP_MEMCPY = 1 << 0
CAP_MEMSET = 1 << 1
CAP_FENCE = 1 << 2

# BAR2 doorbell
DB_STRIDE = 4096

# Descriptor geometry
SQE_SIZE = 64
CQE_SIZE = 16

# Max queues per device
NUM_QUEUES = 16

# Opcodes
OP_MEMCPY = 0x01
OP_MEMSET = 0x02
OP_FENCE  = 0x03

# CQE status values (from sdxi_regs.h CQE status field)
CQE_STATUS_OK = 0
CQE_STATUS_INVAL = 1


def pack_sqe(opcode, flags, sq_id, cid, src_addr, dst_addr, length):
    """Pack one 64-byte SQE."""
    return struct.pack(
        "<IIIIQQQ24x",
        opcode,
        flags,
        sq_id,
        cid,
        src_addr,
        dst_addr,
        length,
    )


def unpack_cqe(data):
    """Unpack one 16-byte CQE into (status, cid, sq_head, sq_id, phase)."""
    status, cid, sq_head, sq_id, phase = struct.unpack_from("<IIHHBxxx", data)
    return status, cid, sq_head, sq_id, phase


class Queue:
    """One SQ/CQ pair configured on the device."""

    _POLL_TIMEOUT = 5.0
    _POLL_INTERVAL = 0.001

    def __init__(self, vfu, mem, qid, sq_size=16, cq_size=16):
        self.vfu = vfu
        self.mem = mem
        self.qid = qid
        self.sq_size = sq_size
        self.cq_size = cq_size

        self.sq_gpa = mem.alloc(sq_size * SQE_SIZE)
        self.cq_gpa = mem.alloc(cq_size * CQE_SIZE)

        self._sq_tail = 0
        self._cq_head = 0

    def _write_bar0(self, offset, value):
        self.vfu.write_bar32(VFU_PCI_DEV_BAR0_REGION_IDX, offset, value)

    def _read_bar0(self, offset):
        return self.vfu.read_bar32(VFU_PCI_DEV_BAR0_REGION_IDX, offset)

    def configure(self):
        """Program queue registers and enable the queue."""
        self._write_bar0(REG_Q_SELECT, self.qid)
        self._write_bar0(REG_Q_SIZE, self.sq_size)
        self._write_bar0(REG_Q_BASE_LO, self.sq_gpa & 0xFFFF_FFFF)
        self._write_bar0(REG_Q_BASE_HI, (self.sq_gpa >> 32) & 0xFFFF_FFFF)
        self._write_bar0(REG_CQ_SIZE, self.cq_size)
        self._write_bar0(REG_CQ_BASE_LO, self.cq_gpa & 0xFFFF_FFFF)
        self._write_bar0(REG_CQ_BASE_HI, (self.cq_gpa >> 32) & 0xFFFF_FFFF)
        self._write_bar0(REG_Q_CTRL, QCTRL_ENABLE)

    def submit(self, sqe_bytes):
        """Write @sqe_bytes into the SQ ring and ring the doorbell."""
        slot = self._sq_tail % self.sq_size
        gpa = self.sq_gpa + slot * SQE_SIZE
        # Host offset within memfd
        offset = gpa - self.mem.base
        self.mem.map[offset : offset + SQE_SIZE] = sqe_bytes
        self._sq_tail += 1
        db_offset = self.qid * DB_STRIDE
        self.vfu.write_bar32(VFU_PCI_DEV_BAR2_REGION_IDX, db_offset, self._sq_tail)

    def wait_cq(self):
        """Block until the next CQE is posted; return (status, cid, sq_head)."""
        cq_head_mod = self._cq_head % self.cq_size
        deadline = time.monotonic() + self._POLL_TIMEOUT
        while time.monotonic() < deadline:
            self._write_bar0(REG_Q_SELECT, self.qid)
            cq_tail = self._read_bar0(REG_CQ_TAIL)
            if cq_tail != cq_head_mod:
                break
            time.sleep(self._POLL_INTERVAL)
        else:
            raise TimeoutError(
                f"queue {self.qid}: CQE not posted within {self._POLL_TIMEOUT}s"
            )
        slot = cq_head_mod
        gpa = self.cq_gpa + slot * CQE_SIZE
        offset = gpa - self.mem.base
        cqe_data = bytes(self.mem.map[offset : offset + CQE_SIZE])
        status, cid, sq_head, sq_id, _phase = unpack_cqe(cqe_data)
        self._cq_head += 1
        new_head_mod = self._cq_head % self.cq_size
        self._write_bar0(REG_Q_SELECT, self.qid)
        self._write_bar0(REG_CQ_HEAD, new_head_mod)
        return status, cid, sq_head


class SdxiDevice:
    """Convenience wrapper around a VfioUserClient for SDXI register access."""

    def __init__(self, vfu):
        self.vfu = vfu

    def read_bar0(self, offset):
        return self.vfu.read_bar32(VFU_PCI_DEV_BAR0_REGION_IDX, offset)

    def write_bar0(self, offset, value):
        self.vfu.write_bar32(VFU_PCI_DEV_BAR0_REGION_IDX, offset, value)

    def queue(self, mem, qid, sq_size=16, cq_size=16):
        """Allocate and configure one queue; return a Queue object."""
        q = Queue(self.vfu, mem, qid, sq_size, cq_size)
        q.configure()
        return q
