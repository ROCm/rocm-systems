"""v3 context-table driver for the SDXI vfio-user emulator.

Implements the two-level context-table access path used by the upstream v3
Linux kernel driver: writes CXT_L2, allocates L1/L2 tables in guest memory,
issues descriptors via the descriptor-ring path, and polls CSBs.

Copyright (C) Advanced Micro Devices, Inc.

SPDX-License-Identifier: MIT
"""

import struct
import time
import zlib

from sdxi.vfu import VFU_PCI_DEV_BAR0_REGION_IDX, VFU_PCI_DEV_BAR2_REGION_IDX

# COMPUTE descriptor type and ELEMWISE subtype
TYPE_COMPUTE       = 0x008
SUBTYPE_ELEMWISE   = 0x01

# Element data type constants (SDXI_DTYPE_*)
DTYPE_FP32      = 0
DTYPE_FP16      = 1
DTYPE_BF16      = 2
DTYPE_INT8      = 3
DTYPE_INT32     = 4
DTYPE_FP8_E4M3  = 5
DTYPE_FP8_E5M2  = 6
DTYPE_INT4      = 7

# Element operation constants (SDXI_ELEMOP_*)
ELEMOP_ADD         = 0
ELEMOP_MIN         = 1
ELEMOP_MAX         = 2
ELEMOP_REDUCE_MIN  = 3
ELEMOP_REDUCE_MAX  = 4

ELEM_SIZE = {
    DTYPE_FP32: 4, DTYPE_INT32: 4,
    DTYPE_FP16: 2, DTYPE_BF16:  2,
    DTYPE_INT8: 1, DTYPE_FP8_E4M3: 1, DTYPE_FP8_E5M2: 1,
    DTYPE_INT4: 1,  # 2 elements per byte; dst_elem_sz used for reduce
}

# BAR0 v3 register offsets (64-bit readq/writeq)
MMIO_CTL0    = 0x000
MMIO_STS0    = 0x100
MMIO_CAP0    = 0x200
MMIO_CAP1    = 0x208
MMIO_VERSION = 0x210
MMIO_CXT_L2  = 0x10000

# Vendor-specific counter registers
MMIO_CTR_SELECT = 0x300
MMIO_CTR_VAL    = 0x308

# Counter stat selectors — must match SDXI_CTR_STAT_* in sdxi_regs.h
CTR_BYTES_MOVED        = 0x00
CTR_NUM_COPY           = 0x01
CTR_NUM_ERRORS         = 0x02
CTR_NUM_INTR           = 0x03
CTR_NUM_CXT_START      = 0x04
CTR_NUM_CXT_STOP       = 0x05
CTR_NUM_NOP            = 0x06
CTR_NUM_SYNC           = 0x07
CTR_BYTES_COMPRESSED   = 0x08
CTR_BYTES_DECOMPRESSED = 0x09
CTR_NUM_COMPRESS       = 0x0A
CTR_NUM_DECOMPRESS     = 0x0B

# Context-table geometry — must match sdxi_engine.c / sdxi_regs.h exactly
#
# Three levels:
#   OUTER  — written to MMIO_CXT_L2; 8 bytes/entry; entry[cid/128] → INNER table
#   INNER  — 128 × 32 bytes/entry; entry[cid%128] → CTX_CTL block
#   CTX_CTL — 64 bytes; offset 0: ds_ring_ptr (VL bit 0); offset 8: ds_ring_sz
#
# Every pointer field has bit 0 = VL (valid) that the emulator checks before dereferencing.
L1_CTX_PER_GROUP = 128     # contexts per OUTER entry / INNER table
OUTER_ENTRY_SZ   = 8       # bytes
INNER_ENTRY_SZ   = 32      # bytes (SDXI_CXT_L1_ENTRY_SIZE)
CTX_CTL_SZ       = 64      # bytes
CTX_CTL_RING_PTR_OFF = 0   # LE64 within ctx_ctl: ring GPA | VL
CTX_CTL_RING_SZ_OFF  = 8   # LE64 within ctx_ctl: ring capacity (slots)

# Descriptor constants
DSC_SIZE = 64

# Descriptor opcode word: bit0=VL, bits15:8=SUBTYPE, bits26:16=TYPE
TYPE_DMAB  = 0x001
TYPE_ADMIN = 0x002

SUBTYPE_NOP       = 0x01
SUBTYPE_COPY      = 0x03
SUBTYPE_CXT_START = 0x03
SUBTYPE_CXT_STOP  = 0x04
SUBTYPE_COMPRESS  = 0x04
SUBTYPE_DECOMPRESS = 0x05

# Doorbell stride (matches SDXI_DB_STRIDE)
DB_STRIDE = 4096

# Admin context ID is always 0
ADMIN_CXT_ID = 0

# CSB (Completion Status Block) geometry
CSB_SIZE       = 32
CSB_SIGNAL_OFF     = 0    # LE64: written to 0 by hardware on completion
CSB_FLAGS_OFF      = 8    # LE32: bit 31 = error (SDXI_CST_BLK_ER_BIT)
CSB_RESULT_LEN_OFF = 12   # LE64: actual output size (compress/decompress)
CSB_ER_BIT = 1 << 31

POLL_TIMEOUT  = 5.0
POLL_INTERVAL = 0.001


def _make_opcode(typ, subtype):
    """Build the LE32 opcode word for a descriptor (VL=1)."""
    return 0x1 | (subtype << 8) | (typ << 16)


def _pack_copy_dsc(src_gpa, dst_gpa, length, csb_gpa):
    """64-byte DMAB COPY descriptor."""
    opcode  = _make_opcode(TYPE_DMAB, SUBTYPE_COPY)
    size_m1 = max(0, length - 1)
    dsc = bytearray(DSC_SIZE)
    struct.pack_into("<I", dsc, 0,  opcode)
    struct.pack_into("<I", dsc, 4,  size_m1)
    struct.pack_into("<Q", dsc, 16, src_gpa)
    struct.pack_into("<Q", dsc, 24, dst_gpa)
    struct.pack_into("<Q", dsc, 56, csb_gpa)   # emulator masks low 5 bits (NP flag)
    return bytes(dsc)


def _pack_compress_dsc(src_gpa, src_len, dst_gpa, dst_cap, level, csb_gpa):
    """64-byte DMAB COMPRESS descriptor."""
    opcode = _make_opcode(TYPE_DMAB, SUBTYPE_COMPRESS)
    dsc = bytearray(DSC_SIZE)
    struct.pack_into("<I", dsc, 0,  opcode)
    struct.pack_into("<I", dsc, 4,  src_len)
    struct.pack_into("<I", dsc, 8,  dst_cap)
    struct.pack_into("<I", dsc, 12, level)
    struct.pack_into("<Q", dsc, 16, src_gpa)
    struct.pack_into("<Q", dsc, 24, dst_gpa)
    struct.pack_into("<Q", dsc, 56, csb_gpa)
    return bytes(dsc)


def _pack_decompress_dsc(src_gpa, src_len, dst_gpa, dst_cap, csb_gpa):
    """64-byte DMAB DECOMPRESS descriptor."""
    opcode = _make_opcode(TYPE_DMAB, SUBTYPE_DECOMPRESS)
    dsc = bytearray(DSC_SIZE)
    struct.pack_into("<I", dsc, 0,  opcode)
    struct.pack_into("<I", dsc, 4,  src_len)
    struct.pack_into("<I", dsc, 8,  dst_cap)
    struct.pack_into("<Q", dsc, 16, src_gpa)
    struct.pack_into("<Q", dsc, 24, dst_gpa)
    struct.pack_into("<Q", dsc, 56, csb_gpa)
    return bytes(dsc)


def _pack_elemwise_dsc(src_gpa, src_len, dst_gpa, dtype, op, scalar_bits, csb_gpa):
    """64-byte COMPUTE ELEMWISE descriptor."""
    opcode = _make_opcode(TYPE_COMPUTE, SUBTYPE_ELEMWISE)
    dsc = bytearray(DSC_SIZE)
    struct.pack_into("<I", dsc,  0, opcode)
    struct.pack_into("<I", dsc,  4, src_len)
    struct.pack_into("<I", dsc,  8, dtype)
    struct.pack_into("<I", dsc, 12, op)
    struct.pack_into("<Q", dsc, 16, src_gpa)
    struct.pack_into("<Q", dsc, 24, dst_gpa)
    struct.pack_into("<Q", dsc, 32, scalar_bits & 0xFFFFFFFFFFFFFFFF)
    struct.pack_into("<Q", dsc, 56, csb_gpa)
    return bytes(dsc)


def _pack_cxt_start_dsc(target_cid):
    """64-byte ADMIN CXT_START descriptor."""
    opcode = _make_opcode(TYPE_ADMIN, SUBTYPE_CXT_START)
    dsc = bytearray(DSC_SIZE)
    struct.pack_into("<I", dsc, 0, opcode)
    struct.pack_into("<I", dsc, 4, target_cid)
    return bytes(dsc)


class CxtContext:
    """One SDXI context — a descriptor ring plus helper methods."""

    def __init__(self, cid, ring_gpa, ring_sz, csb_gpa, mem):
        self.cid = cid
        self.ring_gpa = ring_gpa
        self.ring_sz  = ring_sz
        self.csb_gpa  = csb_gpa
        self.mem      = mem
        self._write_idx = 0

    def _ring_slot_gpa(self, idx):
        return self.ring_gpa + (idx % self.ring_sz) * DSC_SIZE

    def _write_dsc(self, dsc_bytes):
        gpa = self._ring_slot_gpa(self._write_idx)
        self.mem.write(gpa, dsc_bytes)
        self._write_idx += 1
        return self._write_idx

    def _clear_csb(self):
        self.mem.write(self.csb_gpa, b"\xff" * CSB_SIZE)

    def _poll_csb(self):
        deadline = time.monotonic() + POLL_TIMEOUT
        while time.monotonic() < deadline:
            raw = self.mem.read(self.csb_gpa, CSB_SIZE)
            signal = struct.unpack_from("<I", raw, CSB_SIGNAL_OFF)[0]
            if signal != 0xFFFFFFFF:
                flags = struct.unpack_from("<I", raw, CSB_FLAGS_OFF)[0]
                result_len = struct.unpack_from("<Q", raw, CSB_RESULT_LEN_OFF)[0]
                return flags, result_len
            time.sleep(POLL_INTERVAL)
        raise TimeoutError(f"cxt {self.cid}: CSB not written within {POLL_TIMEOUT}s")

    def copy(self, vfu, src_gpa, dst_gpa, length):
        """Submit a DMAB COPY and wait for CSB."""
        self._clear_csb()
        new_wi = self._write_dsc(_pack_copy_dsc(src_gpa, dst_gpa, length, self.csb_gpa))
        db = self.cid * DB_STRIDE
        vfu.write_bar64(VFU_PCI_DEV_BAR2_REGION_IDX, db, new_wi)
        flags, _ = self._poll_csb()
        return (flags & CSB_ER_BIT) == 0

    def compress(self, vfu, src_gpa, src_len, dst_gpa, dst_cap, level=0):
        """Submit a DMAB COMPRESS and return (ok, actual_output_len)."""
        self._clear_csb()
        new_wi = self._write_dsc(
            _pack_compress_dsc(src_gpa, src_len, dst_gpa, dst_cap, level, self.csb_gpa)
        )
        db = self.cid * DB_STRIDE
        vfu.write_bar64(VFU_PCI_DEV_BAR2_REGION_IDX, db, new_wi)
        flags, result_len = self._poll_csb()
        ok = (flags & CSB_ER_BIT) == 0
        return ok, result_len

    def elemwise(self, vfu, src_gpa, src_len, dst_gpa, dtype, op, scalar_bits=0):
        """Submit a COMPUTE ELEMWISE descriptor and return (ok, out_len).

        scalar_bits is the dtype-native bit pattern as an integer (ignored for
        reduce operations but must still be supplied).  dst_gpa must be large
        enough: src_len bytes for elementwise ops, one elem for reduce ops.
        """
        self._clear_csb()
        new_wi = self._write_dsc(
            _pack_elemwise_dsc(src_gpa, src_len, dst_gpa, dtype, op, scalar_bits,
                               self.csb_gpa)
        )
        db = self.cid * DB_STRIDE
        vfu.write_bar64(VFU_PCI_DEV_BAR2_REGION_IDX, db, new_wi)
        flags, result_len = self._poll_csb()
        ok = (flags & CSB_ER_BIT) == 0
        return ok, result_len

    def decompress(self, vfu, src_gpa, src_len, dst_gpa, dst_cap):
        """Submit a DMAB DECOMPRESS and return (ok, actual_output_len)."""
        self._clear_csb()
        new_wi = self._write_dsc(
            _pack_decompress_dsc(src_gpa, src_len, dst_gpa, dst_cap, self.csb_gpa)
        )
        db = self.cid * DB_STRIDE
        vfu.write_bar64(VFU_PCI_DEV_BAR2_REGION_IDX, db, new_wi)
        flags, result_len = self._poll_csb()
        ok = (flags & CSB_ER_BIT) == 0
        return ok, result_len


class CxtDevice:
    """SDXI device in v3 context-table mode."""

    def __init__(self, vfu, mem, num_cxt=4, ring_sz=16):
        self.vfu     = vfu
        self.mem     = mem
        self.num_cxt = num_cxt
        self.ring_sz = ring_sz
        self.contexts = []
        self._setup()

    def _write64(self, offset, val):
        self.vfu.write_bar64(VFU_PCI_DEV_BAR0_REGION_IDX, offset, val)

    def _read64(self, offset):
        return self.vfu.read_bar64(VFU_PCI_DEV_BAR0_REGION_IDX, offset)

    def _setup(self):
        """Allocate tables and descriptor rings; wire CXT_L2.

        Three-level structure expected by sdxi_engine_process_cxt:
          OUTER[cid/128] → INNER table (128 × 32 B, one per 128-context group)
          INNER[cid%128] → CTX_CTL block (64 B)
          CTX_CTL        → descriptor ring
        Every pointer field uses bit 0 as VL (valid); the emulator checks it.
        """
        mem = self.mem

        # OUTER table: 8 bytes/slot; one slot per 128-context group.
        # With num_cxt ≤ 128 we only use slot 0.
        outer_gpa = mem.alloc(4096)   # page-aligned (SDXI_CXT_L2_PTR_MASK strips low 12)

        # INNER table: 128 × 32 bytes, page-aligned.
        inner_gpa = mem.alloc(L1_CTX_PER_GROUP * INNER_ENTRY_SZ)

        # OUTER[0] → INNER table, VL=1
        mem.write_u64(outer_gpa, inner_gpa | 1)

        # Allocate rings, CSBs, and ctx_ctl blocks for each context.
        for cid in range(self.num_cxt):
            ring_gpa    = mem.alloc(self.ring_sz * DSC_SIZE)   # page-aligned ≥ 64-B
            csb_gpa     = mem.alloc(CSB_SIZE)
            cxt_ctl_gpa = mem.alloc(CTX_CTL_SZ)

            mem.write_u64(cxt_ctl_gpa + CTX_CTL_RING_PTR_OFF, ring_gpa | 1)   # VL=1
            mem.write_u64(cxt_ctl_gpa + CTX_CTL_RING_SZ_OFF,  self.ring_sz)

            # INNER[cid] → cxt_ctl, VL=1
            mem.write_u64(inner_gpa + (cid % L1_CTX_PER_GROUP) * INNER_ENTRY_SZ,
                          cxt_ctl_gpa | 1)

            self.contexts.append(CxtContext(cid, ring_gpa, self.ring_sz, csb_gpa, mem))

        # Point the device at the OUTER table.
        self._write64(MMIO_CXT_L2, outer_gpa)

        # Activate non-admin contexts via admin-context CXT_START descriptors.
        admin = self.contexts[ADMIN_CXT_ID]
        for cid in range(1, self.num_cxt):
            admin.mem.write(admin._ring_slot_gpa(admin._write_idx),
                            _pack_cxt_start_dsc(cid))
            admin._write_idx += 1
        if admin._write_idx > 0:
            self.vfu.write_bar64(VFU_PCI_DEV_BAR2_REGION_IDX,
                                 ADMIN_CXT_ID * DB_STRIDE, admin._write_idx)
            time.sleep(0.05)

    def cxt(self, cid):
        return self.contexts[cid]

    def read_counter(self, cid, stat_sel):
        """Read one performance counter via CTR_SELECT / CTR_VAL."""
        sel = (stat_sel << 8) | cid
        self._write64(MMIO_CTR_SELECT, sel)
        return self._read64(MMIO_CTR_VAL)
