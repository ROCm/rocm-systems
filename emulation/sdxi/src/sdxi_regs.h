/* Copyright (c) Advanced Micro Devices, Inc. All rights reserved.
 *
 * SPDX-License-Identifier: MIT
 *
 * SDXI (SNIA Data Exchange Interface) emulator -- PCI IDs, BAR layout,
 * register map, and descriptor/completion formats.
 *
 * Hardware intent: a PCIe copy-engine / data-mover peripheral for the
 * AMD CPU SoC.  The register map follows the SNIA SDXI specification
 * draft and the AMD SDMA doorbell conventions documented in
 * rocm-systems-sdma-mmio-2026-10-09.md.
 */

#ifndef SDXI_REGS_H
#define SDXI_REGS_H

#include <stdint.h>

/* ------------------------------------------------------------------ */
/* PCI identity                                                        */
/* ------------------------------------------------------------------ */

#define SDXI_PCI_VENDOR_ID     0x1022U   /* AMD */
#define SDXI_PCI_DEVICE_ID     0x149fU   /* Provisional SDXI device id */
#define SDXI_PCI_SUBVENDOR_ID  0x1022U
#define SDXI_PCI_SUBDEVICE_ID  0x0001U

/* PCI class: Processing Accelerator (0x12), SDXI sub-class (0x01), prog-if 0x00
 * Matches PCI_CLASS_ACCELERATOR_SDXI=0x120100 added by v3 kernel driver patch 01/23. */
#define SDXI_PCI_CLASS_BASE    0x12U
#define SDXI_PCI_CLASS_SUB     0x01U
#define SDXI_PCI_CLASS_PROGIF  0x00U

/* ------------------------------------------------------------------ */
/* BAR layout                                                          */
/* ------------------------------------------------------------------ */

/*
 * BAR0: 128 KiB control/status register space.  128 KiB is required because
 * the v3 driver's CXT_L2 context-table pointer register sits at offset 0x10000,
 * which falls just past a 64 KiB BAR.
 *   - [0, SDXI_BAR0_REGS_SIZE) : MMIO registers
 *   - [SDXI_MSIX_TABLE_OFFSET, ...) : MSI-X table
 *   - [SDXI_MSIX_PBA_OFFSET,   ...) : MSI-X PBA
 *   - [0x10000]                : CXT_L2 context-table pointer (v3 driver)
 *
 * BAR2: 4 MiB doorbell aperture.
 *   Guest writes the new SQ tail pointer to offset (queue_id * 4096)
 *   within BAR2 to ring the doorbell for that queue.
 */
#define SDXI_BAR0_SIZE         (128U * 1024U)
#define SDXI_BAR0_REGS_SIZE    (32U * 1024U)   /* registers occupy lower 32 KiB */
#define SDXI_BAR2_SIZE         (4U * 1024U * 1024U)

#define SDXI_BAR0_IDX          0   /* VFU_PCI_DEV_BAR0_REGION_IDX */
#define SDXI_BAR2_IDX          2   /* VFU_PCI_DEV_BAR2_REGION_IDX */

/* MSI-X lives in upper BAR0, using BAR0 as the table BIR */
#define SDXI_MSIX_TABLE_BIR    0
#define SDXI_MSIX_TABLE_OFFSET SDXI_BAR0_REGS_SIZE
#define SDXI_MSIX_PBA_OFFSET   (SDXI_MSIX_TABLE_OFFSET + 0x1000U)
#define SDXI_MSIX_NUM_VECTORS  16U   /* one per queue + a few spare */

/* ------------------------------------------------------------------ */
/* v3 kernel driver MMIO register offsets (BAR0, 64-bit accesses)     */
/* ------------------------------------------------------------------ */

/* Register offsets — from v3 patch 04/23 (mmio.h). */
#define SDXI_MMIO_CTL0         0x000U   /* control (RW) */
#define SDXI_MMIO_CTL2         0x010U   /* config (RW, written by driver at probe) */
#define SDXI_MMIO_STS0         0x100U   /* status (RO) */
#define SDXI_MMIO_CAP0         0x200U   /* capability word 0 (RO) */
#define SDXI_MMIO_CAP1         0x208U   /* capability word 1 (RO) */
#define SDXI_MMIO_VERSION      0x210U   /* version (RO) */

/* CTL0 field: FN_GSR bits[1:0] — function global state request.
 * The driver writes here to request stop/start; the emulator mirrors
 * this immediately into STS0 FN_GSV bits[2:0] to satisfy the poll. */
#define SDXI_MMIO_CTL0_FN_GSR_MASK  0x3ULL
#define SDXI_MMIO_STS0_FN_GSV_MASK  0x7ULL

/* CAP0 static value: db_stride bits[22:20] = 0 → 4 KiB stride.
 * Matches SDXI_DB_STRIDE = 4096 (driver computes SZ_4K << field). */
#define SDXI_MMIO_CAP0_VAL     0x0000000000000000ULL

/* CAP1 static value: MAX_CXT bits[31:16] = 128 (SDXI_MAX_CXT contexts).
 * All other fields (MAX_BUFFER, MAX_AKEY_SZ, OPB_000_CAP) left 0. */
#define SDXI_MMIO_CAP1_VAL     0x0000000000800000ULL

/* VERSION: 1.0 */
#define SDXI_MMIO_VERSION_VAL  0x0000000000000100ULL

/* Vendor-specific performance counter registers (BAR0, 64-bit accesses).
 *
 * CTR_SELECT layout:
 *   bits [6:0]  : context ID (0..SDXI_MAX_CXT-1)
 *   bits [15:8] : stat selector (SDXI_CTR_STAT_*)
 *
 * Write CTR_SELECT then read CTR_VAL to get the selected counter.
 */
#define SDXI_MMIO_CTR_SELECT   0x300U
#define SDXI_MMIO_CTR_VAL      0x308U   /* read selected stat (64-bit) */

#define SDXI_CTR_CXT_MASK      0x7FU          /* bits [6:0] */
#define SDXI_CTR_STAT_SHIFT    8U
#define SDXI_CTR_STAT_MASK     (0xFFU << SDXI_CTR_STAT_SHIFT)

/* Stat selectors written into CTR_SELECT bits[15:8]. */
#define SDXI_CTR_STAT_BYTES_MOVED        0x00U  /* cumulative COPY bytes */
#define SDXI_CTR_STAT_NUM_COPY           0x01U
#define SDXI_CTR_STAT_NUM_ERRORS         0x02U
#define SDXI_CTR_STAT_NUM_INTR           0x03U
#define SDXI_CTR_STAT_NUM_CXT_START      0x04U
#define SDXI_CTR_STAT_NUM_CXT_STOP       0x05U
#define SDXI_CTR_STAT_NUM_NOP            0x06U
#define SDXI_CTR_STAT_NUM_SYNC           0x07U
#define SDXI_CTR_STAT_BYTES_COMPRESSED   0x08U  /* cumulative COMPRESS input bytes */
#define SDXI_CTR_STAT_BYTES_DECOMPRESSED 0x09U  /* cumulative DECOMPRESS output bytes */
#define SDXI_CTR_STAT_NUM_COMPRESS       0x0AU
#define SDXI_CTR_STAT_NUM_DECOMPRESS     0x0BU

/* Context-table L2 pointer register at BAR0 offset 0x10000.
 * Driver writes FIELD_PREP(GENMASK_ULL(63,12), l2_dma >> 12); emulator recovers
 * l2_gpa as (reg_value & ~0xFFFULL). */
#define SDXI_MMIO_CXT_L2       0x10000U

/* FN_GSR values written to CTL0 bits[1:0] by the driver. */
#define SDXI_GSR_RESET         0U   /* request stop/reset */
#define SDXI_GSR_STOP_SF       1U   /* stop, software-fault */
#define SDXI_GSR_STOP_HD       2U   /* stop, hardware-disabled */
#define SDXI_GSR_ACTIVE        3U   /* request active */

/* FN_GSV values reported in STS0 bits[2:0] by the hardware. */
#define SDXI_GSV_STOP          0U
#define SDXI_GSV_INIT          1U
#define SDXI_GSV_ACTIVE        2U
#define SDXI_GSV_STOPG_SF      3U
#define SDXI_GSV_STOPG_HD      4U
#define SDXI_GSV_ERROR         5U

/* v3 descriptor opcode field (32-bit LE at offset 0).
 * bit 0 : VL (valid)
 * bits[15:8]  : SUBTYPE
 * bits[26:16] : TYPE */
#define SDXI_DSC_VL            (1U << 0)
#define SDXI_DSC_SUBTYPE_MASK  0x0000FF00U
#define SDXI_DSC_SUBTYPE_SHIFT 8U
#define SDXI_DSC_TYPE_MASK     0x07FF0000U
#define SDXI_DSC_TYPE_SHIFT    16U
#define SDXI_DSC_GET_TYPE(op)    (((op) & SDXI_DSC_TYPE_MASK)    >> SDXI_DSC_TYPE_SHIFT)
#define SDXI_DSC_GET_SUBTYPE(op) (((op) & SDXI_DSC_SUBTYPE_MASK) >> SDXI_DSC_SUBTYPE_SHIFT)

/* Descriptor operation types. */
#define SDXI_DSC_OP_TYPE_DMAB  0x001U
#define SDXI_DSC_OP_TYPE_ADMIN 0x002U
#define SDXI_DSC_OP_TYPE_INTR  0x004U

/* Descriptor operation subtypes. */
#define SDXI_DSC_OP_SUBTYPE_NOP        0x01U
#define SDXI_DSC_OP_SUBTYPE_SYNC       0x02U   /* DMAB fence / ordering barrier */
#define SDXI_DSC_OP_SUBTYPE_COPY       0x03U
#define SDXI_DSC_OP_SUBTYPE_CXT_START  0x03U
#define SDXI_DSC_OP_SUBTYPE_CXT_STOP   0x04U
/* Vendor-specific DMAB subtypes for in-line compression. */
#define SDXI_DSC_OP_SUBTYPE_COMPRESS   0x04U   /* deflate src → dst */
#define SDXI_DSC_OP_SUBTYPE_DECOMPRESS 0x05U   /* inflate src → dst */

/* INTR descriptor: MSI-X vector at byte offset 4 (uint16_t). */
#define SDXI_INTR_VEC_OFF  4U

/* csb_ptr field in the descriptor (64-bit LE at offset 56).
 * bit 0 : NP (new pointer)
 * bits[63:5] : CSB GPA >> 5 */
#define SDXI_DSC_CSBPTR_SHIFT  5U

/* DMAB COPY descriptor field byte offsets within the 64-byte descriptor. */
#define SDXI_COPY_SIZE_OFF     4U    /* LE32: copy length minus 1 */
#define SDXI_COPY_ADDR0_OFF    16U   /* LE64: source GPA */
#define SDXI_COPY_ADDR1_OFF    24U   /* LE64: destination GPA */
#define SDXI_COPY_CSBPTR_OFF   56U   /* LE64: NP | (csb_gpa >> 5) in bits[63:5] */

/* DMAB COMPRESS / DECOMPRESS descriptor field offsets.
 * Both ops share this layout; for DECOMPRESS src_len is the compressed size
 * and dst_cap is the decompressed buffer capacity. */
#define SDXI_COMP_SRC_LEN_OFF  4U    /* LE32: source byte count */
#define SDXI_COMP_DST_CAP_OFF  8U    /* LE32: destination buffer capacity */
#define SDXI_COMP_FLAGS_OFF    12U   /* LE32: compression level (1-9) or 0=default */
#define SDXI_COMP_SRC_ADDR_OFF 16U   /* LE64: source GPA */
#define SDXI_COMP_DST_ADDR_OFF 24U   /* LE64: destination GPA */
#define SDXI_COMP_CSBPTR_OFF   56U   /* LE64: NP | (csb_gpa >> 5) */

/* CSB result_len field offset within sdxi_cst_blk for compress/decompress. */
#define SDXI_CSB_RESULT_LEN_OFF 12U  /* LE64 at bytes [12..19]: actual output size */

/* CSB (Completion Status Block) -- 32 bytes.
 * Hardware clears signal to 0 on completion; driver polls until signal == 0. */
struct sdxi_cst_blk {
    uint64_t signal;    /* cleared to 0 on success (driver initialises to 1) */
    uint32_t flags;     /* bit 31: error */
    uint8_t  rsvd[20];
} __attribute__((packed));

#define SDXI_CST_BLK_SIZE   32U
#define SDXI_CST_BLK_ER_BIT (1U << 31)
_Static_assert(sizeof(struct sdxi_cst_blk) == SDXI_CST_BLK_SIZE,
               "sdxi_cst_blk must be 32 bytes");

/* Context-table constants. */
#define SDXI_L1_TABLE_ENTRIES  128U   /* L1 entries per L2 slot */
#define SDXI_MAX_CXT           128U   /* max context IDs (we cap at L1 for simplicity) */
#define SDXI_ADMIN_CXT_ID      0U

/* Pointer masks for context-table entries (low bits are flags). */
#define SDXI_CXT_L2_PTR_MASK        0xFFFFFFFFFFFFF000ULL  /* bits 63:12 */
#define SDXI_CXT_L2_ENT_VL         (1ULL)
#define SDXI_CXT_L1_PTR_MASK        0xFFFFFFFFFFFFFFC0ULL  /* bits 63:6 */
#define SDXI_CXT_L1_ENT_VL         (1ULL)
#define SDXI_CXT_CTL_DS_RING_PTR_MASK 0xFFFFFFFFFFFFFFC0ULL  /* bits 63:6 */
#define SDXI_CXT_CTL_DS_RING_VL    (1ULL)

/* L1 entry is 32 bytes; cxt_ctl_ptr is the first 8-byte field. */
#define SDXI_CXT_L1_ENTRY_SIZE      32U

/* ------------------------------------------------------------------ */
/* BAR0 register offsets                                               */
/* ------------------------------------------------------------------ */

/* Device identity / capability (read-only) */
#define SDXI_REG_DEVID         0x000U   /* 0x53445849 == 'SDXI' */
#define SDXI_REG_VERSION       0x004U   /* bits[15:8]=major, [7:0]=minor */
#define SDXI_REG_CAP           0x008U   /* capability flags */
#define SDXI_REG_NUM_QUEUES    0x00CU   /* total queues supported (R) */

/* Device control / status */
#define SDXI_REG_STATUS        0x010U   /* device status (R) */
#define SDXI_REG_CTRL          0x014U   /* device control (RW) */

/* Queue management -- indirect access via Q_SELECT */
#define SDXI_REG_Q_SELECT      0x020U   /* select active queue 0..N-1 (RW) */
#define SDXI_REG_Q_CTRL        0x024U   /* selected queue control (RW) */
#define SDXI_REG_Q_STATUS      0x028U   /* selected queue status (R) */
#define SDXI_REG_Q_SIZE        0x02CU   /* selected SQ depth in entries (RW) */
#define SDXI_REG_Q_HEAD        0x030U   /* selected SQ head (device advances, R) */
#define SDXI_REG_Q_TAIL        0x034U   /* selected SQ tail (sw writes doorbell) */
#define SDXI_REG_Q_BASE_LO     0x038U   /* SQ GPA[31:0] (RW, set before enable) */
#define SDXI_REG_Q_BASE_HI     0x03CU   /* SQ GPA[63:32] */

#define SDXI_REG_CQ_SIZE       0x040U   /* selected CQ depth in entries (RW) */
#define SDXI_REG_CQ_HEAD       0x044U   /* CQ head (sw advances after consuming, W) */
#define SDXI_REG_CQ_TAIL       0x048U   /* CQ tail (device advances, R) */
#define SDXI_REG_CQ_BASE_LO    0x04CU   /* CQ GPA[31:0] (RW) */
#define SDXI_REG_CQ_BASE_HI    0x050U   /* CQ GPA[63:32] */

#define SDXI_REG_Q_INTR_VEC    0x054U   /* MSI-X vector for this queue (RW) */

/* ------------------------------------------------------------------ */
/* Register field constants                                            */
/* ------------------------------------------------------------------ */

#define SDXI_DEVID_MAGIC       0x53445849U   /* 'S','D','X','I' */
#define SDXI_VERSION_VAL       0x0100U       /* 1.0 */

/* SDXI_REG_CAP bits */
#define SDXI_CAP_MEMCPY        (1U << 0)
#define SDXI_CAP_MEMSET        (1U << 1)
#define SDXI_CAP_FENCE         (1U << 2)

/* SDXI_REG_STATUS bits */
#define SDXI_STATUS_READY      (1U << 0)
#define SDXI_STATUS_ERROR      (1U << 1)

/* SDXI_REG_CTRL bits */
#define SDXI_CTRL_ENABLE       (1U << 0)
#define SDXI_CTRL_RESET        (1U << 1)   /* self-clearing */

/* SDXI_REG_Q_CTRL bits */
#define SDXI_QCTRL_ENABLE      (1U << 0)
#define SDXI_QCTRL_RESET       (1U << 1)   /* self-clearing */
#define SDXI_QCTRL_INTR_EN     (1U << 2)   /* MSI-X on completion */

/* SDXI_REG_Q_STATUS bits */
#define SDXI_QSTAT_READY       (1U << 0)
#define SDXI_QSTAT_FULL        (1U << 1)
#define SDXI_QSTAT_ERROR       (1U << 2)

/* ------------------------------------------------------------------ */
/* Queue limits                                                        */
/* ------------------------------------------------------------------ */

#define SDXI_MAX_QUEUES        16U
#define SDXI_MAX_QUEUE_DEPTH   256U   /* max SQ/CQ entries */
#define SDXI_MIN_QUEUE_DEPTH   4U

/* ------------------------------------------------------------------ */
/* SQ descriptor (Submission Queue Entry) -- 64 bytes                 */
/* ------------------------------------------------------------------ */

/* Opcodes */
#define SDXI_OP_MEMCPY         0x01U   /* copy length bytes: src_addr -> dst_addr */
#define SDXI_OP_MEMSET         0x02U   /* fill dst_addr with pattern for length bytes */
#define SDXI_OP_FENCE          0x03U   /* ordering barrier; no data movement */

/* SQE flags */
#define SDXI_SQE_FLAG_FENCE    (1U << 0)   /* completion signals fence */

struct sdxi_sqe {
    uint32_t opcode;        /* SDXI_OP_* */
    uint32_t flags;         /* SDXI_SQE_FLAG_* */
    uint32_t sq_id;         /* originating queue id (echo for routing) */
    uint32_t cid;           /* command identifier (echoed in CQE) */
    uint64_t src_addr;      /* source GPA (or fill pattern for MEMSET) */
    uint64_t dst_addr;      /* destination GPA */
    uint64_t length;        /* transfer length in bytes */
    uint8_t  reserved[24];
} __attribute__((packed));

#define SDXI_SQE_SIZE          64U
_Static_assert(sizeof(struct sdxi_sqe) == SDXI_SQE_SIZE,
               "sdxi_sqe must be 64 bytes");

/* ------------------------------------------------------------------ */
/* CQ entry (Completion Queue Entry) -- 16 bytes                      */
/* ------------------------------------------------------------------ */

/* CQE status codes */
#define SDXI_CQE_STATUS_OK     0x00U
#define SDXI_CQE_STATUS_INVAL  0x01U   /* invalid opcode or parameters */
#define SDXI_CQE_STATUS_FAULT  0x02U   /* DMA fault */

struct sdxi_cqe {
    uint32_t status;    /* SDXI_CQE_STATUS_* */
    uint32_t cid;       /* command identifier from SQE */
    uint16_t sq_head;   /* SQ head after consuming this descriptor */
    uint16_t sq_id;     /* originating queue id */
    uint8_t  phase;     /* phase bit -- toggles on each CQ ring wrap */
    uint8_t  reserved[3];
} __attribute__((packed));

#define SDXI_CQE_SIZE          16U
_Static_assert(sizeof(struct sdxi_cqe) == SDXI_CQE_SIZE,
               "sdxi_cqe must be 16 bytes");

/* ------------------------------------------------------------------ */
/* BAR2 doorbell layout                                                */
/* ------------------------------------------------------------------ */

/*
 * Each queue gets a 4 KiB-aligned slot in BAR2.  The guest writes the
 * new SQ tail (32-bit) to offset (queue_id * SDXI_DB_STRIDE).  The
 * emulator sees this as a BAR2 write callback and schedules queue work.
 */
#define SDXI_DB_STRIDE         4096U   /* bytes per doorbell page */
#define SDXI_DB_OFFSET(qid)    ((qid) * SDXI_DB_STRIDE)

#endif /* SDXI_REGS_H */
