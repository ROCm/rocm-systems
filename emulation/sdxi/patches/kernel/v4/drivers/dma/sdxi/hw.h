/* SPDX-License-Identifier: GPL-2.0-only */
/* Copyright Advanced Micro Devices, Inc. */

/*
 * Hardware layouts and constants follow SDXI specification names and
 * definition order. Reserved fields and bits must be zero unless specified
 * otherwise.
 */

#ifndef DMA_SDXI_HW_H
#define DMA_SDXI_HW_H

#include <linux/bits.h>
#include <linux/build_bug.h>
#include <linux/stddef.h>
#include <linux/types.h>
#include <asm/byteorder.h>

/* SDXI 1.0 Table 3-2: Context Level 2 Table Entry (CXT_L2_ENT) */
struct sdxi_cxt_L2_ent {
	__le64 lv01_ptr;
#define SDXI_CXT_L2_ENT_VL       BIT_ULL(0)
#define SDXI_CXT_L2_ENT_LV01_PTR GENMASK_ULL(63, 12)
} __packed __aligned(8);
static_assert(sizeof(struct sdxi_cxt_L2_ent) == 8);

/* SDXI 1.0 3.2.1 Context Level 2 Table */
#define SDXI_L2_TABLE_ENTRIES 512
struct sdxi_cxt_L2_table {
	struct sdxi_cxt_L2_ent entry[SDXI_L2_TABLE_ENTRIES];
};
static_assert(sizeof(struct sdxi_cxt_L2_table) == 4096);

/* SDXI 1.0 Table 3-3: Context Level 1 Table Entry (CXT_L1_ENT) */
struct sdxi_cxt_L1_ent {
	__le64 cxt_ctl_ptr;
#define SDXI_CXT_L1_ENT_VL             BIT_ULL(0)
#define SDXI_CXT_L1_ENT_KA             BIT_ULL(1)
#define SDXI_CXT_L1_ENT_PV             BIT_ULL(2)
#define SDXI_CXT_L1_ENT_CXT_CTL_PTR    GENMASK_ULL(63, 6)
	__le64 akey_ptr;
#define SDXI_CXT_L1_ENT_AKEY_SZ        GENMASK_ULL(3, 0)
#define SDXI_CXT_L1_ENT_AKEY_PTR       GENMASK_ULL(63, 12)
	__le32 misc0;
#define SDXI_CXT_L1_ENT_PASID          GENMASK(19, 0)
#define SDXI_CXT_L1_ENT_MAX_BUFFER     GENMASK(23, 20)
	__le32 opb_000_enb;
	__u8 rsvd_0[8];
} __packed __aligned(32);
static_assert(sizeof(struct sdxi_cxt_L1_ent) == 32);

/* SDXI 1.0 3.2.2 Context Level 1 Table */
#define SDXI_L1_TABLE_ENTRIES 128
struct sdxi_cxt_L1_table {
	struct sdxi_cxt_L1_ent entry[SDXI_L1_TABLE_ENTRIES];
};
static_assert(sizeof(struct sdxi_cxt_L1_table) == 4096);

/* SDXI 1.0 Table 3-4: Context Control (CXT_CTL) */
struct sdxi_cxt_ctl {
	__le64 ds_ring_ptr;
#define SDXI_CXT_CTL_VL             BIT_ULL(0)
#define SDXI_CXT_CTL_QOS            GENMASK_ULL(3, 2)
#define SDXI_CXT_CTL_SE             BIT_ULL(4)
#define SDXI_CXT_CTL_CSA            BIT_ULL(5)
#define SDXI_CXT_CTL_DS_RING_PTR    GENMASK_ULL(63, 6)
	__le32 ds_ring_sz;
	__u8 rsvd_0[4];
	__le64 cxt_sts_ptr;
#define SDXI_CXT_CTL_CXT_STS_PTR    GENMASK_ULL(63, 4)
	__le64 write_index_ptr;
#define SDXI_CXT_CTL_WRITE_INDEX_PTR GENMASK_ULL(63, 3)
	__u8 rsvd_1[32];
} __packed __aligned(64);
static_assert(sizeof(struct sdxi_cxt_ctl) == 64);

/* SDXI 1.0 Table 3-5: Context Status (CXT_STS) */
struct sdxi_cxt_sts {
	__u8 state;
#define SDXI_CXT_STS_STATE GENMASK(3, 0)
	__u8 misc0;
	__u8 rsvd_0[6];
	__le64 read_index;
} __packed __aligned(16);
static_assert(sizeof(struct sdxi_cxt_sts) == 16);

/* SDXI 1.0 Table 3-6: CXT_STS.state Encoding */
/* Valid values for FIELD_GET(SDXI_CXT_STS_STATE, sdxi_cxt_sts.state). */
enum cxt_sts_state {
	CXTV_STOP_SW  = 0x0,
	CXTV_RUN      = 0x1,
	CXTV_STOPG_SW = 0x2,
	CXTV_STOP_FN  = 0x4,
	CXTV_STOPG_FN = 0x6,
	CXTV_ERR_FN   = 0xf,
};

/* SDXI 1.0 Table 3-7: AKey Table Entry (AKEY_ENT) */
struct sdxi_akey_ent {
	__le16 intr_num;
#define SDXI_AKEY_ENT_VL BIT(0)
#define SDXI_AKEY_ENT_IV BIT(1)
#define SDXI_AKEY_ENT_INTR_NUM GENMASK(14, 4)
	__le16 tgt_sfunc;
	__le32 pasid;
	__le16 stag;
	__u8   rsvd_0[2];
	__le16 rkey;
	__u8   rsvd_1[2];
} __packed __aligned(16);
static_assert(sizeof(struct sdxi_akey_ent) == 16);

/* SDXI 1.0 Table 6-4: CST_BLK (Completion Status Block) */
struct sdxi_cst_blk {
	__le64 signal;
	__le32 flags;
#define SDXI_CST_BLK_ER BIT(31)
	__u8 rsvd_0[20];
} __packed __aligned(32);
static_assert(sizeof(struct sdxi_cst_blk) == 32);

/* SDXI 1.0 4.4.2 Completion-Status Modes */
enum sdxi_cst_blk_signal {
	SDXI_CST_BLK_SIGNAL_TERMINAL = 0,
	SDXI_CST_BLK_SIGNAL_INIT     = 1,
};

struct sdxi_desc {
	union {
		/*
		 * SDXI 1.0 Table 6-3: DSC_GENERIC SDXI Descriptor
		 * Common Header and Footer Format
		 */
		struct_group_tagged(sdxi_dsc_generic, generic,
			__le32 opcode;
			__u8 operation[52];
			__le64 csb_ptr;
		);

/* For opcode field */
#define SDXI_DSC_VL  BIT(0)
#define SDXI_DSC_SE  BIT(1)
#define SDXI_DSC_FE  BIT(2)
#define SDXI_DSC_CSR BIT(4)
#define SDXI_DSC_SUBTYPE GENMASK(15, 8)
#define SDXI_DSC_TYPE    GENMASK(26, 16)

/* For csb_ptr field */
#define SDXI_DSC_NP BIT_ULL(0)
#define SDXI_DSC_CSB_PTR GENMASK_ULL(63, 5)

#define define_sdxi_dsc(tag_, name_, op_body_)				\
	struct tag_ {							\
		__le32 opcode;						\
		op_body_						\
		__le64 csb_ptr;						\
	} __packed __aligned(64) name_;					\
	static_assert(sizeof(struct tag_) ==				\
		      sizeof(struct sdxi_dsc_generic));			\
	static_assert(offsetof(struct tag_, csb_ptr) ==			\
		      offsetof(struct sdxi_dsc_generic, csb_ptr))

		/* SDXI 1.0 Table 6-8: DSC_DMAB_COPY Descriptor Format */
		define_sdxi_dsc(sdxi_dsc_dmab_copy, copy,
			__le32 size;
			__u8 attr;
			__u8 rsvd_0[3];
			__le16 akey0;
			__le16 akey1;
			__le64 addr0;
			__le64 addr1;
			__u8 rsvd_1[24];
		);

		/* SDXI 1.0 Table 6-12: DSC_INTR Descriptor Format */
		define_sdxi_dsc(sdxi_dsc_intr, intr,
			__u8 rsvd_0[8];
			__le16 akey;
			__u8 rsvd_1[42];
		);

		/* SDXI 1.0 Table 6-14: DSC_CXT_START Descriptor Format */
		define_sdxi_dsc(sdxi_dsc_cxt_start, cxt_start,
			__u8 rsvd_0;
			__u8 vflags;
			__le16 vf_num;
			__le16 cxt_start;
			__le16 cxt_end;
			__u8 rsvd_1[4];
			__le64 db_value;
			__u8 rsvd_2[32];
		);

		/* SDXI 1.0 Table 6-15: DSC_CXT_STOP Descriptor Format */
		define_sdxi_dsc(sdxi_dsc_cxt_stop, cxt_stop,
			__u8 rsvd_0;
			__u8 vflags;
			__le16 vf_num;
			__le16 cxt_start;
			__le16 cxt_end;
			__u8 rsvd_1[44];
		);

		/* SDXI 1.0 Table 6-18: DSC_CXT_UPD Descriptor Format */
		define_sdxi_dsc(sdxi_dsc_cxt_upd, cxt_upd,
			__u8 cflags;
			__u8 vflags;
			__le16 vf_num;
			__le16 cxt_start;
			__le16 cxt_end;
			__u8 rsvd_0[44];
		);
/* For use with cxt_upd.cflags */
#define SDXI_DSC_CXT_UPD_PCD GENMASK(2, 0)

		/* SDXI 1.0 Table 6-22: DSC_SYNC Descriptor Format */
		define_sdxi_dsc(sdxi_dsc_sync, sync,
			__u8 cflags;
			__u8 vflags;
			__le16 vf_num;
			__le16 cxt_start;
			__le16 cxt_end;
			__le16 key_start;
			__le16 key_end;
			__u8 rsvd_0[40];
		);
/* For use with sync.cflags */
#define SDXI_DSC_SYNC_FLT GENMASK(2, 0)

#undef define_sdxi_dsc
	};
} __packed __aligned(64);
static_assert(sizeof(struct sdxi_desc) == 64);

/* SDXI 1.0 Table 6-1: SDXI Operation Groups */
enum sdxi_dsc_type {
	SDXI_DSC_OP_TYPE_DMAB    = 0x001,
	SDXI_DSC_OP_TYPE_ADMIN   = 0x002,
	SDXI_DSC_OP_TYPE_INTR    = 0x004,
};

/* SDXI 1.0 Table 6-2: SDXI Operation Groups, Types, and Subtypes */
enum sdxi_dsc_subtype {
	/* DMA Base */
	SDXI_DSC_OP_SUBTYPE_COPY    = 0x03,

	/* Interrupt */
	SDXI_DSC_OP_SUBTYPE_INTR = 0x00,

	/* Administrative */
	SDXI_DSC_OP_SUBTYPE_CXT_UPD      = 0x01,
	SDXI_DSC_OP_SUBTYPE_CXT_START_NM = 0x03,
	SDXI_DSC_OP_SUBTYPE_CXT_STOP     = 0x04,
	SDXI_DSC_OP_SUBTYPE_SYNC         = 0x06,
};

#endif /* DMA_SDXI_HW_H */
