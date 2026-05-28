/*
 * Copyright (c) 2025, Broadcom. All rights reserved.  The term
 * Broadcom refers to Broadcom Limited and/or its subsidiaries.
 *
 * This software is available to you under a choice of one of two
 * licenses.  You may choose to be licensed under the terms of the GNU
 * General Public License (GPL) Version 2, available from the file
 * COPYING in the main directory of this source tree, or the
 * BSD license below:
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions
 * are met:
 *
 * 1. Redistributions of source code must retain the above copyright
 *    notice, this list of conditions and the following disclaimer.
 * 2. Redistributions in binary form must reproduce the above copyright
 *    notice, this list of conditions and the following disclaimer in
 *    the documentation and/or other materials provided with the
 *    distribution.
 *
 * THIS SOFTWARE IS PROVIDED BY THE AUTHOR AND CONTRIBUTORS ``AS IS''
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO,
 * THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
 * PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE AUTHOR OR CONTRIBUTORS
 * BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
 * CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
 * SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR
 * BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY,
 * WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE
 * OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN
 * IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 *
 * Description: Fast path definitions for bnxt_re
 */

#ifndef __BNXT_RE_HSI_H__
#define __BNXT_RE_HSI_H__

#ifdef __cplusplus
extern "C" {
#endif
#define true                        1
#define false                        0

#define BNXT_RE_SLOT_SIZE_BB            16
#define BNXT_RE_STATIC_WQE_SIZE_SLOTS   4
#define BNXT_RE_STATIC_WQE_BB           (BNXT_RE_STATIC_WQE_SIZE_SLOTS * BNXT_RE_SLOT_SIZE_BB)
#define BNXT_RE_STATIC_WQE_SHIFT        6

#define BNXT_RE_STATIC_RQE_SIZE_SLOTS   4
#define BNXT_RE_STATIC_RQE_BB           (BNXT_RE_STATIC_RQE_SIZE_SLOTS * BNXT_RE_SLOT_SIZE_BB)
#define BNXT_RE_STATIC_RQE_SHIFT        6

#define BNXT_RE_STATIC_CQE_SIZE_SLOTS   4
#define BNXT_RE_STATIC_CQE_BB           (BNXT_RE_STATIC_CQE_SIZE_SLOTS * BNXT_RE_SLOT_SIZE_BB)
#define BNXT_RE_STATIC_CQE_SHIFT        6

#define BNXT_RE_QUEUE_START_PHASE       0x01

enum bnxt_re_wr_opcode {
	BNXT_RE_WR_OPCD_SEND                = 0x00,
	BNXT_RE_WR_OPCD_SEND_IMM        = 0x01,
	BNXT_RE_WR_OPCD_SEND_INVAL        = 0x02,
	BNXT_RE_WR_OPCD_RDMA_WRITE        = 0x04,
	BNXT_RE_WR_OPCD_RDMA_WRITE_IMM        = 0x05,
	BNXT_RE_WR_OPCD_RDMA_READ        = 0x06,
	BNXT_RE_WR_OPCD_ATOMIC_CS        = 0x08,
	BNXT_RE_WR_OPCD_ATOMIC_FA        = 0x0B,
	BNXT_RE_WR_OPCD_LOC_INVAL        = 0x0C,
	BNXT_RE_WR_OPCD_BIND                = 0x0E,
	BNXT_RE_WR_OPCD_FR_PPMR                = 0x0F,
	BNXT_RE_WR_OPCD_RECV                = 0x80,
	BNXT_RE_WR_OPCD_INVAL                = 0xFF
};

enum bnxt_re_wr_flags {
	BNXT_RE_WR_FLAGS_DBG_TRACE        = 0x40,
	BNXT_RE_WR_FLAGS_TS_EN                = 0x20,
	BNXT_RE_WR_FLAGS_INLINE                = 0x10,
	BNXT_RE_WR_FLAGS_SE                = 0x08,
	BNXT_RE_WR_FLAGS_UC_FENCE        = 0x04,
	BNXT_RE_WR_FLAGS_RD_FENCE        = 0x02,
	BNXT_RE_WR_FLAGS_SIGNALED        = 0x01
};

#define BNXT_RE_MEMW_TYPE_2                0x02
#define BNXT_RE_MEMW_TYPE_1                0x00
enum bnxt_re_wr_bind_acc {
	BNXT_RE_WR_BIND_ACC_LWR                = 0x01,
	BNXT_RE_WR_BIND_ACC_RRD                = 0x02,
	BNXT_RE_WR_BIND_ACC_RWR                = 0x04,
	BNXT_RE_WR_BIND_ACC_RAT                = 0x08,
	BNXT_RE_WR_BIND_ACC_MWB                = 0x10,
	BNXT_RE_WR_BIND_ACC_ZBVA        = 0x01,
	BNXT_RE_WR_BIND_ACC_SHIFT        = 0x10
};

enum bnxt_re_wc_type {
	BNXT_RE_WC_TYPE_SEND                = 0x00,
	BNXT_RE_WC_TYPE_RECV_RC                = 0x01,
	BNXT_RE_WC_TYPE_RECV_UD                = 0x02,
	BNXT_RE_WC_TYPE_RECV_RAW        = 0x03,
	BNXT_RE_WC_TYPE_NOOP                = 0x0D,
	BNXT_RE_WC_TYPE_TERM                = 0x0E,
	BNXT_RE_WC_TYPE_COFF                = 0x0F
};

#define        BNXT_RE_WC_OPCD_RECV                0x80
enum bnxt_re_req_wc_status {
	BNXT_RE_REQ_ST_OK                = 0x00,
	BNXT_RE_REQ_ST_BAD_RESP                = 0x01,
	BNXT_RE_REQ_ST_LOC_LEN                = 0x02,
	BNXT_RE_REQ_ST_LOC_QP_OP        = 0x03,
	BNXT_RE_REQ_ST_PROT                = 0x04,
	BNXT_RE_REQ_ST_MEM_OP                = 0x05,
	BNXT_RE_REQ_ST_REM_INVAL        = 0x06,
	BNXT_RE_REQ_ST_REM_ACC                = 0x07,
	BNXT_RE_REQ_ST_REM_OP                = 0x08,
	BNXT_RE_REQ_ST_RNR_NAK_XCED        = 0x09,
	BNXT_RE_REQ_ST_TRNSP_XCED        = 0x0A,
	BNXT_RE_REQ_ST_WR_FLUSH                = 0x0B
};

enum bnxt_re_rsp_wc_status {
	BNXT_RE_RSP_ST_OK                = 0x00,
	BNXT_RE_RSP_ST_LOC_ACC                = 0x01,
	BNXT_RE_RSP_ST_LOC_LEN                = 0x02,
	BNXT_RE_RSP_ST_LOC_PROT                = 0x03,
	BNXT_RE_RSP_ST_LOC_QP_OP        = 0x04,
	BNXT_RE_RSP_ST_MEM_OP                = 0x05,
	BNXT_RE_RSP_ST_REM_INVAL        = 0x06,
	BNXT_RE_RSP_ST_WR_FLUSH                = 0x07,
	BNXT_RE_RSP_ST_HW_FLUSH                = 0x08
};

enum bnxt_re_hdr_offset {
	BNXT_RE_HDR_WT_MASK                = 0xFF,
	BNXT_RE_HDR_FLAGS_MASK                = 0xFF,
	BNXT_RE_HDR_FLAGS_SHIFT                = 0x08,
	BNXT_RE_HDR_WS_MASK                = 0xFF,
	BNXT_RE_HDR_WS_SHIFT                = 0x10,
	BNXT_RE_HDR_ZB_SHIFT                = 0x16,
	BNXT_RE_HDR_MW_SHIFT                = 0x17,
	BNXT_RE_HDR_ACC_SHIFT                = 0x18,
	BNXT_RE_HDR_IL_MASK                = 0x0F,
	BNXT_RE_HDR_IL_SHIFT                = 0x18,
};

enum bnxt_re_db_que_type {
	BNXT_RE_QUE_TYPE_SQ                = 0x00,
	BNXT_RE_QUE_TYPE_RQ                = 0x01,
	BNXT_RE_QUE_TYPE_SRQ                = 0x02,
	BNXT_RE_QUE_TYPE_SRQ_ARM        = 0x03,
	BNXT_RE_QUE_TYPE_CQ                = 0x04,
	BNXT_RE_QUE_TYPE_CQ_ARMSE        = 0x05,
	BNXT_RE_QUE_TYPE_CQ_ARMALL        = 0x06,
	BNXT_RE_QUE_TYPE_CQ_ARMENA        = 0x07,
	BNXT_RE_QUE_TYPE_SRQ_ARMENA        = 0x08,
	BNXT_RE_QUE_TYPE_CQ_CUT_ACK        = 0x09,
	BNXT_RE_PUSH_TYPE_START                = 0x0C,
	BNXT_RE_PUSH_TYPE_END                = 0x0D,
	BNXT_RE_QUE_TYPE_NULL                = 0x0F
};

enum bnxt_re_db_mask {
	BNXT_RE_DB_INDX_MASK                = 0xFFFFFFUL,
	BNXT_RE_DB_PILO_MASK                = 0x0FFUL,
	BNXT_RE_DB_PILO_SHIFT                = 0x18,
	BNXT_RE_DB_QID_MASK                = 0xFFFFFUL,
	BNXT_RE_DB_PIHI_MASK                = 0xF00UL,
	BNXT_RE_DB_PIHI_SHIFT                = 0x0C, /* Because mask is 0xF00 */
	BNXT_RE_DB_TYP_MASK                = 0x0FUL,
	BNXT_RE_DB_TYP_SHIFT                = 0x1C,
	BNXT_RE_DB_VALID_SHIFT                = 0x1A,
	BNXT_RE_DB_EPOCH_SHIFT                = 0x18,
	BNXT_RE_DB_TOGGLE_SHIFT                = 0x19,

};

enum bnxt_re_psns_mask {
	BNXT_RE_PSNS_SPSN_MASK                = 0xFFFFFF,
	BNXT_RE_PSNS_OPCD_MASK                = 0xFF,
	BNXT_RE_PSNS_OPCD_SHIFT                = 0x18,
	BNXT_RE_PSNS_NPSN_MASK                = 0xFFFFFF,
	BNXT_RE_PSNS_FLAGS_MASK                = 0xFF,
	BNXT_RE_PSNS_FLAGS_SHIFT        = 0x18
};

enum bnxt_re_msns_mask {
	BNXT_RE_SQ_MSN_SEARCH_START_PSN_MASK        = 0xFFFFFFUL,
	BNXT_RE_SQ_MSN_SEARCH_START_PSN_SHIFT        = 0,
	BNXT_RE_SQ_MSN_SEARCH_NEXT_PSN_MASK        = 0xFFFFFF000000ULL,
	BNXT_RE_SQ_MSN_SEARCH_NEXT_PSN_SHIFT        = 0x18,
	BNXT_RE_SQ_MSN_SEARCH_START_IDX_MASK        = 0xFFFF000000000000ULL,
	BNXT_RE_SQ_MSN_SEARCH_START_IDX_SHIFT        = 0x30
};

enum bnxt_re_bcqe_mask {
	BNXT_RE_BCQE_PH_MASK                = 0x01,
	BNXT_RE_BCQE_TYPE_MASK                = 0x0F,
	BNXT_RE_BCQE_TYPE_SHIFT                = 0x01,
	BNXT_RE_BCQE_RESIZE_TOG_MASK        = 0x03,
	BNXT_RE_BCQE_RESIZE_TOG_SHIFT        = 0x05,
	BNXT_RE_BCQE_STATUS_MASK        = 0xFF,
	BNXT_RE_BCQE_STATUS_SHIFT        = 0x08,
	BNXT_RE_BCQE_FLAGS_MASK                = 0xFFFFU,
	BNXT_RE_BCQE_FLAGS_SHIFT        = 0x10,

	/* wr_id for V1/V2 */
	BNXT_RE_BCQE_RWRID_MASK                = 0xFFFFFU,

	/* higher 16b of source QP for V1/V2 */
	BNXT_RE_BCQE_SRCQP_MASK                = 0xFF,
	BNXT_RE_BCQE_SRCQP_SHIFT        = 0x18
};

enum bnxt_re_rc_flags_mask {
	BNXT_RE_RC_FLAGS_SRQ_RQ_MASK        = 0x01,
	BNXT_RE_RC_FLAGS_IMM_MASK        = 0x02,
	BNXT_RE_RC_FLAGS_IMM_SHIFT        = 0x01,
	BNXT_RE_RC_FLAGS_INV_MASK        = 0x04,
	BNXT_RE_RC_FLAGS_INV_SHIFT        = 0x02,
	BNXT_RE_RC_FLAGS_RDMA_MASK        = 0x08,
	BNXT_RE_RC_FLAGS_RDMA_SHIFT        = 0x03
};

enum bnxt_re_ud_flags_mask {
	BNXT_RE_UD_FLAGS_SRQ_RQ_MASK        = 0x01,
	BNXT_RE_UD_FLAGS_SRQ_RQ_SFT        = 0x00,
	BNXT_RE_UD_FLAGS_IMM_MASK        = 0x02,
	BNXT_RE_UD_FLAGS_IMM_SFT        = 0x01,
	BNXT_RE_UD_FLAGS_IP_VER_MASK        = 0x30,
	BNXT_RE_UD_FLAGS_IP_VER_SFT        = 0x4,

	/* the following has been removed in V3 */
	BNXT_RE_UD_FLAGS_META_MASK        = 0x3C0,
	BNXT_RE_UD_FLAGS_META_SFT        = 0x6,
	BNXT_RE_UD_FLAGS_EXT_META_MASK        = 0xC00,
	BNXT_RE_UD_FLAGS_EXT_META_SFT        = 0x10,
};

enum bnxt_re_ud_cqe_mask {
	BNXT_RE_UD_CQE_MAC_MASK                = 0xFFFFFFFFFFFFULL,
	BNXT_RE_UD_CQE_SRCQPLO_MASK        = 0xFFFF,
	BNXT_RE_UD_CQE_SRCQPLO_SHIFT        = 0x30,
	BNXT_RE_UD_CQE_LEN_MASK                = 0x3FFFU
};

enum bnxt_re_que_flags_mask {
	BNXT_RE_FLAG_EPOCH_TAIL_SHIFT        = 0x0UL,
	BNXT_RE_FLAG_EPOCH_HEAD_SHIFT        = 0x1UL,
	BNXT_RE_FLAG_EPOCH_TAIL_MASK        = 0x1UL,
	BNXT_RE_FLAG_EPOCH_HEAD_MASK        = 0x2UL,
};

enum bnxt_re_db_epoch_flag_shift {
	BNXT_RE_DB_EPOCH_TAIL_SHIFT        = BNXT_RE_DB_EPOCH_SHIFT,
	BNXT_RE_DB_EPOCH_HEAD_SHIFT        = (BNXT_RE_DB_EPOCH_SHIFT - 1)
};

enum bnxt_re_ppp_st_en_mask {
	BNXT_RE_PPP_ENABLED_MASK        = 0x1UL,
	BNXT_RE_PPP_STATE_MASK                = 0x2UL,
};

enum bnxt_re_ppp_st_shift {
	BNXT_RE_PPP_ST_SHIFT                = 0x1UL
};

struct bnxt_re_db_hdr {
	__u64 typ_qid_indx; /* typ: 4, qid:20 (qid:12 on V3), indx:24 */
};

#define BNXT_RE_CHIP_ID0_CHIP_NUM_SFT                0x00
#define BNXT_RE_CHIP_ID0_CHIP_REV_SFT                0x10
#define BNXT_RE_CHIP_ID0_CHIP_MET_SFT                0x18

#define BNXT_RE_STATIC_WQE_MAX_SGE			0x06
#define BNXT_RE_WQE_MODES_WQE_MODE_MASK                0x01

struct bnxt_re_bcqe {
	__u32 flg_st_typ_ph;
	__u32 qphi_rwrid;        /* This field becomes opaque in V3 */
} __attribute__((packed));

struct bnxt_re_req_cqe {
	__u64 qp_handle;
	__u32 con_indx; /* 16 bits valid. */
	__u32 rsvd1;
	__u64 rsvd2;
} __attribute__((packed));

struct bnxt_re_rc_cqe {
	__u32 length;
	__u32 imm_key;
	__u64 qp_handle;
	__u64 mr_handle;
} __attribute__((packed));

struct bnxt_re_ud_cqe {
	__u32 length; /* 14 bits */
	__u32 immd;
	__u64 qp_handle;
	__u64 qplo_mac; /* 16:48*/
} __attribute__((packed));

struct bnxt_re_term_cqe {
	__u64 qp_handle;
	__u32 rq_sq_cidx;
	__u32 rsvd;
	__u64 rsvd1;
} __attribute__((packed));

struct bnxt_re_cutoff_cqe {
	__u64 rsvd1;
	__u64 rsvd2;
	__u64 rsvd3;
	__u8 cqe_type_toggle;
	__u8 status;
	__u16 rsvd4;
	__u32 rsvd5;
} __attribute__((packed));

union lower_shdr {
	__u64 qkey_len;
	__u64 lkey_plkey;
	__u64 rva;
};

struct bnxt_re_bsqe {
	__u32 rsv_ws_fl_wt;
	__u32 key_immd;
	union lower_shdr lhdr;
} __attribute__((packed));

struct bnxt_re_psns_ext {
	__u32 opc_spsn;
	__u32 flg_npsn;
	__u16 st_slot_idx;
	__u16 rsvd0;
	__u32 rsvd1;
} __attribute__((packed));

/* sq_msn_search (size:64b/8B) */
struct bnxt_re_msns {
	__u64  start_idx_next_psn_start_psn;
} __attribute__((packed));

struct bnxt_re_psns {
	__u32 opc_spsn;
	__u32 flg_npsn;
} __attribute__((packed));

struct bnxt_re_sge {
	__u64 pa;
	__u32 lkey;
	__u32 length;
} __attribute__((packed));

struct bnxt_re_send {
	__u32 dst_qp;
	__u32 avid;
	__u64 rsvd;
} __attribute__((packed));

struct bnxt_re_raw {
	__u32 cfa_meta;
	__u32 ts; /* timestamp for V3 */
	__u64 rsvd3; /* timestamp for V1/V2 */
} __attribute__((packed));

struct bnxt_re_rdma {
	__u64 rva;
	__u32 rkey;
	__u32 ts; /* timestamp for V3 */
} __attribute__((packed));

struct bnxt_re_atomic {
	__u64 swp_dt;
	__u64 cmp_dt;
} __attribute__((packed));

struct bnxt_re_inval {
	__u64 rsvd[2];
} __attribute__((packed));

struct bnxt_re_bind {
	__u64 va;
	__u64 len; /* only 40 bits are valid for V1/V2. Full 64-bit for V3 */
} __attribute__((packed));

struct bnxt_re_brqe {
	__u32 rsv_ws_fl_wt;
	__u32 opaque; /* opaque is V3 only */
	__u32 wrid; /* wrid is V1/V2 only */
	__u32 rsvd1;
} __attribute__((packed));

/* V1/V2 only. For V3, sge immediately follows struct bnxt_re_brqe */
struct bnxt_re_rqe {
	__u64 rsvd[2];
} __attribute__((packed));

/* SRQ */
struct bnxt_re_srqe {
	__u64 rsvd[2];
} __attribute__((packed));

struct bnxt_re_push_wqe {
	__u64 addr[32];
} __attribute__((packed));
#ifdef __cplusplus
}
#endif
#endif
