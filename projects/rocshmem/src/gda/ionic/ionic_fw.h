/* SPDX-License-Identifier: GPL-2.0 OR Linux-OpenIB */
/*
 * Copyright (c) 2018-2022 Pensando Systems, Inc.  All rights reserved.
 *               2022-2024 Advanced Micro Devices, Inc.  All rights reserved.
 */

#ifndef IONIC_FW_H
#define IONIC_FW_H

#if !defined(__cplusplus)
#include <util/util.h>
#else
#ifndef BIT
#define BIT(n) (1u << (n))
#endif
#endif

#define IONIC_EXP_DBELL_SZ 8

/* common to all versions */

/* wqe scatter gather element */
struct ionic_sge {
	__be64				va;
	__be32				len;
	__be32				lkey;
};

/* admin queue mr type */
enum ionic_mr_flags {
	/* bits that determine mr access */
	IONIC_MRF_LOCAL_WRITE		= BIT(0),
	IONIC_MRF_REMOTE_WRITE		= BIT(1),
	IONIC_MRF_REMOTE_READ		= BIT(2),
	IONIC_MRF_REMOTE_ATOMIC		= BIT(3),
	IONIC_MRF_MW_BIND		= BIT(4),
	IONIC_MRF_ZERO_BASED		= BIT(5),
	IONIC_MRF_ON_DEMAND		= BIT(6),
	IONIC_MRF_PB			= BIT(7),
	IONIC_MRF_ACCESS_MASK		= BIT(12) - 1,

	/* bits that determine mr type */
	IONIC_MRF_IS_MW			= BIT(14),
	IONIC_MRF_INV_EN		= BIT(15),

	/* base flags combinations for mr types */
	IONIC_MRF_USER_MR		= 0,
	IONIC_MRF_PHYS_MR		= IONIC_MRF_INV_EN,
	IONIC_MRF_MW_1			= IONIC_MRF_IS_MW,
	IONIC_MRF_MW_2			= IONIC_MRF_IS_MW | IONIC_MRF_INV_EN,
};

static inline int to_ionic_mr_flags(int access)
{
	int flags = 0;

	if (access & IBV_ACCESS_LOCAL_WRITE)
		flags |= IONIC_MRF_LOCAL_WRITE;

	if (access & IBV_ACCESS_REMOTE_READ)
		flags |= IONIC_MRF_REMOTE_READ;

	if (access & IBV_ACCESS_REMOTE_WRITE)
		flags |= IONIC_MRF_REMOTE_WRITE;

	if (access & IBV_ACCESS_REMOTE_ATOMIC)
		flags |= IONIC_MRF_REMOTE_ATOMIC;

	if (access & IBV_ACCESS_MW_BIND)
		flags |= IONIC_MRF_MW_BIND;

	if (access & IBV_ACCESS_ZERO_BASED)
		flags |= IONIC_MRF_ZERO_BASED;

	return flags;
}

/* cqe status indicated in status_length field when err bit is set */
enum ionic_status {
	IONIC_STS_OK,
	IONIC_STS_LOCAL_LEN_ERR,
	IONIC_STS_LOCAL_QP_OPER_ERR,
	IONIC_STS_LOCAL_PROT_ERR,
	IONIC_STS_WQE_FLUSHED_ERR,
	IONIC_STS_MEM_MGMT_OPER_ERR,
	IONIC_STS_BAD_RESP_ERR,
	IONIC_STS_LOCAL_ACC_ERR,
	IONIC_STS_REMOTE_INV_REQ_ERR,
	IONIC_STS_REMOTE_ACC_ERR,
	IONIC_STS_REMOTE_OPER_ERR,
	IONIC_STS_RETRY_EXCEEDED,
	IONIC_STS_RNR_RETRY_EXCEEDED,
	IONIC_STS_XRC_VIO_ERR,
};

static inline int ionic_to_ibv_status(int sts)
{
	switch (sts) {
	case IONIC_STS_OK:
		return IBV_WC_SUCCESS;
	case IONIC_STS_LOCAL_LEN_ERR:
		return IBV_WC_LOC_LEN_ERR;
	case IONIC_STS_LOCAL_QP_OPER_ERR:
		return IBV_WC_LOC_QP_OP_ERR;
	case IONIC_STS_LOCAL_PROT_ERR:
		return IBV_WC_LOC_PROT_ERR;
	case IONIC_STS_WQE_FLUSHED_ERR:
		return IBV_WC_WR_FLUSH_ERR;
	case IONIC_STS_MEM_MGMT_OPER_ERR:
		return IBV_WC_MW_BIND_ERR;
	case IONIC_STS_BAD_RESP_ERR:
		return IBV_WC_BAD_RESP_ERR;
	case IONIC_STS_LOCAL_ACC_ERR:
		return IBV_WC_LOC_ACCESS_ERR;
	case IONIC_STS_REMOTE_INV_REQ_ERR:
		return IBV_WC_REM_INV_REQ_ERR;
	case IONIC_STS_REMOTE_ACC_ERR:
		return IBV_WC_REM_ACCESS_ERR;
	case IONIC_STS_REMOTE_OPER_ERR:
		return IBV_WC_REM_OP_ERR;
	case IONIC_STS_RETRY_EXCEEDED:
		return IBV_WC_RETRY_EXC_ERR;
	case IONIC_STS_RNR_RETRY_EXCEEDED:
		return IBV_WC_RNR_RETRY_EXC_ERR;
	case IONIC_STS_XRC_VIO_ERR:
	default:
		return IBV_WC_GENERAL_ERR;
	}
}

/* fw abi v1 */

/* data payload part of v1 wqe */
union ionic_v1_pld {
	struct ionic_sge	sgl[2];
	__be32			spec32[8];
	__be16			spec16[16];
	__u8			data[32];
};

/* completion queue v1 cqe */
struct ionic_v1_cqe {
	union {
		struct {
			__le64		wqe_idx_timestamp;
			__be32		src_qpn_op;
			__u8		src_mac[6];
			__be16		vlan_tag;
			__be32		imm_data_rkey;
		} recv;
		struct {
			__u8		rsvd[4];
			__be32		msg_msn;
			__u8		rsvd2[8];
			__le64		npg_wqe_idx_timestamp;
		} send;
	};
	__be32				status_length;
	__be32				qid_type_flags;
};

/* bits for cqe wqe_idx and timestamp */
enum ionic_v1_cqe_wqe_idx_timestamp_bits {
	IONIC_V1_CQE_WQE_IDX_MASK	= 0xffff,
	IONIC_V1_CQE_TIMESTAMP_SHIFT	= 16,
};

/* bits for cqe recv */
enum ionic_v1_cqe_src_qpn_bits {
	IONIC_V1_CQE_RECV_QPN_MASK	= 0xffffff,
	IONIC_V1_CQE_RECV_OP_SHIFT	= 24,

	/* MASK could be 0x3, but need 0x1f for makeshift values:
	 * OP_TYPE_RDMA_OPER_WITH_IMM, OP_TYPE_SEND_RCVD
	 */
	IONIC_V1_CQE_RECV_OP_MASK	= 0x1f,
	IONIC_V1_CQE_RECV_OP_SEND	= 0,
	IONIC_V1_CQE_RECV_OP_SEND_INV	= 1,
	IONIC_V1_CQE_RECV_OP_SEND_IMM	= 2,
	IONIC_V1_CQE_RECV_OP_RDMA_IMM	= 3,

	IONIC_V1_CQE_RECV_IS_IPV4	= BIT(7 + IONIC_V1_CQE_RECV_OP_SHIFT),
	IONIC_V1_CQE_RECV_IS_VLAN	= BIT(6 + IONIC_V1_CQE_RECV_OP_SHIFT),
};

/* bits for cqe qid_type_flags */
enum ionic_v1_cqe_qtf_bits {
	IONIC_V1_CQE_COLOR		= BIT(0),
	IONIC_V1_CQE_ERROR		= BIT(1),
	IONIC_V1_CQE_TYPE_SHIFT		= 5,
	IONIC_V1_CQE_TYPE_MASK		= 0x7,
	IONIC_V1_CQE_QID_SHIFT		= 8,

	IONIC_V1_CQE_TYPE_RECV		= 1,
	IONIC_V1_CQE_TYPE_SEND_MSN	= 2,
	IONIC_V1_CQE_TYPE_SEND_NPG	= 3,
	IONIC_V1_CQE_TYPE_RECV_INDIR	= 4,
};

#if !defined(__HIP_PLATFORM_AMD__) && !defined(__HIP_PLATFORM_HCC__)

static inline bool ionic_v1_cqe_color(struct ionic_v1_cqe *cqe)
{
	return !!(cqe->qid_type_flags & htobe32(IONIC_V1_CQE_COLOR));
}

static inline bool ionic_v1_cqe_error(struct ionic_v1_cqe *cqe)
{
	return !!(cqe->qid_type_flags & htobe32(IONIC_V1_CQE_ERROR));
}

static inline bool ionic_v1_cqe_recv_is_ipv4(struct ionic_v1_cqe *cqe)
{
	return !!(cqe->recv.src_qpn_op &
		  htobe32(IONIC_V1_CQE_RECV_IS_IPV4));
}

static inline bool ionic_v1_cqe_recv_is_vlan(struct ionic_v1_cqe *cqe)
{
	return !!(cqe->recv.src_qpn_op &
		  htobe32(IONIC_V1_CQE_RECV_IS_VLAN));
}

static inline void ionic_v1_cqe_clean(struct ionic_v1_cqe *cqe)
{
	cqe->qid_type_flags |= htobe32(~0u << IONIC_V1_CQE_QID_SHIFT);
}

static inline uint32_t ionic_v1_cqe_qtf(struct ionic_v1_cqe *cqe)
{
	return be32toh(cqe->qid_type_flags);
}

#endif // !defined(__HIP_PLATFORM_AMD__) && !defined(__HIP_PLATFORM_HCC__)

static inline uint8_t ionic_v1_cqe_qtf_type(uint32_t qtf)
{
	return (qtf >> IONIC_V1_CQE_TYPE_SHIFT) & IONIC_V1_CQE_TYPE_MASK;
}

static inline uint32_t ionic_v1_cqe_qtf_qid(uint32_t qtf)
{
	return qtf >> IONIC_V1_CQE_QID_SHIFT;
}

/* v1 base wqe header */
struct ionic_v1_base_hdr {
	__le64				wqe_idx;
	__u8				op;
	__u8				num_sge_key;
	__be16				flags;
	__be32				imm_data_key;
};

/* v1 receive wqe body */
struct ionic_v1_recv_bdy {
	__u8				rsvd[16];
	union ionic_v1_pld		pld;
};

/* v1 send/rdma wqe body (common, has sgl) */
struct ionic_v1_common_bdy {
	union {
		struct {
			__be32		ah_id;
			__be32		dest_qpn;
			__be32		dest_qkey;
		} send;
		struct {
			__be32		remote_va_high;
			__be32		remote_va_low;
			__be32		remote_rkey;
		} rdma;
	};
	__be32				length;
	union ionic_v1_pld		pld;
};

/* v1 atomic wqe body */
struct ionic_v1_atomic_bdy {
	__be32				remote_va_high;
	__be32				remote_va_low;
	__be32				remote_rkey;
	__be32				swap_add_high;
	__be32				swap_add_low;
	__be32				compare_high;
	__be32				compare_low;
	__u8				rsvd[4];
	struct ionic_sge		sge;
};

/* v2 atomic wqe body */
struct ionic_v2_atomic_bdy {
	__be32				remote_va_high;
	__be32				remote_va_low;
	__be32				remote_rkey;
	__be32				swap_add_high;
	__be32				swap_add_low;
	__be32				compare_high;
	__be32				compare_low;
	__be32				lkey;
	__be64				local_va;
	__u8				rsvd_expdb[8];
};

/* v1 bind mw wqe body */
struct ionic_v1_bind_mw_bdy {
	__be64				va;
	__be64				length;
	__be32				lkey;
	__be16				flags;
	__u8				rsvd[26];
};

/* v1 send/recv wqe */
struct ionic_v1_wqe {
	struct ionic_v1_base_hdr	base;
	union {
		struct ionic_v1_recv_bdy	recv;
		struct ionic_v1_common_bdy	common;
		struct ionic_v1_atomic_bdy	atomic;
		struct ionic_v2_atomic_bdy	atomic_v2;
		struct ionic_v1_bind_mw_bdy	bind_mw;
	};
};

/* queue pair v1 send opcodes */
enum ionic_v1_op {
	IONIC_V1_OP_SEND,
	IONIC_V1_OP_SEND_INV,
	IONIC_V1_OP_SEND_IMM,
	IONIC_V1_OP_RDMA_READ,
	IONIC_V1_OP_RDMA_WRITE,
	IONIC_V1_OP_RDMA_WRITE_IMM,
	IONIC_V1_OP_ATOMIC_CS,
	IONIC_V1_OP_ATOMIC_FA,
	IONIC_V1_OP_REG_MR,
	IONIC_V1_OP_LOCAL_INV,
	IONIC_V1_OP_BIND_MW,

	/* flags */
	IONIC_V1_FLAG_FENCE		= BIT(0),
	IONIC_V1_FLAG_SOL		= BIT(1),
	IONIC_V1_FLAG_INL		= BIT(2),
	IONIC_V1_FLAG_SIG		= BIT(3),
	IONIC_V1_FLAG_COLOR		= BIT(4),

	/* flags last four bits for sgl spec format */
	IONIC_V1_FLAG_SPEC32		= (1u << 12),
	IONIC_V1_FLAG_SPEC16		= (2u << 12),
	IONIC_V1_SPEC_FIRST_SGE		= 2,
};

/* queue pair v2 send opcodes */
enum ionic_v2_op {
	IONIC_V2_OPSL_OUT               = 0x20,
	IONIC_V2_OPSL_IMM               = 0x40,
	IONIC_V2_OPSL_INV               = 0x80,

	IONIC_V2_OP_SEND                = 0x0 | IONIC_V2_OPSL_OUT,
	IONIC_V2_OP_SEND_IMM            = IONIC_V2_OP_SEND | IONIC_V2_OPSL_IMM,
	IONIC_V2_OP_SEND_INV            = IONIC_V2_OP_SEND | IONIC_V2_OPSL_INV,

	IONIC_V2_OP_RDMA_WRITE          = 0x1 | IONIC_V2_OPSL_OUT,
	IONIC_V2_OP_RDMA_WRITE_IMM      = IONIC_V2_OP_RDMA_WRITE | IONIC_V2_OPSL_IMM,

	IONIC_V2_OP_RDMA_READ           = 0x2,

	IONIC_V2_OP_ATOMIC_CS           = 0x4,
	IONIC_V2_OP_ATOMIC_FA           = 0x5,
	IONIC_V2_OP_REG_MR              = 0x6,
	IONIC_V2_OP_LOCAL_INV           = 0x7,
	IONIC_V2_OP_BIND_MW             = 0x8,
};

#if !defined(__cplusplus)

static inline size_t ionic_v1_send_wqe_min_size(int min_sge, int min_data,
						int spec, bool expdb)
{
	size_t sz_wqe, sz_sgl, sz_data;

	if (spec > IONIC_V1_SPEC_FIRST_SGE)
		min_sge += IONIC_V1_SPEC_FIRST_SGE;

	if (expdb) {
		min_sge += 1;
		min_data += IONIC_EXP_DBELL_SZ;
	}

	sz_wqe = sizeof(struct ionic_v1_wqe);
	sz_sgl = offsetof(struct ionic_v1_wqe, common.pld.sgl[min_sge]);
	sz_data = offsetof(struct ionic_v1_wqe, common.pld.data[min_data]);

	if (sz_sgl > sz_wqe)
		sz_wqe = sz_sgl;

	if (sz_data > sz_wqe)
		sz_wqe = sz_data;

	return roundup_pow_of_two(sz_wqe);
}

static inline int ionic_v1_send_wqe_max_sge(uint8_t stride_log2, int spec, bool expdb)
{
	struct ionic_v1_wqe *wqe = (void *)0;
	struct ionic_sge *sge = (void *)(1ull << stride_log2);
	int num_sge = 0;

	if (expdb)
		sge -= 1;

	if (spec > IONIC_V1_SPEC_FIRST_SGE)
		num_sge = IONIC_V1_SPEC_FIRST_SGE;

	num_sge = sge - &wqe->common.pld.sgl[num_sge];

	if (spec && num_sge > spec)
		num_sge = spec;

	return num_sge;
}

static inline int ionic_v1_send_wqe_max_data(uint8_t stride_log2, bool expdb)
{
	struct ionic_v1_wqe *wqe = (void *)0;
	__u8 *data = (void *)(1ull << stride_log2);

	if (expdb)
		data -= IONIC_EXP_DBELL_SZ;

	return data - wqe->common.pld.data;
}

static inline size_t ionic_v1_recv_wqe_min_size(int min_sge, int spec, bool expdb)
{
	size_t sz_wqe, sz_sgl;

	if (spec > IONIC_V1_SPEC_FIRST_SGE)
		min_sge += IONIC_V1_SPEC_FIRST_SGE;

	if (expdb)
		min_sge += 1;

	sz_wqe = sizeof(struct ionic_v1_wqe);
	sz_sgl = offsetof(struct ionic_v1_wqe, recv.pld.sgl[min_sge]);

	if (sz_sgl > sz_wqe)
		sz_wqe = sz_sgl;

	return sz_wqe;
}

static inline int ionic_v1_recv_wqe_max_sge(uint8_t stride_log2, int spec, bool expdb)
{
	struct ionic_v1_wqe *wqe = (void *)0;
	struct ionic_sge *sge = (void *)(1ull << stride_log2);
	int num_sge = 0;

	if (expdb)
		sge -= 1;

	if (spec > IONIC_V1_SPEC_FIRST_SGE)
		num_sge = IONIC_V1_SPEC_FIRST_SGE;

	num_sge = sge - &wqe->recv.pld.sgl[num_sge];

	if (spec && num_sge > spec)
		num_sge = spec;

	return num_sge;
}

static inline int ionic_v1_use_spec_sge(int min_sge, int spec)
{
	if (!spec || min_sge > spec)
		return 0;

	if (min_sge <= IONIC_V1_SPEC_FIRST_SGE)
		return IONIC_V1_SPEC_FIRST_SGE;

	return spec;
}

#define IONIC_RCQ_SIZE 4096
#define IONIC_RCQ_DEPTH 128
#define IONIC_RCQ_DEPTH_LOG2 7
#define IONIC_RCQ_STRIDE_LOG2 4

struct ionic_rcq_hdr {
	uint8_t pad[60];
	uint32_t seq_pad;
};

struct ionic_rcqe {
	uint32_t status_length;
	uint32_t imm_data;
	uint32_t seq_flags;
	uint32_t rsvd;
};

enum ionic_rcqe_flag {
	IONIC_RCQE_C = BIT(7),
	IONIC_RCQE_I = BIT(6),
};

struct ionic_rcq {
	struct ionic_rcq_hdr hdr;
	struct ionic_rcqe ring[IONIC_RCQ_DEPTH];
};

static inline uint32_t ionic_rcq_hdr_seq(struct ionic_rcq_hdr *hdr)
{
	return be32toh(hdr->seq_pad) >> 8;
}

static inline uint32_t ionic_rcqe_seq(struct ionic_rcqe *rcqe)
{
	return be32toh(rcqe->seq_flags) >> 8;
}

static inline bool ionic_rcqe_color(struct ionic_rcqe *rcqe)
{
	return !!(rcqe->seq_flags & htobe32(IONIC_RCQE_C));
}

static inline bool ionic_rcqe_imm(struct ionic_rcqe *rcqe)
{
	return !!(rcqe->seq_flags & htobe32(IONIC_RCQE_I));
}

#endif // !defined(__cplusplus)

#endif /* IONIC_FW_H */
