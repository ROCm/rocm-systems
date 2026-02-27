/*
 * Copyright (c) 2017 Mellanox Technologies, Inc.  All rights reserved.
 *
 * This software is available to you under a choice of one of two
 * licenses.  You may choose to be licensed under the terms of the GNU
 * General Public License (GPL) Version 2, available from the file
 * COPYING in the main directory of this source tree, or the
 * OpenIB.org BSD license below:
 *
 *     Redistribution and use in source and binary forms, with or
 *     without modification, are permitted provided that the following
 *     conditions are met:
 *
 *      - Redistributions of source code must retain the above
 *        copyright notice, this list of conditions and the following
 *        disclaimer.
 *
 *      - Redistributions in binary form must reproduce the above
 *        copyright notice, this list of conditions and the following
 *        disclaimer in the documentation and/or other materials
 *        provided with the distribution.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
 * EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
 * MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND
 * NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS
 * BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN
 * ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN
 * CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 * SOFTWARE.
 */

#ifndef LIBRARY_SRC_GDA_MLX5DV_CORE_HPP_
#define LIBRARY_SRC_GDA_MLX5DV_CORE_HPP_

#include <stdint.h>
#include <linux/types.h> /* For the __be64 type */
#include <sys/types.h>
#include "gda/ibv_core.hpp"

extern "C" {

/* infiniband/mlx5_user_ioctl_verbs.h */
enum mlx5_ib_uapi_uar_alloc_type {
	MLX5_IB_UAPI_UAR_ALLOC_TYPE_BF = 0x0,
	MLX5_IB_UAPI_UAR_ALLOC_TYPE_NC = 0x1,
};

/* infiniband/mlx5_api.h */
#define MLX5DV_UAR_ALLOC_TYPE_BF MLX5_IB_UAPI_UAR_ALLOC_TYPE_BF
#define MLX5DV_UAR_ALLOC_TYPE_NC MLX5_IB_UAPI_UAR_ALLOC_TYPE_NC
#define MLX5DV_UAR_ALLOC_TYPE_NC_DEDICATED (1U << 31)

/* infiniband/tm_types.h */
struct ibv_tmh {
	uint8_t		opcode;      /* from enum ibv_tmh_op */
	uint8_t		reserved[3]; /* must be zero */
	__be32		app_ctx;     /* opaque user data */
	__be64		tag;
};

/* infiniband/mlx5dv.h */
enum {
	MLX5_RCV_DBR	= 0,
	MLX5_SND_DBR	= 1,
};

struct mlx5dv_qp {
	__be32			*dbrec;
	struct {
		void		*buf;
		uint32_t	wqe_cnt;
		uint32_t	stride;
	} sq;
	struct {
		void		*buf;
		uint32_t	wqe_cnt;
		uint32_t	stride;
	} rq;
	struct {
		void		*reg;
		uint32_t	size;
	} bf;
	uint64_t		comp_mask;
	off_t			uar_mmap_offset;
	uint32_t		tirn;
	uint32_t		tisn;
	uint32_t		rqn;
	uint32_t		sqn;
	uint64_t		tir_icm_addr;
};

struct mlx5dv_cq {
	void			*buf;
	__be32			*dbrec;
	uint32_t		cqe_cnt;
	uint32_t		cqe_size;
	void			*cq_uar;
	uint32_t		cqn;
	uint64_t		comp_mask;
};

struct mlx5_wqe_av;

struct mlx5dv_ah {
	struct mlx5_wqe_av      *av;
	uint64_t		comp_mask;
};

struct mlx5dv_pd {
	uint32_t		pdn;
	uint64_t		comp_mask;
};

struct mlx5dv_obj {
	struct {
		struct ibv_qp		*in;
		struct mlx5dv_qp	*out;
	} qp;
	struct {
		struct ibv_cq		*in;
		struct mlx5dv_cq	*out;
	} cq;
	struct {
		struct ibv_srq		*in;
		struct mlx5dv_srq	*out;
	} srq;
	struct {
		struct ibv_wq		*in;
		struct mlx5dv_rwq	*out;
	} rwq;
	struct {
		struct ibv_dm		*in;
		struct mlx5dv_dm	*out;
	} dm;
	struct {
		struct ibv_ah		*in;
		struct mlx5dv_ah	*out;
	} ah;
	struct {
		struct ibv_pd		*in;
		struct mlx5dv_pd	*out;
	} pd;
	struct {
		struct mlx5dv_devx_obj *in;
		struct mlx5dv_devx *out;
	} devx;
};

enum mlx5dv_obj_type {
	MLX5DV_OBJ_QP	= 1 << 0,
	MLX5DV_OBJ_CQ	= 1 << 1,
	MLX5DV_OBJ_SRQ	= 1 << 2,
	MLX5DV_OBJ_RWQ	= 1 << 3,
	MLX5DV_OBJ_DM	= 1 << 4,
	MLX5DV_OBJ_AH	= 1 << 5,
	MLX5DV_OBJ_PD	= 1 << 6,
	MLX5DV_OBJ_DEVX	= 1 << 7,
};

enum {
	MLX5_OPCODE_NOP			= 0x00,
	MLX5_OPCODE_SEND_INVAL		= 0x01,
	MLX5_OPCODE_RDMA_WRITE		= 0x08,
	MLX5_OPCODE_RDMA_WRITE_IMM	= 0x09,
	MLX5_OPCODE_SEND		= 0x0a,
	MLX5_OPCODE_SEND_IMM		= 0x0b,
	MLX5_OPCODE_TSO			= 0x0e,
	MLX5_OPCODE_RDMA_READ		= 0x10,
	MLX5_OPCODE_ATOMIC_CS		= 0x11,
	MLX5_OPCODE_ATOMIC_FA		= 0x12,
	MLX5_OPCODE_ATOMIC_MASKED_CS	= 0x14,
	MLX5_OPCODE_ATOMIC_MASKED_FA	= 0x15,
	MLX5_OPCODE_FMR			= 0x19,
	MLX5_OPCODE_LOCAL_INVAL		= 0x1b,
	MLX5_OPCODE_CONFIG_CMD		= 0x1f,
	MLX5_OPCODE_SET_PSV		= 0x20,
	MLX5_OPCODE_UMR			= 0x25,
	MLX5_OPCODE_TAG_MATCHING	= 0x28,
	MLX5_OPCODE_FLOW_TBL_ACCESS     = 0x2c,
	MLX5_OPCODE_MMO			= 0x2F,
};

enum {
	MLX5_CQE_SYNDROME_LOCAL_LENGTH_ERR		= 0x01,
	MLX5_CQE_SYNDROME_LOCAL_QP_OP_ERR		= 0x02,
	MLX5_CQE_SYNDROME_LOCAL_PROT_ERR		= 0x04,
	MLX5_CQE_SYNDROME_WR_FLUSH_ERR			= 0x05,
	MLX5_CQE_SYNDROME_MW_BIND_ERR			= 0x06,
	MLX5_CQE_SYNDROME_BAD_RESP_ERR			= 0x10,
	MLX5_CQE_SYNDROME_LOCAL_ACCESS_ERR		= 0x11,
	MLX5_CQE_SYNDROME_REMOTE_INVAL_REQ_ERR		= 0x12,
	MLX5_CQE_SYNDROME_REMOTE_ACCESS_ERR		= 0x13,
	MLX5_CQE_SYNDROME_REMOTE_OP_ERR			= 0x14,
	MLX5_CQE_SYNDROME_TRANSPORT_RETRY_EXC_ERR	= 0x15,
	MLX5_CQE_SYNDROME_RNR_RETRY_EXC_ERR		= 0x16,
	MLX5_CQE_SYNDROME_REMOTE_ABORTED_ERR		= 0x22,
};

enum {
	MLX5_CQE_OWNER_MASK	= 1,
	MLX5_CQE_REQ		= 0,
	MLX5_CQE_RESP_WR_IMM	= 1,
	MLX5_CQE_RESP_SEND	= 2,
	MLX5_CQE_RESP_SEND_IMM	= 3,
	MLX5_CQE_RESP_SEND_INV	= 4,
	MLX5_CQE_RESIZE_CQ	= 5,
	MLX5_CQE_NO_PACKET	= 6,
	MLX5_CQE_SIG_ERR	= 12,
	MLX5_CQE_REQ_ERR	= 13,
	MLX5_CQE_RESP_ERR	= 14,
	MLX5_CQE_INVALID	= 15,
};

struct mlx5_err_cqe {
	uint8_t		rsvd0[32];
	uint32_t	srqn;
	uint8_t		rsvd1[18];
	uint8_t		vendor_err_synd;
	uint8_t		syndrome;
	uint32_t	s_wqe_opcode_qpn;
	uint16_t	wqe_counter;
	uint8_t		signature;
	uint8_t		op_own;
};

struct mlx5_tm_cqe {
	__be32		success;
	__be16		hw_phase_cnt;
	uint8_t		rsvd0[12];
};

struct mlx5_cqe64 {
	union {
		struct {
			uint8_t		rsvd0[2];
			__be16		wqe_id;
			uint8_t		rsvd4[13];
			uint8_t		ml_path;
			uint8_t		rsvd20[4];
			__be16		slid;
			__be32		flags_rqpn;
			uint8_t		hds_ip_ext;
			uint8_t		l4_hdr_type_etc;
			__be16		vlan_info;
		};
		struct mlx5_tm_cqe tm_cqe;
		/* TMH is scattered to CQE upon match */
		struct ibv_tmh tmh;
	};
	__be32		srqn_uidx;
	__be32		imm_inval_pkey;
	uint8_t		app;
	uint8_t		app_op;
	__be16		app_info;
	__be32		byte_cnt;
	__be64		timestamp;
	__be32		sop_drop_qpn;
	__be16		wqe_counter;
	uint8_t		signature;
	uint8_t		op_own;
};

enum {
	MLX5_WQE_CTRL_CQ_UPDATE	= 2 << 2,
	MLX5_WQE_CTRL_SOLICITED	= 1 << 1,
	MLX5_WQE_CTRL_FENCE	= 4 << 5,
	MLX5_WQE_CTRL_INITIATOR_SMALL_FENCE = 1 << 5,
};

enum {
	MLX5_SEND_WQE_BB	= 64,
	MLX5_SEND_WQE_SHIFT	= 6,
};

enum {
	MLX5_INLINE_SEG	= 0x80000000,
};

struct mlx5_wqe_data_seg {
	__be32			byte_count;
	__be32			lkey;
	__be64			addr;
};

struct mlx5_wqe_ctrl_seg {
	__be32		opmod_idx_opcode;
	__be32		qpn_ds;
	uint8_t		signature;
	__be16		dci_stream_channel_id;
	uint8_t		fm_ce_se;
	__be32		imm;
} __attribute__((__packed__)) __attribute__((__aligned__(4)));

struct mlx5_wqe_raddr_seg {
	__be64		raddr;
	__be32		rkey;
	__be32		reserved;
};

struct mlx5_wqe_atomic_seg {
	__be64		swap_add;
	__be64		compare;
};

struct mlx5_wqe_inl_data_seg {
	uint32_t	byte_count;
};

enum mlx5dv_context_attr_flags {
	MLX5DV_CONTEXT_FLAGS_DEVX = 1 << 0,
};

enum mlx5dv_context_attr_comp_mask {
	MLX5DV_CONTEXT_ATTR_MASK_FD_ARRAY = 1 << 0,
};

struct mlx5dv_context_attr {
	uint32_t flags; /* Use enum mlx5dv_context_attr_flags */
	uint64_t comp_mask; /* Use enum mlx5dv_context_attr_comp_mask */
	struct ibv_fd_arr *fds;
};

struct mlx5dv_devx_obj;

struct mlx5dv_devx_umem {
	uint32_t umem_id;
};

enum  mlx5dv_devx_umem_in_mask {
	MLX5DV_UMEM_MASK_DMABUF = 1 << 0,
};

struct mlx5dv_devx_umem_in {
	void *addr;
	size_t size;
	uint32_t access;
	uint64_t pgsz_bitmap;
	uint64_t comp_mask;
	int dmabuf_fd;
};

struct mlx5dv_devx_uar {
	void *reg_addr;
	void *base_addr;
	uint32_t page_id;
	off_t mmap_off;
	uint64_t comp_mask;
};

#define __devx_nullp(typ) ((struct mlx5_ifc_##typ##_bits *)NULL)
#define __devx_st_sz_bits(typ) sizeof(struct mlx5_ifc_##typ##_bits)
#define __devx_bit_sz(typ, fld) sizeof(__devx_nullp(typ)->fld)
#define __devx_bit_off(typ, fld) offsetof(struct mlx5_ifc_##typ##_bits, fld)
#define __devx_dw_off(bit_off) ((bit_off) / 32)
#define __devx_64_off(bit_off) ((bit_off) / 64)
#define __devx_dw_bit_off(bit_sz, bit_off) (32 - (bit_sz) - ((bit_off) & 0x1f))
#define __devx_mask(bit_sz) ((uint32_t)((1ull << (bit_sz)) - 1))
#define __devx_dw_mask(bit_sz, bit_off)                                        \
	(__devx_mask(bit_sz) << __devx_dw_bit_off(bit_sz, bit_off))

#define DEVX_FLD_SZ_BYTES(typ, fld) (__devx_bit_sz(typ, fld) / 8)
#define DEVX_ST_SZ_BYTES(typ) (sizeof(struct mlx5_ifc_##typ##_bits) / 8)
#define DEVX_ST_SZ_DW(typ) (sizeof(struct mlx5_ifc_##typ##_bits) / 32)
#define DEVX_ST_SZ_QW(typ) (sizeof(struct mlx5_ifc_##typ##_bits) / 64)
#define DEVX_UN_SZ_BYTES(typ) (sizeof(union mlx5_ifc_##typ##_bits) / 8)
#define DEVX_UN_SZ_DW(typ) (sizeof(union mlx5_ifc_##typ##_bits) / 32)
#define DEVX_BYTE_OFF(typ, fld) (__devx_bit_off(typ, fld) / 8)
#define DEVX_ADDR_OF(typ, p, fld)                                              \
	((unsigned char *)(p) + DEVX_BYTE_OFF(typ, fld))

static inline void _devx_set(void *p, uint32_t value, size_t bit_off,
			     size_t bit_sz)
{
	__be32 *fld = (__be32 *)(p) + __devx_dw_off(bit_off);
	uint32_t dw_mask = __devx_dw_mask(bit_sz, bit_off);
	uint32_t mask = __devx_mask(bit_sz);

	*fld = htobe32((be32toh(*fld) & (~dw_mask)) |
		       ((value & mask) << __devx_dw_bit_off(bit_sz, bit_off)));
}

#define DEVX_SET(typ, p, fld, v)                                               \
	_devx_set(p, v, __devx_bit_off(typ, fld), __devx_bit_sz(typ, fld))

static inline uint32_t _devx_get(const void *p, size_t bit_off, size_t bit_sz)
{
	return ((be32toh(*((const __be32 *)(p) + __devx_dw_off(bit_off))) >>
		 __devx_dw_bit_off(bit_sz, bit_off)) &
		__devx_mask(bit_sz));
}

#define DEVX_GET(typ, p, fld)                                                  \
	_devx_get(p, __devx_bit_off(typ, fld), __devx_bit_sz(typ, fld))

static inline void _devx_set64(void *p, uint64_t v, size_t bit_off)
{
	*((__be64 *)(p) + __devx_64_off(bit_off)) = htobe64(v);
}

#define DEVX_SET64(typ, p, fld, v) _devx_set64(p, v, __devx_bit_off(typ, fld))

static inline uint64_t _devx_get64(const void *p, size_t bit_off)
{
	return be64toh(*((const __be64 *)(p) + __devx_64_off(bit_off)));
}

#define DEVX_GET64(typ, p, fld) _devx_get64(p, __devx_bit_off(typ, fld))

#define DEVX_ARRAY_SET64(typ, p, fld, idx, v) do { \
	DEVX_SET64(typ, p, fld[idx], v); \
} while (0)

} /* extern "C" */

#endif  // LIBRARY_SRC_GDA_MLX5DV_CORE_HPP_
