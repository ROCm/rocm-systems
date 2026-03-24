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

#ifndef _MLX5DV_H_
#define _MLX5DV_H_

#include <sys/types.h>
#include "gda/ibv_core.hpp"

enum {
	MLX5_RCV_DBR	= 0,
	MLX5_SND_DBR	= 1,
};

struct ibv_tmh {
	uint8_t		opcode;      /* from enum ibv_tmh_op */
	uint8_t		reserved[3]; /* must be zero */
	__be32		app_ctx;     /* opaque user data */
	__be64		tag;
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
};

enum mlx5dv_obj_type {
	MLX5DV_OBJ_QP	= 1 << 0,
	MLX5DV_OBJ_CQ	= 1 << 1,
	MLX5DV_OBJ_SRQ	= 1 << 2,
	MLX5DV_OBJ_RWQ	= 1 << 3,
	MLX5DV_OBJ_DM	= 1 << 4,
	MLX5DV_OBJ_AH	= 1 << 5,
	MLX5DV_OBJ_PD	= 1 << 6,
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

#endif /* _MLX5DV_H_ */
