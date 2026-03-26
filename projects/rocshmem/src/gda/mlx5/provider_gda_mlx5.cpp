/******************************************************************************
 * Copyright (c) Advanced Micro Devices, Inc. All rights reserved.
 *
 * SPDX-License-Identifier: MIT
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to
 * deal in the Software without restriction, including without limitation the
 * rights to use, copy, modify, merge, publish, distribute, sublicense, and/or
 * sell copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
 * FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS
 * IN THE SOFTWARE.
 *****************************************************************************/

#include <limits>
#include <hip/hip_runtime.h>

#include "util.hpp"
#include "gda/bit.hpp"
#include "gda/ibv_wrapper.hpp"
#include "gda/mlx5/provider_gda_mlx5.hpp"
#include "gda/mlx5/mlx5dv_core.hpp"
#include "gda/mlx5/mlx5_ifc_core.hpp"

namespace rocshmem {

using create_qp_in    = uint8_t[DEVX_ST_SZ_BYTES(create_qp_in)   ];
using create_qp_out   = uint8_t[DEVX_ST_SZ_BYTES(create_qp_out)  ];
using rst2init_qp_in  = uint8_t[DEVX_ST_SZ_BYTES(rst2init_qp_in) ];
using rst2init_qp_out = uint8_t[DEVX_ST_SZ_BYTES(rst2init_qp_out)];
using init2rtr_qp_in  = uint8_t[DEVX_ST_SZ_BYTES(init2rtr_qp_in) ];
using init2rtr_qp_out = uint8_t[DEVX_ST_SZ_BYTES(init2rtr_qp_out)];
using rtr2rts_qp_in   = uint8_t[DEVX_ST_SZ_BYTES(rtr2rts_qp_in)  ];
using rtr2rts_qp_out  = uint8_t[DEVX_ST_SZ_BYTES(rtr2rts_qp_out) ];

static int mlx5_modify_qp_reset2init(const mlx5dv_funcs_t& mlx5dv, mlx5_devx_qp* qp,
                                     struct ibv_qp_attr* attr, int attr_mask);
static int mlx5_modify_qp_init2rtr(const mlx5dv_funcs_t& mlx5dv, mlx5_devx_qp* qp,
                                   struct ibv_qp_attr* attr, int attr_mask, uint32_t gid_type);
static int mlx5_modify_qp_rtr2rts(const mlx5dv_funcs_t& mlx5dv, mlx5_devx_qp* qp,
                                  struct ibv_qp_attr* attr, int attr_mask);

struct hipGDAHostAllocator {
  static void* malloc(size_t size) {
    void* dev_ptr{nullptr};
    CHECK_HIP(hipHostMalloc(&dev_ptr, size, hipHostMallocDefault));
    memset(dev_ptr, 0, size);
    return dev_ptr;
  }

  static void free(void* ptr) {
    CHECK_HIP(hipFree(ptr));
  }
};

struct hipGDAUncachedAllocator {
  static void* malloc(size_t size) {
    void* dev_ptr{nullptr};
    CHECK_HIP(hipExtMallocWithFlags(&dev_ptr, size, hipDeviceMallocUncached));
    memset(dev_ptr, 0, size);
    return dev_ptr;
  }

  static void free(void* ptr) {
    CHECK_HIP(hipFree(ptr));
  }
};

using QPAllocator = hipGDAHostAllocator;

static inline bool is_device_ptr(const void* ptr) {
  hipPointerAttribute_t ptr_attr;
  CHECK_HIP(hipPointerGetAttributes(&ptr_attr, ptr));
  return ptr_attr.type == hipMemoryTypeDevice;
}

template <typename T>
static constexpr inline size_t mlx5_align_amdgpu_cache_line(const T& value) {
  // different cache levels can have different cache line sizes; varies by GPU arch as well
  // L2$ is 128B on CDNA2, CDNA3, and CDNA4, so use that for now
  constexpr size_t amdgpu_cache_line_size = 128;
  return __builtin_align_up(value, amdgpu_cache_line_size);
}

template <typename T>
static constexpr inline size_t mlx5_align_page(const T& value) {
  // mlx5 HCA counts pages in 4 KiB chunks, but supports larger host page sizes
  // amdgpu page size is not well-documented, but should at least support 4 KiB pages
  constexpr size_t mlx5_page_size = MLX5_ADAPTER_PAGE_SIZE;
  return __builtin_align_up(value, mlx5_page_size);
}

static inline size_t mlx5_calc_qp_umem_size(uint16_t& sq_depth,
                                            size_t& wq_size,
                                            size_t& dbrec_offset) {
  // align doorbell record to cache line size
  constexpr size_t dbrec_size = mlx5_align_amdgpu_cache_line<size_t>(MLX5_DOORBELL_RECORD_SIZE);
  // round up to power of 2 WQEBB
  size_t wq_size_initial = bit_ceil(sq_depth) * MLX5_SEND_WQE_BB;
  // round up to page size
  size_t umem_size = mlx5_align_page(wq_size_initial + dbrec_size);
  // doorbell record in last cache line of umem allocation
  dbrec_offset = umem_size - dbrec_size;
  // round back down to a power of 2
  wq_size = bit_floor(umem_size - dbrec_size);
  // compute WQEBB count
  sq_depth = wq_size / MLX5_SEND_WQE_BB;
  return umem_size;
}

static inline mlx5dv_devx_umem* mlx5_umem_reg(const mlx5dv_funcs_t& mlx5dv,
                                              struct ibv_context *ctx, void* addr, size_t size) {
  int dmabuf_fd = -1;
  uint64_t dmabuf_offset = std::numeric_limits<uint64_t>::max();

  mlx5dv_devx_umem_in umem_in = {0};

  umem_in.addr = addr;
  umem_in.size = size;
  umem_in.access = IBV_ACCESS_LOCAL_WRITE;
  umem_in.pgsz_bitmap = std::numeric_limits<uint64_t>::max() & ~(MLX5_ADAPTER_PAGE_SIZE - 1);

  // to use dmabuf_fd, set comp_mask = MLX5DV_UMEM_MASK_DMABUF
  if (ibv.is_dmabuf_supported() && is_device_ptr(addr)) {
    CHECK_HSA(hsa_amd_portable_export_dmabuf(addr, size, &dmabuf_fd, &dmabuf_offset));
    umem_in.comp_mask = MLX5DV_UMEM_MASK_DMABUF;
    umem_in.dmabuf_fd = dmabuf_fd;
  }

  mlx5dv_devx_umem* umem = mlx5dv.devx_umem_reg_ex(ctx, &umem_in);
  CHECK_NNULL(umem, "mlx5dv_devx_umem_reg_ex");

  // can we close the dmabuf_fd here?
  if (dmabuf_fd != -1) {
    CHECK_HSA(hsa_amd_portable_close_dmabuf(dmabuf_fd));
  }

  return umem;
}

static inline uint32_t mlx5_pdn(const mlx5dv_funcs_t& mlx5dv, struct ibv_pd *pd) {
  mlx5dv_pd mlx5_pd;
  mlx5dv_obj obj{ .pd = { .in = pd, .out = &mlx5_pd } };
  int err = mlx5dv.init_obj(&obj, MLX5DV_OBJ_PD);
  CHECK_ZERO(err, "mlx5dv_init_obj (PD)");
  return mlx5_pd.pdn;
}

static inline uint32_t mlx5_cqn(const mlx5dv_funcs_t& mlx5dv, struct ibv_cq *cq) {
  mlx5dv_cq mlx5_cq;
  mlx5dv_obj obj{ .cq = { .in = cq, .out = &mlx5_cq } };
  int err = mlx5dv.init_obj(&obj, MLX5DV_OBJ_CQ);
  CHECK_ZERO(err, "mlx5dv_init_obj (CQ)");
  return mlx5_cq.cqn;
}

// these are technically identical, but for completeness
static inline uint32_t mlx5_mtu(enum ibv_mtu mtu) {
  switch (mtu) {
  case IBV_MTU_256:
    return MLX5_QPC_MTU_256_BYTES;
  case IBV_MTU_512:
    return MLX5_QPC_MTU_512_BYTES;
  case IBV_MTU_1024:
    return MLX5_QPC_MTU_1K_BYTES;
  case IBV_MTU_2048:
    return MLX5_QPC_MTU_2K_BYTES;
  case IBV_MTU_4096:
    return MLX5_QPC_MTU_4K_BYTES;
  default:
    fprintf(stderr, "Error: invalid ibv_mtu enumerator %u\n", static_cast<uint32_t>(mtu));
    return static_cast<uint32_t>(mtu);
  }
}

/**
 * Linux kernel source drivers/infiniband/hw/mlx5/qp.c
 *
 * ib_to_mlx5_rate_map - map from IB user verbs rate to mlx5 rate
 */
static inline uint8_t mlx5_stat_rate(uint8_t rate) {
  switch (rate) {
  case IBV_RATE_MAX:
    return 0;
  case IBV_RATE_56_GBPS:
    return 1;
  case IBV_RATE_25_GBPS:
    return 2;
  case IBV_RATE_100_GBPS:
    return 3;
  case IBV_RATE_200_GBPS:
    return 4;
  case IBV_RATE_50_GBPS:
    return 5;
  case IBV_RATE_400_GBPS:
    return 6;
  default:
    return rate + MLX5_STAT_RATE_OFFSET;
  }
}

/**
 * Linux kernel source include/rdma/ib_verbs.h
 *
 * rdma_calc_flow_label - generate a RDMA symmetric flow label value based on
 *                        local and remote qpn values
 *
 * This function folded the multiplication results of two qpns, 24 bit each,
 * fields, and converts it to a 20 bit results.
 *
 * This function will create symmetric flow_label value based on the local
 * and remote qpn values. this will allow both the requester and responder
 * to calculate the same flow_label for a given connection.
 *
 * This helper function should be used by driver in case the upper layer
 * provide a zero flow_label value. This is to improve entropy of RDMA
 * traffic in the network.
 */
static inline uint32_t mlx5_calc_flow_label(uint32_t local_qpn, uint32_t remote_qpn) {
  uint64_t v = static_cast<uint64_t>(local_qpn) * static_cast<uint64_t>(remote_qpn);
  v ^= v >> 20;
  v ^= v >> 40;
  return static_cast<uint32_t>(v & IB_GRH_FLOWLABEL_MASK);
}

int mlx5_devx_qp::create(const mlx5dv_funcs_t& mlx5dv, struct ibv_context *ctx,
                         struct ibv_qp_init_attr_ex *attr) {
  assert(attr->cap.max_send_wr > 0 && "ibv_qp_init_attr_ex::cap::max_send_wr is 0");
  create_qp_in  in  = {0};
  create_qp_out out = {0};

  void* qpc = DEVX_ADDR_OF(create_qp_in, in, qpc);

  uint32_t pdn = mlx5_pdn(mlx5dv, attr->pd);
  uint32_t cqn = mlx5_cqn(mlx5dv, attr->send_cq);

  // calculate buffer size needed for SQ + dbrec
  uint16_t sq_depth = attr->cap.max_send_wr;
  size_t wq_size = 0;
  size_t dbrec_offset = 0;
  size_t umem_size = mlx5_calc_qp_umem_size(sq_depth, wq_size, dbrec_offset);

  // allocate SQ + dbrec buffer
  void* umem_buffer = QPAllocator::malloc(umem_size);
  // register SQ + dbrec buffer
  mlx5dv_devx_umem* umem = mlx5_umem_reg(mlx5dv, ctx, umem_buffer, umem_size);

  /* allocate UAR
   *
   * MLX5DV_UAR_ALLOC_TYPE_NC always returns the same singleton UAR, but this causes problems
   * when multiple threads in a wave write to the same doorbell address concurrently
   * MLX5DV_UAR_ALLOC_TYPE_NC_DEDICATED dynamically allocates a UAR page, but these are limited
   * using MLX5DV_UAR_ALLOC_TYPE_NC_DEDICATED requires rdma-core v45 or later (released March 2023)
   * see https://github.com/linux-rdma/rdma-core/commit/bf550b9fa83374cfed51330760a583d82a7600f4 */
  mlx5dv_devx_uar* uar = mlx5dv.devx_alloc_uar(ctx, MLX5DV_UAR_ALLOC_TYPE_NC_DEDICATED);
  CHECK_NNULL(uar, "mlx5dv_devx_alloc_uar");

  DEVX_SET(create_qp_in, in, opcode, MLX5_CMD_OP_CREATE_QP);

  DEVX_SET(qpc, qpc, st,          MLX5_QPC_ST_RC);
  DEVX_SET(qpc, qpc, pm_state,    MLX5_QPC_PM_STATE_MIGRATED);
  DEVX_SET(qpc, qpc, pd,          pdn);
  DEVX_SET(qpc, qpc, uar_page,    uar->page_id);
  DEVX_SET(qpc, qpc, log_sq_size, bit_log2(sq_depth));
  DEVX_SET(qpc, qpc, rq_type,     MLX5_QPC_RQ_TYPE_ZERO_LEN_RQ);
  DEVX_SET(qpc, qpc, cqn_snd,     cqn);

  // disable scatter CQE to requester, responder
  DEVX_SET(qpc, qpc, cs_req, MLX5_QPC_CS_REQ_DISABLE);
  DEVX_SET(qpc, qpc, cs_res, MLX5_QPC_CS_RES_DISABLE);

  // set dbrec umem attributes, dbr_addr is offset into umem when umem_valid
  DEVX_SET64(qpc, qpc, dbr_addr,       dbrec_offset);
  DEVX_SET  (qpc, qpc, dbr_umem_valid, true);
  DEVX_SET  (qpc, qpc, dbr_umem_id,    umem->umem_id);

  // set WQ umem attributes, SQ is at the start of the umem buffer
  DEVX_SET64(create_qp_in, in, wq_umem_offset, 0);
  DEVX_SET(  create_qp_in, in, wq_umem_id,     umem->umem_id);
  DEVX_SET(  create_qp_in, in, wq_umem_valid,  true);

  // create QP
  mlx5dv_devx_obj* devx_obj = mlx5dv.devx_obj_create(ctx, in, sizeof(in), out, sizeof(out));
  CHECK_NNULL(devx_obj, "mlx5dv_devx_obj_create (QP)");

  // extract QP number
  uint32_t qpn = DEVX_GET(create_qp_out, out, qpn);

  // set the object's fields
  this->ctx       = ctx;
  this->devx_obj  = devx_obj;
  this->uar       = uar;
  this->umem      = umem;
  this->sq        = umem_buffer;
  this->dbrec     = reinterpret_cast<uint32_t*>(
      reinterpret_cast<uint8_t*>(umem_buffer) + dbrec_offset);
  this->qpn       = qpn;
  this->sq_depth  = sq_depth;

  return 0;
}

int mlx5_devx_qp::modify(const mlx5dv_funcs_t& mlx5dv, struct ibv_qp_attr *attr, int attr_mask,
                         uint32_t gid_type) {
  // check attr->qp_state is valid
  if (!(attr_mask & IBV_QP_STATE)) {
    assert(false && "invalid attr_mask");
    return EINVAL;
  }

  // assume QP is in correct state for transition
  switch (attr->qp_state) {
  case IBV_QPS_INIT:
    return mlx5_modify_qp_reset2init(mlx5dv, this, attr, attr_mask);
  case IBV_QPS_RTR:
    return mlx5_modify_qp_init2rtr(mlx5dv, this, attr, attr_mask, gid_type);
  case IBV_QPS_RTS:
    return mlx5_modify_qp_rtr2rts(mlx5dv, this, attr, attr_mask);
  case IBV_QPS_RESET:
  case IBV_QPS_SQD:
  case IBV_QPS_SQE:
  case IBV_QPS_ERR:
    assert(false && "unsupported mlx5 QP state transition");
    return EOPNOTSUPP;
  default:
    assert(false && "invalid QP state transition");
    return EINVAL;
  }
}

static int mlx5_modify_qp_reset2init(const mlx5dv_funcs_t& mlx5dv, mlx5_devx_qp* qp,
                                     struct ibv_qp_attr* attr, int attr_mask) {
  // man 3 ibv_modify_qp
  constexpr int required_attr_mask = IBV_QP_STATE | IBV_QP_PKEY_INDEX | IBV_QP_PORT |
                                     IBV_QP_ACCESS_FLAGS;
  constexpr unsigned int access_flags = IBV_ACCESS_LOCAL_WRITE | IBV_ACCESS_REMOTE_READ |
                                        IBV_ACCESS_REMOTE_WRITE | IBV_ACCESS_REMOTE_ATOMIC;
  assert((attr_mask & required_attr_mask) == required_attr_mask && "missing required attr");
  assert((attr->qp_access_flags & access_flags) == access_flags && "missing access flags");

  rst2init_qp_in  in  = {0};
  rst2init_qp_out out = {0};

  void* qpc          = DEVX_ADDR_OF(rst2init_qp_in, in,  qpc);
  void* primary_addr = DEVX_ADDR_OF(qpc,            qpc, primary_address_path);

  DEVX_SET(rst2init_qp_in, in, opcode, MLX5_CMD_OP_RST2INIT_QP);
  DEVX_SET(rst2init_qp_in, in, qpn,    qp->qpn);

  DEVX_SET(qpc, qpc, pm_state,    MLX5_QPC_PM_STATE_MIGRATED);
  DEVX_SET(qpc, qpc, atomic_mode, MLX5_QPC_ATOMIC_MODE_UP_TO_8B);
  DEVX_SET(qpc, qpc, rre,         true); // IBV_ACCESS_REMOTE_READ
  DEVX_SET(qpc, qpc, rwe,         true); // IBV_ACCESS_REMOTE_WRITE
  DEVX_SET(qpc, qpc, rae,         true); // IBV_ACCESS_REMOTE_ATOMIC

  DEVX_SET(ads, primary_addr, vhca_port_num, attr->port_num);
  DEVX_SET(ads, primary_addr, pkey_index,    attr->pkey_index);

  return mlx5dv.devx_obj_modify(qp->devx_obj, in, sizeof(in), out, sizeof(out));
}

static int mlx5_modify_qp_init2rtr(const mlx5dv_funcs_t& mlx5dv, mlx5_devx_qp* qp,
                                   struct ibv_qp_attr* attr, int attr_mask, uint32_t gid_type) {
  // man 3 ibv_modify_qp
  constexpr int required_attr_mask = IBV_QP_STATE | IBV_QP_AV | IBV_QP_PATH_MTU | IBV_QP_DEST_QPN |
                                     IBV_QP_RQ_PSN | IBV_QP_MAX_DEST_RD_ATOMIC |
                                     IBV_QP_MIN_RNR_TIMER;
  assert((attr_mask & required_attr_mask) == required_attr_mask && "missing required attr");
  assert(attr->max_dest_rd_atomic > 0 && "ibv_qp_attr::max_dest_rd_atomic is 0");

  init2rtr_qp_in  in  = {0};
  init2rtr_qp_out out = {0};

  void* qpc          = DEVX_ADDR_OF(init2rtr_qp_in, in, qpc);
  void* primary_addr = DEVX_ADDR_OF(qpc, qpc, primary_address_path);
  void* rgid_rip     = DEVX_ADDR_OF(ads, primary_addr, rgid_rip);
  void* rmac         = DEVX_ADDR_OF(ads, primary_addr, rmac_47_32);

  DEVX_SET(init2rtr_qp_in, in, opcode, MLX5_CMD_OP_INIT2RTR_QP);
  DEVX_SET(init2rtr_qp_in, in, qpn,    qp->qpn);

  DEVX_SET(qpc, qpc, mtu,          mlx5_mtu(attr->path_mtu));
  DEVX_SET(qpc, qpc, log_msg_max,  MLX5_QPC_LOG_MSG_SZ_MAX);
  DEVX_SET(qpc, qpc, remote_qpn,   attr->dest_qp_num);
  DEVX_SET(qpc, qpc, log_rra_max,  bit_log2(attr->max_dest_rd_atomic));
  DEVX_SET(qpc, qpc, min_rnr_nak,  attr->min_rnr_timer);
  DEVX_SET(qpc, qpc, next_rcv_psn, attr->rq_psn);

  struct ibv_ah_attr* ah_attr = &attr->ah_attr;
  struct ibv_global_route* grh = &ah_attr->grh;

  // calculate new flow label if not set
  uint32_t flow_label = grh->flow_label ? grh->flow_label
                                        : mlx5_calc_flow_label(qp->qpn, attr->dest_qp_num);

  // all transports
  DEVX_SET(ads, primary_addr, stat_rate, mlx5_stat_rate(ah_attr->static_rate));

  // if there's a GRH: RoCE v1, RoCE v2, and IB + GRH
  if (ah_attr->is_global) {
    DEVX_SET(ads, primary_addr, src_addr_index, grh->sgid_index);
    DEVX_SET(ads, primary_addr, hop_limit,      grh->hop_limit);
    DEVX_SET(ads, primary_addr, tclass,         grh->traffic_class);
    DEVX_SET(ads, primary_addr, flow_label,     flow_label);
    // remote GID/IP gets copied directly
    memcpy(rgid_rip, grh->dgid.raw, sizeof(grh->dgid.raw));
  }

  // InfiniBand
  if (gid_type == IBV_GID_TYPE_IB) {
    // IB + GRH needs to set GRH bit
    DEVX_SET(ads, primary_addr, grh,  static_cast<bool>(ah_attr->is_global));
    DEVX_SET(ads, primary_addr, mlid, ah_attr->src_path_bits);
    DEVX_SET(ads, primary_addr, rlid, ah_attr->dlid);
    DEVX_SET(ads, primary_addr, sl,   ah_attr->sl);
  }

  // RoCE v1 and RoCE v2
  if (gid_type == IBV_GID_TYPE_ROCE_V1 ||
      gid_type == IBV_GID_TYPE_ROCE_V2) {
    assert(ah_attr->is_global && "ibv_qp_attr::ah_attr::is_global not set, but gid_type is RoCE");
    // get remote MAC address
    uint8_t remote_mac[ETHERNET_LL_SIZE];
    int err = ibv.resolve_eth_l2_from_gid(qp->ctx, ah_attr, remote_mac, /* VLAN id */ nullptr);
    CHECK_ZERO(err, "ibv_resolve_eth_l2_from_gid");

    DEVX_SET(ads, primary_addr, eth_prio, ah_attr->sl);
    // remote MAC address gets copied directly
    memcpy(rmac, &remote_mac, sizeof(remote_mac));
  }

  // RoCE v2
  if (gid_type == IBV_GID_TYPE_ROCE_V2) {
    DEVX_SET(ads, primary_addr, dscp,      grh->traffic_class >> 2);
    DEVX_SET(ads, primary_addr, udp_sport, ibv.flow_label_to_udp_sport(flow_label));
  }

  return mlx5dv.devx_obj_modify(qp->devx_obj, in, sizeof(in), out, sizeof(out));
}

static int mlx5_modify_qp_rtr2rts(const mlx5dv_funcs_t& mlx5dv, mlx5_devx_qp* qp,
                                  struct ibv_qp_attr* attr, int attr_mask) {
  // man 3 ibv_modify_qp
  constexpr int required_attr_mask = IBV_QP_STATE | IBV_QP_SQ_PSN | IBV_QP_MAX_QP_RD_ATOMIC |
                                     IBV_QP_RETRY_CNT | IBV_QP_RNR_RETRY | IBV_QP_TIMEOUT;
  assert((attr_mask & required_attr_mask) == required_attr_mask && "missing required attr");
  assert(attr->max_rd_atomic > 0 && "ibv_qp_attr::max_rd_atomic is 0");

  rtr2rts_qp_in  in  = {0};
  rtr2rts_qp_out out = {0};

  void* qpc          = DEVX_ADDR_OF(rtr2rts_qp_in, in, qpc);
  void* primary_addr = DEVX_ADDR_OF(qpc, qpc, primary_address_path);

  DEVX_SET(rtr2rts_qp_in, in, opcode, MLX5_CMD_OP_RTR2RTS_QP);
  DEVX_SET(rtr2rts_qp_in, in, qpn,    qp->qpn);

  DEVX_SET(qpc, qpc, log_ack_req_freq, MLX5_QPC_LOG_ACK_REQ_FREQ_MAX);
  DEVX_SET(qpc, qpc, log_sra_max,      bit_log2(attr->max_rd_atomic));
  DEVX_SET(qpc, qpc, next_send_psn,    attr->sq_psn);
  DEVX_SET(qpc, qpc, retry_count,      attr->retry_cnt);
  DEVX_SET(qpc, qpc, rnr_retry,        attr->rnr_retry);

  DEVX_SET(ads, primary_addr, ack_timeout, attr->timeout);

  return mlx5dv.devx_obj_modify(qp->devx_obj, in, sizeof(in), out, sizeof(out));
}

int mlx5_devx_qp::destroy(const mlx5dv_funcs_t& mlx5dv) {
  int err = 0;

  err = mlx5dv.devx_obj_destroy(this->devx_obj);
  CHECK_ZERO(err, "mlx5dv_devx_obj_destroy (QP)");

  // mlx5dv_devx_free_uar returns void, can't check for errors
  mlx5dv.devx_free_uar(this->uar);

  err = mlx5dv.devx_umem_dereg(this->umem);
  CHECK_ZERO(err, "mlx5dv_devx_umem_dereg");

  QPAllocator::free(this->sq);

  // clear the object's fields
  this->ctx       = nullptr;
  this->devx_obj  = nullptr;
  this->uar       = nullptr;
  this->umem      = nullptr;
  this->sq        = nullptr;
  this->dbrec     = nullptr;
  this->qpn       = 0;
  this->sq_depth  = 0;

  return err;
}

void mlx5_devx_qp::dump(int conn_num) {
  DPRINTF("\n");
  DPRINTF("===============================================\n");
  DPRINTF("     INITIALIZED MLX5_DEVX_QP FOR CONNECTION#%d\n", conn_num);
  DPRINTF("===============================================\n");
  DPRINTF("=================== QP_DUMP ===================\n");
  DPRINTF("  (uint32_t)  qpn              = 0x%x\n",  this->qpn);
  DPRINTF("  (void*)     sq               = %p\n",    this->sq);
  DPRINTF("  (uint16_t)  sq_depth         = %hu\n",   this->sq_depth);
  DPRINTF("  (uint32_t*) dbrec            = %p\n",    this->dbrec);
  DPRINTF("  (void*)     uar->reg_addr    = %p\n",    this->uar->reg_addr);
  DPRINTF("  (void*)     uar->base_addr   = %p\n",    this->uar->base_addr);
  DPRINTF("  (uint32_t)  uar->page_id     = 0x%x\n",  this->uar->page_id);
  DPRINTF("  (off_t)     uar->mmap_offset = 0x%lx\n", this->uar->mmap_off);
  DPRINTF("  (uint64_t)  uar->comp_mask   = 0x%lx\n", this->uar->comp_mask);
  DPRINTF("================== QP_DUMP_END ================\n");
}

}  // namespace rocshmem
