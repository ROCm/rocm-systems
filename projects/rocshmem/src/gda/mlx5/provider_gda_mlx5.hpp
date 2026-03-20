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

#ifndef LIBRARY_SRC_GDA_MLX5_GDA_PROVIDER_HPP_
#define LIBRARY_SRC_GDA_MLX5_GDA_PROVIDER_HPP_

#include "gda/endian.hpp"
#include "gda/mlx5/mlx5dv_core.hpp"
#include "gda/mlx5/mlx5_ifc_core.hpp"

namespace rocshmem {

union gda_mlx5_wqe_segment {
  mlx5_wqe_ctrl_seg     ctrl;
  mlx5_wqe_raddr_seg    raddr;
  mlx5_wqe_data_seg     data;
  mlx5_wqe_inl_data_seg inl_data;
  mlx5_wqe_atomic_seg   atomic;
};
static_assert(sizeof(gda_mlx5_wqe_segment) == 16, "mlx5 WQE segments are 16 bytes");

struct gda_mlx5_wqe_ctrl : public mlx5_wqe_ctrl_seg {
  __device__ constexpr inline gda_mlx5_wqe_ctrl(
      uint16_t wqe_idx, uint8_t opcode,
      uint32_t /* 24-bit */ qpn, uint8_t /* 6-bit */ ds,
      uint8_t /* [7:5 fm|4|3:2 ce|1 se|0] */ fm_ce_se)
    : mlx5_wqe_ctrl_seg{
        .opmod_idx_opcode = endian::to_be<uint32_t>(
            (static_cast<uint32_t>(wqe_idx) << 8) | static_cast<uint32_t>(opcode)),
        .qpn_ds = endian::to_be<uint32_t>((qpn << 8) | static_cast<uint32_t>(ds & 0x3F)),
        .signature = 0,
        .dci_stream_channel_id = 0,
        .fm_ce_se = fm_ce_se,
        .imm = 0,
      } { }
} __attribute__((__packed__)) __attribute__((__aligned__(16)));

struct gda_mlx5_wqe_raddr : public mlx5_wqe_raddr_seg {
  __device__ constexpr inline gda_mlx5_wqe_raddr(uintptr_t raddr, __be32 rkey)
    : mlx5_wqe_raddr_seg{
        .raddr = endian::to_be<uint64_t>(raddr),
        .rkey = rkey,
        .reserved = 0,
      } { }
} __attribute__((__packed__)) __attribute__((__aligned__(16)));

struct gda_mlx5_wqe_data : public mlx5_wqe_data_seg {
  __device__ constexpr inline gda_mlx5_wqe_data(
      uintptr_t addr, __be32 lkey, uint32_t byte_count)
    : mlx5_wqe_data_seg{
        .byte_count = endian::to_be<uint32_t>(byte_count & ~MLX5_INLINE_SEG),
        .lkey = lkey,
        .addr = endian::to_be<uint64_t>(addr),
      } { }
} __attribute__((__packed__)) __attribute__((__aligned__(16)));

struct gda_mlx5_wqe_atomic : public mlx5_wqe_atomic_seg {
  __device__ constexpr inline gda_mlx5_wqe_atomic(
      uint64_t swap_add, uint64_t compare)
    : mlx5_wqe_atomic_seg{
        .swap_add = endian::to_be<uint64_t>(swap_add),
        .compare = endian::to_be<uint64_t>(compare),
      } { }
} __attribute__((__packed__)) __attribute__((__aligned__(16)));

// use up to two segments for mlx5 inline RMA WQEs, since we have space in the 64B WQEBB anyway
struct gda_mlx5_wqe_inline_data {
  __be32  byte_count; // only first 10 bits are valid, MSB must be 1
  uint8_t data[28];   // when byte_count > 12, set ctrl.ds = 4

  __device__ constexpr inline gda_mlx5_wqe_inline_data(
      uintptr_t addr, uint32_t /* 10-bit */ byte_count)
    : byte_count{endian::to_be<uint32_t>((byte_count & 0x3FF) | MLX5_INLINE_SEG)}, data{} {
    memcpy(&this->data[0], reinterpret_cast<void*>(addr), byte_count);
  }
} __attribute__((__packed__)) __attribute__((__aligned__(16)));
static_assert(sizeof(gda_mlx5_wqe_inline_data) == sizeof(gda_mlx5_wqe_segment[2]),
              "mlx5 RMA WQEs have up to 2 inline data segments");

// RMA WQEs have either a 16B indirect data segment or an inline data segment with up to 28B data
union gda_mlx5_wqe_rma {
  gda_mlx5_wqe_data        data;
  gda_mlx5_wqe_inline_data inline_data;

  __device__ constexpr inline gda_mlx5_wqe_rma(
      uintptr_t laddr, __be32 lkey, uint32_t byte_count)
    : data{laddr, lkey, byte_count} { }

  __device__ constexpr inline gda_mlx5_wqe_rma(
      uintptr_t laddr, uint32_t byte_count)
    : inline_data{laddr, byte_count} { }

  __device__ static constexpr inline uint8_t ds{3};

  __device__ static constexpr inline uint8_t inline_ds(uint32_t byte_count) {
    return (byte_count <= 12 ? 3 : 4);
  }

  __device__ static constexpr inline bool can_inline(
      uint8_t opcode, uint32_t byte_count, uint32_t inline_threshold) {
    return (opcode == MLX5_OPCODE_RDMA_WRITE) && (byte_count <= inline_threshold);
  }
} __attribute__((__packed__)) __attribute__((__aligned__(16)));

// AMO WQEs have a 16B atomic segment and an (optional?) indirect data segment for fetching atomics
struct gda_mlx5_wqe_amo {
  gda_mlx5_wqe_atomic atomic;
  gda_mlx5_wqe_data   fetch;  // for fetching atomics, set ctrl.ds = 4

  __device__ constexpr inline gda_mlx5_wqe_amo(
      uint64_t swap_add, uint64_t compare,
      uintptr_t laddr, __be32 lkey)
    : atomic{swap_add, compare},
      fetch{laddr, lkey, 8} { }

  __device__ static constexpr inline uint8_t ds{4};
} __attribute__((__packed__)) __attribute__((__aligned__(16)));

// WQEs have a 16B control segment, 16B remote address segment, and one or two additional 16B segments
struct gda_mlx5_wqe {
  gda_mlx5_wqe_ctrl  ctrl;
  gda_mlx5_wqe_raddr raddr;
  union {
    gda_mlx5_wqe_rma rma;
    gda_mlx5_wqe_amo amo;
  };

  // RMA, memory pointer
  __device__ constexpr inline gda_mlx5_wqe(
      uint16_t wqe_idx, uint8_t opcode, uint32_t qpn, uint8_t fm_ce_se,
      uintptr_t raddr, __be32 rkey,
      uintptr_t laddr, __be32 lkey, uint32_t byte_count)
    : ctrl{wqe_idx, opcode, qpn, gda_mlx5_wqe_rma::ds, fm_ce_se},
      raddr{raddr, rkey},
      rma{laddr, lkey, byte_count} { }

  // RMA, inline
  __device__ constexpr inline gda_mlx5_wqe(
      uint16_t wqe_idx, uint8_t opcode, uint32_t qpn, uint8_t fm_ce_se,
      uintptr_t raddr, __be32 rkey,
      uintptr_t laddr, uint32_t byte_count)
    : ctrl{wqe_idx, opcode, qpn, gda_mlx5_wqe_rma::inline_ds(byte_count), fm_ce_se},
      raddr{raddr, rkey},
      rma{laddr, byte_count} { }

  // AMO
  __device__ constexpr inline gda_mlx5_wqe(
      uint16_t wqe_idx, uint8_t opcode, uint32_t qpn, uint8_t fm_ce_se,
      uintptr_t raddr, __be32 rkey,
      uint64_t swap_add, uint64_t compare,
      uintptr_t laddr, __be32 lkey)
    : ctrl{wqe_idx, opcode, qpn, gda_mlx5_wqe_amo::ds, fm_ce_se},
      raddr{raddr, rkey},
      amo{swap_add, compare, laddr, lkey} { }

private:
  // return by value to trigger copy elision/prvalue semantics
  __device__ static constexpr inline gda_mlx5_wqe gda_mlx5_wqe_init_inline_rma(
      uint16_t wqe_idx, uint8_t opcode, uint32_t qpn, uint8_t fm_ce_se,
      uintptr_t raddr, __be32 rkey,
      uintptr_t laddr, __be32 lkey, uint32_t byte_count,
      bool send_inline) {
    // return braced-init-list to directly initialize the returned object
    if (send_inline) {
      return {wqe_idx, opcode, qpn, fm_ce_se, raddr, rkey, laddr, byte_count};
    } else {
      return {wqe_idx, opcode, qpn, fm_ce_se, raddr, rkey, laddr, lkey, byte_count};
    }
  }

public:
  // use gda_mlx5_wqe_init_inline_rma to construct either the inline or memory pointer RMA WQE
  __device__ constexpr inline gda_mlx5_wqe(
      uint16_t wqe_idx, uint8_t opcode, uint32_t qpn, uint8_t fm_ce_se,
      uintptr_t raddr, __be32 rkey,
      uintptr_t laddr, __be32 lkey, uint32_t byte_count,
      bool send_inline)
    : gda_mlx5_wqe{gda_mlx5_wqe_init_inline_rma(
        wqe_idx, opcode, qpn, fm_ce_se,
        raddr, rkey,
        laddr, lkey, byte_count,
        send_inline)} { }
} __attribute__((__aligned__(64)));

static_assert(sizeof(gda_mlx5_wqe) == sizeof(gda_mlx5_wqe_segment[4]),
              "mlx5 WQEs have up to 4 segments");

// WQE header is the first two 4-byte fields in the WQE control segment
struct gda_mlx5_wqe_header {
  __be32 opmod_idx_opcode;
  __be32 qpn_ds;
} __attribute__((__packed__)) __attribute__((__aligned__(8)));

// doorbell value is WQE header
union gda_mlx5_db_register {
  uint64_t            val;
  gda_mlx5_wqe_header wqe_header;

  __device__ constexpr inline gda_mlx5_db_register(uint64_t val) : val{val} { }
  __device__ constexpr inline gda_mlx5_db_register(const gda_mlx5_wqe_header& wqe_header)
    : wqe_header{wqe_header} { }
  __device__ constexpr inline gda_mlx5_db_register(__be32 opmod_idx_opcode, __be32 qpn_ds)
    : val{(static_cast<uint64_t>(qpn_ds) << 32) | static_cast<uint64_t>(opmod_idx_opcode)} { }
  __device__ constexpr inline gda_mlx5_db_register(const gda_mlx5_wqe_ctrl& ctrl)
    : gda_mlx5_db_register{gda_mlx5_wqe_header{ctrl.opmod_idx_opcode, ctrl.qpn_ds}} { }
  __device__ constexpr inline gda_mlx5_db_register(const gda_mlx5_wqe& wqe)
    : gda_mlx5_db_register{wqe.ctrl} { }
} __attribute__((__packed__)) __attribute__((__aligned__(8)));
static_assert(sizeof(gda_mlx5_db_register) == sizeof(uint64_t),
              "mlx5 doorbells registers must be written as an atomic 64-bit write");

/* BlueFlame buffer size should technically be checked, but it seems to always be 256B
 * BlueFlame buffer "must be written in chunks of DWORDs or multiple DWORDs"
 * WQEs need to be aligned to WQEBB (64B), which implies that the BlueFlame buffer is as well?
 * given that a UAR page is 4 KiB and that the array of BlueFlame buffers
 * begins at offset 0x800 in the UAR page, BlueFlame buffers *should* be aligned to 256 bytes
 * first 8 bytes is the doorbell register */
union gda_mlx5_bf_buffer {
  gda_mlx5_db_register db_reg;
  uint8_t              data[MLX5_DB_BLUEFLAME_BUFFER_SIZE];
  __be32               dword[MLX5_DB_BLUEFLAME_BUFFER_SIZE / sizeof(__be32)];
} __attribute__((__packed__)) __attribute__((__aligned__(alignof(gda_mlx5_wqe))));
static_assert(sizeof(gda_mlx5_bf_buffer) == MLX5_DB_BLUEFLAME_BUFFER_SIZE,
              "mlx5 BlueFlame buffers are 256 bytes");

/* BlueFlame buffers are paired into two halves that should be written alternately
 * This only matters when using the BlueFlame mechanism, which requires Write Combining
 * If just ringing the doorbell, we can write to the same half */
struct gda_mlx5_doorbell {
  gda_mlx5_bf_buffer bf[1];
} __attribute__((__packed__)) __attribute__((__aligned__(alignof(gda_mlx5_bf_buffer))));
static_assert(sizeof(gda_mlx5_doorbell) == sizeof(gda_mlx5_bf_buffer[1]),
              "mlx5 non-cached doorbells are a single BlueFlame buffer");

template <typename T>
struct gda_mlx5_device_queue {
  T* buf;
  __be32* dbrec;
  uint32_t lock;

  __host__ inline gda_mlx5_device_queue(T* buf, __be32* dbrec)
    : buf{buf}, dbrec{dbrec}, lock{0} { }

  __host__ inline gda_mlx5_device_queue() : gda_mlx5_device_queue{nullptr, nullptr} { }
};

struct gda_mlx5_device_cq : public gda_mlx5_device_queue<mlx5_cqe64> {
  // inherit constructors
  using gda_mlx5_device_queue::gda_mlx5_device_queue;
};

struct gda_mlx5_device_sq : public gda_mlx5_device_queue<gda_mlx5_wqe> {
  gda_mlx5_doorbell* db;
  uint64_t post;
  uint16_t depth;
  uint16_t tail;

  __host__ inline gda_mlx5_device_sq(gda_mlx5_wqe* buf, __be32* dbrec,
                                     gda_mlx5_doorbell* db, uint16_t depth)
    : gda_mlx5_device_queue{buf, dbrec},
      db{db}, post{0}, depth{depth}, tail{0} { }

  __host__ inline gda_mlx5_device_sq() : gda_mlx5_device_sq{nullptr, nullptr, nullptr, 0} { }

  __device__ inline gda_mlx5_bf_buffer* bf_buffer() {
    return &db->bf[0];
  }
};

/*
 * QP layout:
 * [ [ WQ: WQE | WQE | WQE | ... | WQE ] : 64 * 2^N
 *   [ CQ: CQE | CQE ]                   : 64 * 2
 *   [ padding - to page table - 256   ] : 3712 - max(4096 - 64 * 2^N, 0)
 *   [ SQ DBREC ]                        : 8
 *   [ padding - to AMDGPU cache line  ] : 120
 *   [ CQ DBREC ]                        : 8
 *   [ padding - to AMDGPU cache line  ] : 120
 * ]
 * Total size: 4096 if sq_depth < 64, else 4096 + 64 * 2^N
 */
struct mlx5_devx_qp {
  ibv_context*      ctx;
  mlx5dv_devx_obj*  devx_cq_obj;
  mlx5dv_devx_obj*  devx_qp_obj;
  mlx5dv_devx_uar*  uar;
  mlx5dv_devx_umem* umem;
  void*             cq;
  void*             sq;
  uint32_t*         cq_dbrec;
  uint32_t*         qp_dbrec;
  uint32_t          cqn;
  uint32_t          qpn;
  uint16_t          sq_depth;

  void dump(int conn_num);

  inline size_t cq_offset() {
    return reinterpret_cast<uint8_t*>(cq) - reinterpret_cast<uint8_t*>(sq);
  }
  inline size_t wq_offset() {
    return reinterpret_cast<uint8_t*>(sq) - reinterpret_cast<uint8_t*>(sq);
  }
  inline size_t cq_dbrec_offset() {
    return reinterpret_cast<uint8_t*>(cq_dbrec) - reinterpret_cast<uint8_t*>(sq);
  }
  inline size_t qp_dbrec_offset() {
    return reinterpret_cast<uint8_t*>(qp_dbrec) - reinterpret_cast<uint8_t*>(sq);
  }
};

struct mlx5dv_funcs_t {
  int (*init_obj)(struct mlx5dv_obj *obj, uint64_t obj_type);
  struct ibv_context * (*open_device)(struct ibv_device *device, struct mlx5dv_context_attr *attr);
  struct mlx5dv_devx_obj * (*devx_obj_create)(
      struct ibv_context *context, const void *in, size_t inlen, void *out, size_t outlen);
  int (*devx_obj_modify)(
      struct mlx5dv_devx_obj *obj, const void *in, size_t inlen, void *out, size_t outlen);
  int (*devx_obj_destroy)(struct mlx5dv_devx_obj *obj);
  struct mlx5dv_devx_umem * (*devx_umem_reg_ex)(
      struct ibv_context *ctx, struct mlx5dv_devx_umem_in *umem_in);
  int (*devx_umem_dereg)(struct mlx5dv_devx_umem *umem);
  struct mlx5dv_devx_uar * (*devx_alloc_uar)(struct ibv_context *context, uint32_t flags);
  void (*devx_free_uar)(struct mlx5dv_devx_uar *devx_uar);
  int (*devx_query_eqn)(struct ibv_context *context, uint32_t vector, uint32_t *eqn);

  int create_qp(mlx5_devx_qp& qp, struct ibv_context *ctx, struct ibv_pd* pd, uint16_t sq_depth);
  int modify_qp(mlx5_devx_qp& qp, struct ibv_qp_attr *attr, int attr_mask, uint32_t gid_type);
  int destroy_qp(mlx5_devx_qp& qp);
};

}  // namespace rocshmem

#endif  //LIBRARY_SRC_GDA_MLX5_GDA_PROVIDER_HPP_
