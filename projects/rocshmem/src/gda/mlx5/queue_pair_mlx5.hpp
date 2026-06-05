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

#ifndef LIBRARY_SRC_GDA_MLX5_QUEUE_PAIR_HPP_
#define LIBRARY_SRC_GDA_MLX5_QUEUE_PAIR_HPP_

#include <utility>

#include "gda/endian.hpp"
#include "gda/mlx5/provider_gda_mlx5.hpp"
#include "gda/queue_pair.hpp"

namespace rocshmem {

class QueuePairMLX5;

template <> struct QueuePairTraits<QueuePairMLX5> {
  enum class OpCode : uint8_t {
    RDMA_WRITE = MLX5_OPCODE_RDMA_WRITE,
    RDMA_READ  = MLX5_OPCODE_RDMA_READ,
    ATOMIC_CS  = MLX5_OPCODE_ATOMIC_CS,
    ATOMIC_FA  = MLX5_OPCODE_ATOMIC_FA,
  };

  /**
   * @brief mlx5 uses big-endian ordering
   */
  static constexpr endian::Order Endianness = endian::Order::Big;
};

class QueuePairMLX5 : public QueuePairBase<QueuePairMLX5> {
public:
  __host__ explicit QueuePairMLX5(uint32_t qpn, void *base_heap, size_t heap_size,
                                  uint32_t lkey, uint32_t rkey, struct ibv_pd* pd,
                                  gda_mlx5_device_sq&& sq, gda_mlx5_device_cq&& cq);

  __host__ QueuePairMLX5(const QueuePairMLX5& other)            = delete;
  __host__ QueuePairMLX5& operator=(const QueuePairMLX5& other) = delete;
  __host__ QueuePairMLX5(QueuePairMLX5&& other) noexcept        = default;
  __host__ QueuePairMLX5& operator=(QueuePairMLX5&& other)      = default;
  __host__ ~QueuePairMLX5()                                     = default;

public:
  template <OpCode Op,
            bool RingDB = true, bool ThreadSafe = true, bool CheckCQ = true>
  __device__ void post_wqe_rma(uintptr_t laddr, uintptr_t raddr, size_t size,
                               const ActiveWFInfo& wf_info);

  template <OpCode Op,
            bool RingDB = true, bool ThreadSafe = true, bool CheckCQ = true>
  __device__ void post_wqe_rma_single(uintptr_t laddr, uintptr_t raddr, size_t size);

  template <OpCode Op, AMOFetchType Fetch,
            bool RingDB = true, bool ThreadSafe = true, bool CheckCQ = true>
  __device__ amo_ret_t<Fetch> post_wqe_amo(uintptr_t raddr, uint64_t value, uint64_t cond,
                                                 const ActiveWFInfo& wf_info);

  template <OpCode Op, AMOFetchType Fetch,
            bool RingDB = true, bool ThreadSafe = true, bool CheckCQ = true>
  __device__ amo_ret_t<Fetch> post_wqe_amo_single(uintptr_t raddr, uint64_t value, uint64_t cond);

  __device__ void quiet_single();

private:
  __device__ void ring_doorbell(uint64_t sq_post, const gda_mlx5_wqe& wqe);
  __device__ void poll_cq_until(uint16_t requested_available_slots);

  __device__ static void acquire_lock(uint32_t* lock);
  __device__ static void release_lock(uint32_t* lock);

  __device__ inline uint16_t get_wqe_idx(uint8_t lane_id) {
    return static_cast<uint16_t>(sq.post + lane_id);
  }

  __device__ inline uint16_t get_sq_idx(uint16_t wqe_idx) {
    // sq.depth is a power of 2, so just mask off everything above that
    return wqe_idx & sq.depth_mask;
  }

  template <bool ThreadSafe, bool CheckCQ>
  __device__ void lock_pollcq(int wqe_count);

  template <bool RingDB, bool ThreadSafe>
  __device__ void post_ringdb_unlock(int wqe_count, const gda_mlx5_wqe& wqe);

#if defined(BUILD_DEBUG_DEVICE)
  __device__ __attribute__((noinline)) void print_cqe_error(const mlx5_cqe64* cqe,
                                                            uint8_t opcode, uint8_t owner);
#endif

  gda_mlx5_device_sq sq;
  gda_mlx5_device_cq cq;
};

template <bool ThreadSafe, bool CheckCQ>
__device__ void QueuePairMLX5::lock_pollcq(int wqe_count) {
  if constexpr (ThreadSafe || CheckCQ) {
    if constexpr (ThreadSafe) {
      acquire_lock(&sq.lock);
    }
    if constexpr (CheckCQ) {
      poll_cq_until(wqe_count);
    }
  } else {
    // need to at least acquire so that post counter is visible
    __builtin_amdgcn_fence(__ATOMIC_ACQUIRE, "agent");
  }
}

template <bool RingDB, bool ThreadSafe>
__device__ void QueuePairMLX5::post_ringdb_unlock(int wqe_count, const gda_mlx5_wqe& wqe) {
  sq.post += wqe_count;
  if constexpr (RingDB || ThreadSafe) {
    if constexpr (RingDB) {
      ring_doorbell(sq.post, wqe);
    }
    if constexpr (ThreadSafe) {
      release_lock(&sq.lock);
    }
  } else {
    // need to at least release so that post counter is available
    __builtin_amdgcn_fence(__ATOMIC_RELEASE, "agent");
  }
}

// can be called with all active lanes using any number of different QPs, don't assume anything
template <QueuePairMLX5::OpCode Op, bool RingDB, bool ThreadSafe, bool CheckCQ>
__device__ void QueuePairMLX5::post_wqe_rma(
    uintptr_t laddr, uintptr_t raddr, size_t size, const ActiveWFInfo& wf_info) {
  uint32_t byte_count = static_cast<uint32_t>(size);
  if (wf_info.is_pe_group_last) {
    // acquire SQ lock and poll until we have enough WQEBB for all lanes using this QP
    lock_pollcq<ThreadSafe, CheckCQ>(wf_info.num_pe_group_lanes);
  }

  // wqe_idx is the logical WQE id that wraps at 0xFFFF, sq_idx is the index into the actual SQ
  uint16_t wqe_idx = get_wqe_idx(wf_info.pe_group_logical_lane_id);
  uint16_t sq_idx  = get_sq_idx(wqe_idx);

  // can we inline the data into the WQE?
  bool send_inline = gda_mlx5_wqe_rma::can_inline<static_cast<uint8_t>(Op)>(byte_count);
  uint32_t lkey = !send_inline ? get_lkey(laddr) : 0;

  // construct the WQE on the stack
  gda_mlx5_wqe wqe{wqe_idx, static_cast<uint8_t>(Op), qp_num, MLX5_WQE_CTRL_CQ_UPDATE,
                   raddr, rkey, laddr, lkey, byte_count, send_inline};

  // copy to SQ
  sq.buf[sq_idx] = wqe;

  if (wf_info.is_pe_group_last) {
    /* increment post counter, ring doorbell, and release SQ lock
     * we are the last thread in the wavefront, so we have the last WQE posted */
    post_ringdb_unlock<RingDB, ThreadSafe>(wf_info.num_pe_group_lanes, wqe);
  }
}

// precondition: called with all active lanes using different QPs
template <QueuePairMLX5::OpCode Op, bool RingDB, bool ThreadSafe, bool CheckCQ>
__device__ void QueuePairMLX5::post_wqe_rma_single(
    uintptr_t laddr, uintptr_t raddr, size_t size) {
  uint32_t byte_count = static_cast<uint32_t>(size);
  // acquire SQ lock and poll until we have enough space for at least one WQEBB
  lock_pollcq<ThreadSafe, CheckCQ>(1);

  // wqe_idx is the logical WQE id that wraps at 0xFFFF, sq_idx is the index into the actual SQ
  uint16_t wqe_idx = get_wqe_idx(0);
  uint16_t sq_idx  = get_sq_idx(wqe_idx);

  // can we inline the data into the WQE?
  bool send_inline = gda_mlx5_wqe_rma::can_inline<static_cast<uint8_t>(Op)>(byte_count);
  uint32_t lkey = !send_inline ? get_lkey(laddr) : 0;

  // construct the WQE on the stack
  gda_mlx5_wqe wqe{wqe_idx, static_cast<uint8_t>(Op), qp_num, MLX5_WQE_CTRL_CQ_UPDATE,
                   raddr, rkey, laddr, lkey, byte_count, send_inline};

  // copy to SQ
  sq.buf[sq_idx] = wqe;

  // increment post counter, ring doorbell for this WQE, and release SQ lock
  post_ringdb_unlock<RingDB, ThreadSafe>(1, wqe);
}

// can be called with all active lanes using any number of different QPs, don't assume anything
template <QueuePairMLX5::OpCode Op, AMOFetchType Fetch,
          bool RingDB, bool ThreadSafe, bool CheckCQ>
__device__ QueuePairMLX5::amo_ret_t<Fetch> QueuePairMLX5::post_wqe_amo(
    uintptr_t raddr, uint64_t value, uint64_t cond, const ActiveWFInfo& wf_info) {
  static_assert(Fetch != AMOFetchType::NonBlocking);
  if (wf_info.is_pe_group_last) {
    // acquire SQ lock and poll until we have enough WQEBB for all lanes using this QP
    lock_pollcq<ThreadSafe, CheckCQ>(wf_info.num_pe_group_lanes);
  }

  uint64_t* atomic_laddr = get_atomic_addr<Fetch>();
  uint32_t atomic_lkey   = get_atomic_lkey<Fetch>();
  if constexpr (Fetch == AMOFetchType::Blocking) {
    uint32_t atomic_idx = (fetching_atomic_idx + wf_info.pe_group_logical_lane_id) % FETCHING_ATOMIC_CNT;
    atomic_laddr += atomic_idx;
  }

  // wqe_idx is the logical WQE id that wraps at 0xFFFF, sq_idx is the index into the actual SQ
  uint16_t wqe_idx = get_wqe_idx(wf_info.pe_group_logical_lane_id);
  uint16_t sq_idx  = get_sq_idx(wqe_idx);

  // construct the WQE on the stack
  gda_mlx5_wqe wqe{wqe_idx, static_cast<uint8_t>(Op), qp_num, MLX5_WQE_CTRL_CQ_UPDATE,
                   raddr, rkey, value, cond, reinterpret_cast<uintptr_t>(atomic_laddr), atomic_lkey};

  // copy to SQ
  sq.buf[sq_idx] = wqe;

  if (wf_info.is_pe_group_last) {
    // increment fetching-atomic counter
    if constexpr (Fetch == AMOFetchType::Blocking) {
      fetching_atomic_idx += wf_info.num_pe_group_lanes;
    }
    /* increment post counter, ring doorbell, and release SQ lock
     * we are the last thread in the wavefront, so we have the last WQE posted */
    post_ringdb_unlock<RingDB, ThreadSafe>(wf_info.num_pe_group_lanes, wqe);
    // wait until fetch completes
    if constexpr (Fetch == AMOFetchType::Blocking) {
      quiet_single();
    }
  }

  if constexpr (Fetch == AMOFetchType::Blocking) {
    return *atomic_laddr;
  }
}

// precondition: called with all active lanes using different QPs
template <QueuePairMLX5::OpCode Op, AMOFetchType Fetch,
          bool RingDB, bool ThreadSafe, bool CheckCQ>
__device__ QueuePairMLX5::amo_ret_t<Fetch> QueuePairMLX5::post_wqe_amo_single(
    uintptr_t raddr, uint64_t value, uint64_t cond) {
  static_assert(Fetch != AMOFetchType::NonBlocking);
  // acquire SQ lock and poll until we have enough space for at least one WQEBB
  lock_pollcq<ThreadSafe, CheckCQ>(1);

  uint64_t* atomic_laddr = get_atomic_addr<Fetch>();
  uint32_t atomic_lkey   = get_atomic_lkey<Fetch>();
  if constexpr (Fetch == AMOFetchType::Blocking) {
    uint32_t atomic_idx = fetching_atomic_idx % FETCHING_ATOMIC_CNT;
    atomic_laddr += atomic_idx;
  }

  // wqe_idx is the logical WQE id that wraps at 0xFFFF, sq_idx is the index into the actual SQ
  uint16_t wqe_idx = get_wqe_idx(0);
  uint16_t sq_idx  = get_sq_idx(wqe_idx);

  // construct the WQE on the stack
  gda_mlx5_wqe wqe{wqe_idx, static_cast<uint8_t>(Op), qp_num, MLX5_WQE_CTRL_CQ_UPDATE,
                   raddr, rkey, value, cond, reinterpret_cast<uintptr_t>(atomic_laddr), atomic_lkey};

  // copy to SQ
  sq.buf[sq_idx] = wqe;

  // increment fetching-atomic counter
  if constexpr (Fetch == AMOFetchType::Blocking) {
    fetching_atomic_idx += 1;
  }
  // increment post counter, ring doorbell for this WQE, and release SQ lock
  post_ringdb_unlock<RingDB, ThreadSafe>(1, wqe);
  // wait until fetch completes
  if constexpr (Fetch == AMOFetchType::Blocking) {
    quiet_single();
    return *atomic_laddr;
  }
}

}  // namespace rocshmem

#endif  // LIBRARY_SRC_GDA_MLX5_QUEUE_PAIR_HPP_
