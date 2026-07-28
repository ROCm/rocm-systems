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

  static constexpr size_t InlineThreshold = sizeof(gda_mlx5_wqe_inline_data::data);
};

class QueuePairMLX5 : public QueuePairBase<QueuePairMLX5> {
public:
  __host__ explicit QueuePairMLX5(uint32_t qpn, uintptr_t heap_laddr, uint32_t heap_lkey,
                                  uintptr_t heap_raddr, uint32_t heap_rkey, size_t heap_size,
                                  const QpSymmEntry *symm_entries, const int *symm_count,
                                  struct ibv_pd* pd,
                                  gda_mlx5_device_sq&& sq, gda_mlx5_device_cq&& cq);

  __host__ QueuePairMLX5(const QueuePairMLX5& other)            = delete;
  __host__ QueuePairMLX5& operator=(const QueuePairMLX5& other) = delete;
  __host__ QueuePairMLX5(QueuePairMLX5&& other) noexcept        = default;
  __host__ QueuePairMLX5& operator=(QueuePairMLX5&& other)      = default;
  __host__ ~QueuePairMLX5()                                     = default;

public:
  template <OpCode Op, typename... Options>
  __device__ __noinline__
  void post_wqe_rma(uintptr_t laddr, uint32_t lkey,
                    uintptr_t raddr, uint32_t rkey, size_t size,
                    const ActiveWFInfo& wf_info, PostOpt<Options...> = {});

  template <OpCode Op, typename... Options>
  __device__ __noinline__
  void post_wqe_rma_single(uintptr_t laddr, uint32_t lkey,
                           uintptr_t raddr, uint32_t rkey, size_t size,
                           PostOpt<Options...> = {});

  template <OpCode Op, AMOFetchType Fetch, typename... Options>
  __device__ __noinline__
  amo_ret_t<Fetch> post_wqe_amo(uintptr_t raddr, uint32_t rkey,
                                uint64_t swap_add, uint64_t compare,
                                const ActiveWFInfo& wf_info, PostOpt<Options...> = {});

  template <OpCode Op, AMOFetchType Fetch, typename... Options>
  __device__ __noinline__
  amo_ret_t<Fetch> post_wqe_amo_single(uintptr_t raddr, uint32_t rkey,
                                       uint64_t swap_add, uint64_t compare,
                                       PostOpt<Options...> = {});

  __device__ __noinline__ void quiet_single();

private:
  __device__ void ring_doorbell(uint64_t sq_post, const gda_mlx5_wqe& wqe);
  __device__ void poll_cq_until(uint16_t requested_available_slots);

  static __device__ void acquire_lock(uint32_t* lock);
  static __device__ void release_lock(uint32_t* lock);

  __device__ inline uint16_t get_wqe_idx(uint8_t lane_id) {
    return static_cast<uint16_t>(sq.post + lane_id);
  }

  __device__ inline uint16_t get_sq_idx(uint16_t wqe_idx) {
    // sq.depth is a power of 2, so just mask off everything above that
    return wqe_idx & sq.depth_mask;
  }

  template <typename PostOptions>
  __device__ void lock_pollcq(int wqe_count);

  template <typename PostOptions>
  __device__ void post_ringdb_unlock(int wqe_count, const gda_mlx5_wqe& wqe);

#if defined(BUILD_DEBUG_DEVICE)
  __device__ __attribute__((noinline)) void print_cqe_error(const mlx5_cqe64* cqe,
                                                            uint8_t opcode, uint8_t owner);
#endif

  gda_mlx5_device_sq sq;
  gda_mlx5_device_cq cq;
};

template <typename PostOptions>
__device__ void QueuePairMLX5::lock_pollcq(int wqe_count) {
  if constexpr (PostOptions::ThreadSafe || PostOptions::CheckSQ) {
    if constexpr (PostOptions::ThreadSafe) {
      acquire_lock(&sq.lock);
    }
    if constexpr (PostOptions::CheckSQ) {
      poll_cq_until(wqe_count);
    }
  } else {
    // need to at least acquire so that post counter is visible
    __builtin_amdgcn_fence(__ATOMIC_ACQUIRE, "agent");
  }
}

template <typename PostOptions>
__device__ void QueuePairMLX5::post_ringdb_unlock(int wqe_count, const gda_mlx5_wqe& wqe) {
  sq.post += wqe_count;
  if constexpr (PostOptions::RingDB || PostOptions::ThreadSafe) {
    if constexpr (PostOptions::RingDB) {
      ring_doorbell(sq.post, wqe);
    }
    if constexpr (PostOptions::ThreadSafe) {
      release_lock(&sq.lock);
    }
  } else {
    // need to at least release so that post counter is available
    __builtin_amdgcn_fence(__ATOMIC_RELEASE, "agent");
  }
}

// can be called with all active lanes using any number of different QPs, don't assume anything
template <QueuePairMLX5::OpCode Op, typename... Options>
__device__ __noinline__ void QueuePairMLX5::post_wqe_rma(
    uintptr_t laddr, uint32_t lkey, uintptr_t raddr, uint32_t rkey, size_t size,
    const ActiveWFInfo& wf_info, PostOpt<Options...>) {
  using PostOptions = PostOpt<Options...>;
  uint32_t byte_count = static_cast<uint32_t>(size);
  if (wf_info.is_pe_group_last) {
    // acquire SQ lock and poll until we have enough WQEBB for all lanes using this QP
    lock_pollcq<PostOptions>(wf_info.num_pe_group_lanes);
  }

  // wqe_idx is the logical WQE id that wraps at 0xFFFF, sq_idx is the index into the actual SQ
  uint16_t wqe_idx = get_wqe_idx(wf_info.pe_group_logical_lane_id);
  uint16_t sq_idx  = get_sq_idx(wqe_idx);

  // can we inline the data into the WQE?
  bool send_inline = can_inline<Op>(size);

  // should we update CQ for this WQE?
  uint8_t fm_ce_se = PostOptions::signal_completion(wf_info) ? MLX5_WQE_CTRL_CQ_UPDATE : 0;

  // construct the WQE on the stack
  gda_mlx5_wqe wqe{wqe_idx, static_cast<uint8_t>(Op), qp_num, fm_ce_se,
                   raddr, rkey, laddr, lkey, byte_count, send_inline};

  // copy to SQ
  sq.buf[sq_idx] = wqe;

  if (wf_info.is_pe_group_last) {
    /* increment post counter, ring doorbell, and release SQ lock
     * we are the last thread in the wavefront, so we have the last WQE posted */
    post_ringdb_unlock<PostOptions>(wf_info.num_pe_group_lanes, wqe);
  }
}

// precondition: called with all active lanes using different QPs
template <QueuePairMLX5::OpCode Op, typename... Options>
__device__ __noinline__ void QueuePairMLX5::post_wqe_rma_single(
    uintptr_t laddr, uint32_t lkey, uintptr_t raddr, uint32_t rkey, size_t size, PostOpt<Options...>) {
  using PostOptions = PostOpt<Options...>;
  uint32_t byte_count = static_cast<uint32_t>(size);
  // acquire SQ lock and poll until we have enough space for at least one WQEBB
  lock_pollcq<PostOptions>(1);

  // wqe_idx is the logical WQE id that wraps at 0xFFFF, sq_idx is the index into the actual SQ
  uint16_t wqe_idx = get_wqe_idx(0);
  uint16_t sq_idx  = get_sq_idx(wqe_idx);

  // can we inline the data into the WQE?
  bool send_inline = can_inline<Op>(size);

  // should we update CQ for this WQE?
  uint8_t fm_ce_se = PostOptions::signal_completion_single() ? MLX5_WQE_CTRL_CQ_UPDATE : 0;

  // construct the WQE on the stack
  gda_mlx5_wqe wqe{wqe_idx, static_cast<uint8_t>(Op), qp_num, fm_ce_se,
                   raddr, rkey, laddr, lkey, byte_count, send_inline};

  // copy to SQ
  sq.buf[sq_idx] = wqe;

  // increment post counter, ring doorbell for this WQE, and release SQ lock
  post_ringdb_unlock<PostOptions>(1, wqe);
}

// can be called with all active lanes using any number of different QPs, don't assume anything
template <QueuePairMLX5::OpCode Op, AMOFetchType Fetch, typename... Options>
__device__ __noinline__ QueuePairMLX5::amo_ret_t<Fetch> QueuePairMLX5::post_wqe_amo(
    uintptr_t raddr, uint32_t rkey, uint64_t swap_add, uint64_t compare,
    const ActiveWFInfo& wf_info, PostOpt<Options...>) {
  static_assert(Fetch != AMOFetchType::NonBlocking);
  using PostOptions = PostOpt<Options...>;
  if (wf_info.is_pe_group_last) {
    // acquire SQ lock and poll until we have enough WQEBB for all lanes using this QP
    lock_pollcq<PostOptions>(wf_info.num_pe_group_lanes);
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

  // should we update CQ for this WQE?
  uint8_t fm_ce_se = PostOptions::signal_completion(wf_info) ? MLX5_WQE_CTRL_CQ_UPDATE : 0;

  // construct the WQE on the stack
  gda_mlx5_wqe wqe{wqe_idx, static_cast<uint8_t>(Op), qp_num, fm_ce_se,
                   raddr, rkey, swap_add, compare, reinterpret_cast<uintptr_t>(atomic_laddr), atomic_lkey};

  // copy to SQ
  sq.buf[sq_idx] = wqe;

  if (wf_info.is_pe_group_last) {
    // increment fetching-atomic counter
    if constexpr (Fetch == AMOFetchType::Blocking) {
      fetching_atomic_idx += wf_info.num_pe_group_lanes;
    }
    /* increment post counter, ring doorbell, and release SQ lock
     * we are the last thread in the wavefront, so we have the last WQE posted */
    post_ringdb_unlock<PostOptions>(wf_info.num_pe_group_lanes, wqe);
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
template <QueuePairMLX5::OpCode Op, AMOFetchType Fetch, typename... Options>
__device__ __noinline__ QueuePairMLX5::amo_ret_t<Fetch> QueuePairMLX5::post_wqe_amo_single(
    uintptr_t raddr, uint32_t rkey, uint64_t swap_add, uint64_t compare, PostOpt<Options...>) {
  static_assert(Fetch != AMOFetchType::NonBlocking);
  using PostOptions = PostOpt<Options...>;
  // acquire SQ lock and poll until we have enough space for at least one WQEBB
  lock_pollcq<PostOptions>(1);

  uint64_t* atomic_laddr = get_atomic_addr<Fetch>();
  uint32_t atomic_lkey   = get_atomic_lkey<Fetch>();
  if constexpr (Fetch == AMOFetchType::Blocking) {
    uint32_t atomic_idx = fetching_atomic_idx % FETCHING_ATOMIC_CNT;
    atomic_laddr += atomic_idx;
  }

  // wqe_idx is the logical WQE id that wraps at 0xFFFF, sq_idx is the index into the actual SQ
  uint16_t wqe_idx = get_wqe_idx(0);
  uint16_t sq_idx  = get_sq_idx(wqe_idx);

  // should we update CQ for this WQE?
  uint8_t fm_ce_se = PostOptions::signal_completion_single() ? MLX5_WQE_CTRL_CQ_UPDATE : 0;

  // construct the WQE on the stack
  gda_mlx5_wqe wqe{wqe_idx, static_cast<uint8_t>(Op), qp_num, fm_ce_se,
                   raddr, rkey, swap_add, compare, reinterpret_cast<uintptr_t>(atomic_laddr), atomic_lkey};

  // copy to SQ
  sq.buf[sq_idx] = wqe;

  // increment fetching-atomic counter
  if constexpr (Fetch == AMOFetchType::Blocking) {
    fetching_atomic_idx += 1;
  }
  // increment post counter, ring doorbell for this WQE, and release SQ lock
  post_ringdb_unlock<PostOptions>(1, wqe);
  // wait until fetch completes
  if constexpr (Fetch == AMOFetchType::Blocking) {
    quiet_single();
    return *atomic_laddr;
  }
}

}  // namespace rocshmem

#endif  // LIBRARY_SRC_GDA_MLX5_QUEUE_PAIR_HPP_
