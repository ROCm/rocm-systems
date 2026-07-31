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

#ifndef LIBRARY_SRC_GDA_BNXT_QUEUE_PAIR_HPP_
#define LIBRARY_SRC_GDA_BNXT_QUEUE_PAIR_HPP_

#include <utility>

#include "gda/endian.hpp"
#include "gda/bnxt/provider_gda_bnxt.hpp"
#include "gda/queue_pair.hpp"

namespace rocshmem {

class QueuePairBNXT;

template <> struct QueuePairTraits<QueuePairBNXT> {
  enum class OpCode : uint8_t {
    RDMA_WRITE = BNXT_RE_WR_OPCD_RDMA_WRITE,
    RDMA_READ  = BNXT_RE_WR_OPCD_RDMA_READ,
    ATOMIC_CS  = BNXT_RE_WR_OPCD_ATOMIC_CS,
    ATOMIC_FA  = BNXT_RE_WR_OPCD_ATOMIC_FA,
  };

  /**
   * @brief bnxt uses little-endian ordering
   */
  static constexpr endian::Order Endianness = endian::Order::Little;

  /**
   * @brief bnxt inlining maximum is one WQE segment: sizeof(struct bnxt_re_sge) = 16 bytes
   */
  static constexpr size_t InlineMax = sizeof(struct bnxt_re_sge);

  /**
   * @brief bnxt preferred inlining threshold is 8 bytes, for performance reasons
   */
  static constexpr size_t InlineThreshold = sizeof(uint64_t);
};

class QueuePairBNXT : public QueuePairBase<QueuePairBNXT> {
public:
  __host__ explicit QueuePairBNXT(uint32_t qpn, uintptr_t heap_laddr, uint32_t heap_lkey,
                                  uintptr_t heap_raddr, uint32_t heap_rkey, size_t heap_size,
                                  const QpSymmEntry *symm_entries, const int *symm_count,
                                  struct ibv_pd* pd,
                                  uint64_t* dbr, bnxt_device_sq&& sq, bnxt_device_cq&& cq);

  __host__ QueuePairBNXT(const QueuePairBNXT& other)            = delete;
  __host__ QueuePairBNXT& operator=(const QueuePairBNXT& other) = delete;
  __host__ QueuePairBNXT(QueuePairBNXT&& other) noexcept        = default;
  __host__ QueuePairBNXT& operator=(QueuePairBNXT&& other)      = default;
  __host__ ~QueuePairBNXT()                                     = default;

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
  __device__ void ring_doorbell(uint32_t slot_idx);
  __device__ void poll_cq_until(uint32_t requested_available_slots);

  static __device__ void acquire_lock(uint32_t* lock);
  static __device__ void release_lock(uint32_t* lock);

  template <OpCode Op, bool CheckSQ>
  __device__ void write_rma_wqe(uintptr_t laddr, uint32_t lkey,
                                uintptr_t raddr, uint32_t rkey, size_t size, bool signaled);

  template <OpCode Op, AMOFetchType Fetch, bool CheckSQ>
  __device__ uint64_t* write_amo_wqe(uintptr_t raddr, uint32_t rkey,
                                     uint64_t swap_add, uint64_t compare, bool signaled);

  static __device__ void* get_hwqe(const bnxt_device_sq& sq, uint32_t idx);
  static __device__ void fill_psns_for_msntbl(bnxt_device_sq& sq, uint32_t msg_len);
  static __device__ void incr_tail(bnxt_device_sq& sq, uint8_t cnt);

#if defined(BUILD_DEBUG_DEVICE)
  __device__ __attribute__((noinline)) void print_cqe_error(uint8_t status);
#endif

  uint64_t* dbr;
  bnxt_device_sq sq;
  bnxt_device_cq cq;
};

template <QueuePairBNXT::OpCode Op, bool CheckSQ>
__device__ void QueuePairBNXT::write_rma_wqe(
    uintptr_t laddr, uint32_t lkey, uintptr_t raddr, uint32_t rkey, size_t size, bool signaled) {
  struct bnxt_re_bsqe hdr;
  struct bnxt_re_rdma rdma;
  struct bnxt_re_sge sge;
  struct bnxt_re_bsqe *hdr_ptr;
  struct bnxt_re_rdma *rdma_ptr;
  struct bnxt_re_sge *sge_ptr;
  uint32_t wqe_size;
  uint32_t wqe_type;
  uint32_t hdr_flags;

  bool inline_msg = can_inline<Op>(size);

  if constexpr (CheckSQ) {
    poll_cq_until(GDA_BNXT_WQE_SLOT_COUNT);
  }

  hdr_ptr  = (struct bnxt_re_bsqe*) get_hwqe(sq, 0);
  rdma_ptr = (struct bnxt_re_rdma*) get_hwqe(sq, 1);
  sge_ptr  = (struct bnxt_re_sge*)  get_hwqe(sq, 2);

  /* Populate Header Segment */
  wqe_type  = BNXT_RE_HDR_WT_MASK & static_cast<uint8_t>(Op);
  wqe_size  = BNXT_RE_HDR_WS_MASK & GDA_BNXT_WQE_SLOT_COUNT;
  hdr_flags = signaled ? (BNXT_RE_HDR_FLAGS_MASK & BNXT_RE_WR_FLAGS_SIGNALED) : 0;

  if (inline_msg) {
    hdr_flags |= ((uint32_t) BNXT_RE_WR_FLAGS_INLINE);
  }

  hdr.rsv_ws_fl_wt  = (wqe_size  << BNXT_RE_HDR_WS_SHIFT)
                    | (hdr_flags << BNXT_RE_HDR_FLAGS_SHIFT)
                    | wqe_type;
  hdr.key_immd      = 0;
  hdr.lhdr.qkey_len = size;

  /* Populate RDMA Segment */
  rdma.rva  = raddr;
  rdma.rkey = rkey;

  if (!inline_msg) {
    /* Populate SG Segment */
    sge.pa     = laddr;
    sge.lkey   = lkey;
    sge.length = size;
  }

  /* Write WQE to SQ */
  memcpy(hdr_ptr,  &hdr,  sizeof(struct bnxt_re_bsqe));
  memcpy(rdma_ptr, &rdma, sizeof(struct bnxt_re_rdma));

  if (inline_msg) {
    memcpy(sge_ptr, reinterpret_cast<void*>(laddr), size);
  } else {
    memcpy(sge_ptr, &sge, sizeof(struct bnxt_re_sge));
  }

  /* Populate MSN Table */
  fill_psns_for_msntbl(sq, size);

  /* Update SQ Pointer */
  incr_tail(sq, GDA_BNXT_WQE_SLOT_COUNT);
}

// can be called with all active lanes using any number of different QPs, don't assume anything
template <QueuePairBNXT::OpCode Op, typename... Options>
__device__ __noinline__ void QueuePairBNXT::post_wqe_rma(
    uintptr_t laddr, uint32_t lkey, uintptr_t raddr, uint32_t rkey, size_t size,
    const ActiveWFInfo& wf_info, PostOpt<Options...>) {
  using PostOptions = PostOpt<Options...>;
  if constexpr (PostOptions::ThreadSafe) {
    if (wf_info.is_pe_group_first) {
      acquire_lock(&sq.lock);
    }
  } else if constexpr (!PostOptions::CheckSQ) {
    // need to at least acquire so that tail is visible
    __builtin_amdgcn_fence(__ATOMIC_ACQUIRE, "agent");
  }

  bool signaled = PostOptions::signal_completion(wf_info);

  for (int i = 0; i < wf_info.num_pe_group_lanes; i++) {
    if (i == wf_info.pe_group_logical_lane_id) {
      /* Write WQE to SQ */
      write_rma_wqe<Op, PostOptions::CheckSQ>(laddr, lkey, raddr, rkey, size, signaled);

      /* Ring Doorbell */
      if constexpr (PostOptions::RingDB) {
        ring_doorbell(sq.tail);
      }
    }
  }

  if constexpr (PostOptions::ThreadSafe) {
    if (wf_info.is_pe_group_first) {
      release_lock(&sq.lock);
    }
  } else if constexpr (!PostOptions::RingDB) {
    // need to at least release so that tail is available
    __builtin_amdgcn_fence(__ATOMIC_RELEASE, "agent");
  }
}

// precondition: called with all active lanes using different QPs
template <QueuePairBNXT::OpCode Op, typename... Options>
__device__ __noinline__ void QueuePairBNXT::post_wqe_rma_single(
    uintptr_t laddr, uint32_t lkey, uintptr_t raddr, uint32_t rkey, size_t size, PostOpt<Options...>) {
  using PostOptions = PostOpt<Options...>;
  if constexpr (PostOptions::ThreadSafe) {
    acquire_lock(&sq.lock);
  } else if constexpr (!PostOptions::CheckSQ) {
    // need to at least acquire so that tail is visible
    __builtin_amdgcn_fence(__ATOMIC_ACQUIRE, "agent");
  }

  bool signaled = PostOptions::signal_completion_single();

  /* Write WQE to SQ */
  write_rma_wqe<Op, PostOptions::CheckSQ>(laddr, lkey, raddr, rkey, size, signaled);

  /* Ring Doorbell */
  if constexpr (PostOptions::RingDB) {
    ring_doorbell(sq.tail);
  }

  if constexpr (PostOptions::ThreadSafe) {
    release_lock(&sq.lock);
  } else if constexpr (!PostOptions::RingDB) {
    // need to at least release so that tail is available
    __builtin_amdgcn_fence(__ATOMIC_RELEASE, "agent");
  }
}

template <QueuePairBNXT::OpCode Op, AMOFetchType Fetch, bool CheckSQ>
__device__ uint64_t* QueuePairBNXT::write_amo_wqe(
    uintptr_t raddr, uint32_t rkey, uint64_t swap_add, uint64_t compare, bool signaled) {
  static_assert(Fetch != AMOFetchType::NonBlocking);
  static constexpr size_t size = sizeof(uint64_t);

  struct bnxt_re_bsqe hdr;
  struct bnxt_re_atomic amo;
  struct bnxt_re_sge sge;
  struct bnxt_re_bsqe *hdr_ptr;
  struct bnxt_re_atomic *amo_ptr;
  struct bnxt_re_sge *sge_ptr;
  uint32_t wqe_size;
  uint32_t wqe_type;
  uint32_t hdr_flags;
  uint64_t* atomic_laddr;

  if constexpr (CheckSQ) {
    poll_cq_until(GDA_BNXT_WQE_SLOT_COUNT);
  }

  hdr_ptr = (struct bnxt_re_bsqe*)   get_hwqe(sq, 0);
  amo_ptr = (struct bnxt_re_atomic*) get_hwqe(sq, 1);
  sge_ptr = (struct bnxt_re_sge*)    get_hwqe(sq, 2);

  /* Populate Header Segment */
  wqe_type  = BNXT_RE_HDR_WT_MASK & static_cast<uint8_t>(Op);
  wqe_size  = BNXT_RE_HDR_WS_MASK & GDA_BNXT_WQE_SLOT_COUNT;
  hdr_flags = signaled ? (BNXT_RE_HDR_FLAGS_MASK & BNXT_RE_WR_FLAGS_SIGNALED) : 0;

  hdr.rsv_ws_fl_wt  = (wqe_size  << BNXT_RE_HDR_WS_SHIFT)
                    | (hdr_flags << BNXT_RE_HDR_FLAGS_SHIFT)
                    | wqe_type;
  hdr.key_immd = rkey;
  hdr.lhdr.rva = raddr;

  /* Populate AMO Segment */
  amo.swp_dt = swap_add;
  amo.cmp_dt = compare;

  /* Populate SG Segment - (Return address of atomic) */
  atomic_laddr = get_atomic_addr<Fetch>();
  if constexpr (Fetch == AMOFetchType::Blocking) {
    atomic_laddr += (fetching_atomic_idx++ % FETCHING_ATOMIC_CNT);
  }
  sge.pa     = reinterpret_cast<uintptr_t>(atomic_laddr);
  sge.lkey   = get_atomic_lkey<Fetch>();
  sge.length = size;

  /* Write WQE to SQ */
  memcpy(hdr_ptr, &hdr, sizeof(struct bnxt_re_bsqe));
  memcpy(amo_ptr, &amo, sizeof(struct bnxt_re_atomic));
  memcpy(sge_ptr, &sge, sizeof(struct bnxt_re_sge));

  /* Populate MSN Table */
  fill_psns_for_msntbl(sq, size);

  /* Update SQ Pointer */
  incr_tail(sq, GDA_BNXT_WQE_SLOT_COUNT);

  return atomic_laddr;
}

// can be called with all active lanes using any number of different QPs, don't assume anything
template <QueuePairBNXT::OpCode Op, AMOFetchType Fetch, typename... Options>
__device__ __noinline__ QueuePairBNXT::amo_ret_t<Fetch> QueuePairBNXT::post_wqe_amo(
    uintptr_t raddr, uint32_t rkey, uint64_t swap_add, uint64_t compare,
    const ActiveWFInfo& wf_info, PostOpt<Options...>) {
  static_assert(Fetch != AMOFetchType::NonBlocking);
  using PostOptions = PostOpt<Options...>;
  uint64_t* atomic_laddr = nullptr;

  if constexpr (PostOptions::ThreadSafe) {
    if (wf_info.is_pe_group_first) {
      acquire_lock(&sq.lock);
    }
  } else if constexpr (!PostOptions::CheckSQ) {
    // need to at least acquire so that tail is visible
    __builtin_amdgcn_fence(__ATOMIC_ACQUIRE, "agent");
  }

  bool signaled = PostOptions::signal_completion(wf_info);

  for (int i = 0; i < wf_info.num_pe_group_lanes; i++) {
    if (i == wf_info.pe_group_logical_lane_id) {
      /* Write WQE to SQ */
      atomic_laddr = write_amo_wqe<Op, Fetch, PostOptions::CheckSQ>(raddr, rkey, swap_add, compare, signaled);

      /* Ring Doorbell */
      if constexpr (PostOptions::RingDB) {
        ring_doorbell(sq.tail);
      }
    }
  }

  if constexpr (PostOptions::ThreadSafe) {
    if (wf_info.is_pe_group_first) {
      release_lock(&sq.lock);
    }
  } else if constexpr (!PostOptions::RingDB) {
    // need to at least release so that tail is available
    __builtin_amdgcn_fence(__ATOMIC_RELEASE, "agent");
  }

  if constexpr (Fetch == AMOFetchType::Blocking) {
    quiet(wf_info);
    return *atomic_laddr;
  }
}

// precondition: called with all active lanes using different QPs
template <QueuePairBNXT::OpCode Op, AMOFetchType Fetch, typename... Options>
__device__ __noinline__ QueuePairBNXT::amo_ret_t<Fetch> QueuePairBNXT::post_wqe_amo_single(
    uintptr_t raddr, uint32_t rkey, uint64_t swap_add, uint64_t compare, PostOpt<Options...>) {
  static_assert(Fetch != AMOFetchType::NonBlocking);
  using PostOptions = PostOpt<Options...>;
  uint64_t* atomic_laddr = nullptr;

  if constexpr (PostOptions::ThreadSafe) {
    acquire_lock(&sq.lock);
  } else if constexpr (!PostOptions::CheckSQ) {
    // need to at least acquire so that tail is visible
    __builtin_amdgcn_fence(__ATOMIC_ACQUIRE, "agent");
  }

  bool signaled = PostOptions::signal_completion_single();

  /* Write WQE to SQ */
  atomic_laddr = write_amo_wqe<Op, Fetch, PostOptions::CheckSQ>(raddr, rkey, swap_add, compare, signaled);

  /* Ring Doorbell */
  if constexpr (PostOptions::RingDB) {
    ring_doorbell(sq.tail);
  }

  if constexpr (PostOptions::ThreadSafe) {
    release_lock(&sq.lock);
  } else if constexpr (!PostOptions::RingDB) {
    // need to at least release so that tail is available
    __builtin_amdgcn_fence(__ATOMIC_RELEASE, "agent");
  }

  if constexpr (Fetch == AMOFetchType::Blocking) {
    quiet_single();
    return *atomic_laddr;
  }
}

}  // namespace rocshmem

#endif  // LIBRARY_SRC_GDA_BNXT_QUEUE_PAIR_HPP_

