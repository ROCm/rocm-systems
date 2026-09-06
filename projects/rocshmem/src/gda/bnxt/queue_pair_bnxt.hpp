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

#include "log.hpp"
#include "gda/endian.hpp"
#include "gda/bnxt/provider_gda_bnxt.hpp"
#include "gda/queue_pair/queue_pair_device.hpp"

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

class QueuePairBNXT : public QueuePairDevice<QueuePairBNXT> {
private:
  uint64_t* dbr;
  bnxt_device_sq sq;
  bnxt_device_cq cq;

public:
  __host__ explicit QueuePairBNXT(uint32_t qpn, uintptr_t heap_laddr, uint32_t heap_lkey,
                                  uintptr_t heap_raddr, uint32_t heap_rkey, size_t heap_size,
                                  uint64_t *fetching_atomic, uint32_t fetching_atomic_lkey,
                                  uint64_t *nonfetching_atomic, uint32_t nonfetching_atomic_lkey,
                                  FreeList<uint64_t*> *fetching_atomic_freelist,
                                  const BufferInfo *local_buffers, size_t num_user_buffers,
                                  const SymmBufferInfo *symm_buffers, const int *symm_count,
                                  uint64_t* dbr, bnxt_device_sq&& sq, bnxt_device_cq&& cq)
    : QueuePairDevice{qpn, heap_laddr, heap_lkey, heap_raddr, heap_rkey, heap_size,
                      fetching_atomic, fetching_atomic_lkey,
                      nonfetching_atomic, nonfetching_atomic_lkey,
                      fetching_atomic_freelist,
                      local_buffers, num_user_buffers,
                      symm_buffers, symm_count},
      dbr{dbr}, sq{std::move(sq)}, cq{std::move(cq)} { }

  __host__ explicit QueuePairBNXT(uint32_t qpn,
                                  uint64_t *nonfetching_atomic, uint32_t nonfetching_atomic_lkey,
                                  uint64_t* dbr, bnxt_device_sq&& sq, bnxt_device_cq&& cq)
    : QueuePairDevice{qpn, nonfetching_atomic, nonfetching_atomic_lkey},
      dbr{dbr}, sq{std::move(sq)}, cq{std::move(cq)} { }

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

  static __device__ struct bnxt_re_msns* pull_psn_buff(const bnxt_device_sq& sq);
  static __device__ uint64_t update_msn_tbl(uint32_t st_idx, uint32_t npsn, uint32_t start_psn);

#if defined(BUILD_DEBUG_DEVICE)
  __device__ __noinline__ void print_cqe_error(uint8_t status);
#endif
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
  static_assert(Fetch != AMOFetchType::NonBlocking, "non-blocking AMOs not yet implemented");
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
  static_assert(Fetch != AMOFetchType::NonBlocking, "non-blocking AMOs not yet implemented");
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
  static_assert(Fetch != AMOFetchType::NonBlocking, "non-blocking AMOs not yet implemented");
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

// precondition: called with all active lanes using different QPs
__device__ inline __noinline__ void QueuePairBNXT::quiet_single() {
  poll_cq_until(sq.depth);
}

// precondition: called with all active lanes using different QPs
__device__ inline void QueuePairBNXT::poll_cq_until(uint32_t requested_available_slots) {
  struct bnxt_re_req_cqe *cqe;
  uint32_t sq_tail;
  uint32_t sq_head;
  uint32_t sq_depth;
  uint32_t consumed_slots;
  uint32_t available_slots;

  sq_depth = sq.depth;

  do {
    cqe = (struct bnxt_re_req_cqe *) cq.buf;

#ifdef BUILD_DEBUG_DEVICE
    uint32_t flg_val =
        __hip_atomic_load(static_cast<uint32_t*>(
            __builtin_assume_aligned((char*)cqe + sizeof(struct bnxt_re_req_cqe)
                                                + offsetof(struct bnxt_re_bcqe, flg_st_typ_ph),
                                     4)),
            __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_SYSTEM);
    uint8_t status = (flg_val >> BNXT_RE_BCQE_STATUS_SHIFT) & BNXT_RE_BCQE_STATUS_MASK;
    if (status != BNXT_RE_REQ_ST_OK) {
      print_cqe_error(status);
    }
#endif

    /* Update the SQ head
     * This param provides us the wqe_idx but we need to convert to the slot idx.
     * We assume a static slots size of GDA_BNXT_WQE_SLOT_COUNT thus can multiply by this value */
    sq_head = (((cqe->con_indx & 0xFFFF) * GDA_BNXT_WQE_SLOT_COUNT) % sq_depth);
    sq.head = sq_head;

    sq_tail = __hip_atomic_load(&sq.tail, __ATOMIC_SEQ_CST, __HIP_MEMORY_SCOPE_AGENT);

    consumed_slots  = (sq_tail - sq_head + sq_depth) % sq_depth;
    available_slots = sq_depth - consumed_slots;
  } while (available_slots < requested_available_slots);
}

__device__ __forceinline__ void QueuePairBNXT::ring_doorbell(uint32_t slot_idx) {
  struct bnxt_re_db_hdr hdr;
  uint32_t epoch;
  uint64_t key_lo;
  uint64_t key_hi;

  epoch = (sq.flags & BNXT_RE_FLAG_EPOCH_TAIL_MASK) << BNXT_RE_DB_EPOCH_TAIL_SHIFT;

  key_lo = (slot_idx | epoch);

  key_hi = (qp_num & BNXT_RE_DB_QID_MASK)
         | (((uint64_t) BNXT_RE_QUE_TYPE_SQ & BNXT_RE_DB_TYP_MASK) << BNXT_RE_DB_TYP_SHIFT)
         | (0x1UL << BNXT_RE_DB_VALID_SHIFT);

  hdr.typ_qid_indx = (key_lo | (key_hi << 32));

  __threadfence_system();
  __hip_atomic_store(dbr, hdr.typ_qid_indx, __ATOMIC_SEQ_CST, __HIP_MEMORY_SCOPE_SYSTEM);
}

__device__ __forceinline__ void* QueuePairBNXT::get_hwqe(
    const bnxt_device_sq& sq, uint32_t idx) {
  idx += sq.tail;
  if (idx >= sq.depth)
    idx -= sq.depth;
  return (void *)((char*)sq.buf + (idx << 4));
}

__device__ __forceinline__ void QueuePairBNXT::fill_psns_for_msntbl(
    bnxt_device_sq& sq, uint32_t msg_len) {
   uint32_t npsn = 0, start_psn = 0, next_psn = 0;
   struct bnxt_re_msns msns;
   uint64_t *msns_ptr;
   uint32_t pkt_cnt = 0;
   /* Start slot index of the WQE */
   uint32_t st_idx = sq.tail; // * BNXT_RE_STATIC_WQE_SIZE_SLOTS; Do we need this?
   // Get the MSN table address
   msns_ptr = (uint64_t *)pull_psn_buff(sq);
   // Start PSN is the last recorded PSN
   // Calculate the packet count based on the len of the WQE/MTU
   msns.start_idx_next_psn_start_psn = 0;
   start_psn = sq.psn;
   pkt_cnt = (msg_len / sq.mtu);

   if (msg_len % sq.mtu)
       pkt_cnt++;

   /* Increment the psn even for 0 len packets
    * e.g. for opcode rdma-write-with-imm-data
    * with length field = 0
    */
   if (msg_len == 0)
       pkt_cnt = 1;

   /* make it 24 bit */
   next_psn = sq.psn + pkt_cnt;
   npsn = next_psn;
   sq.psn = next_psn;
   msns.start_idx_next_psn_start_psn |= update_msn_tbl(st_idx, npsn, start_psn);
   sq.msn++;
   sq.msn %= sq.msn_tbl_sz;

   memcpy(msns_ptr, &msns, sizeof(uint64_t));
}

__device__ __forceinline__ void QueuePairBNXT::incr_tail(
    bnxt_device_sq& sq, uint8_t cnt) {
  sq.tail += cnt;
  if (sq.tail >= sq.depth) {
    sq.tail %= sq.depth;
    /* Rolled over, Toggle Tail bit in epoch flags */
    sq.flags ^= 1UL << BNXT_RE_FLAG_EPOCH_TAIL_SHIFT;
  }
}

__device__ __forceinline__ struct bnxt_re_msns* QueuePairBNXT::pull_psn_buff(
    const bnxt_device_sq& sq) {
  return (struct bnxt_re_msns*)(((char *) sq.msntbl) + ((sq.msn) << sq.psn_sz_log2));
}

__device__ __forceinline__ uint64_t QueuePairBNXT::update_msn_tbl(
    uint32_t st_idx, uint32_t npsn, uint32_t start_psn) {
  return ((((uint64_t)(st_idx)    << BNXT_RE_SQ_MSN_SEARCH_START_IDX_SHIFT)
                                            & BNXT_RE_SQ_MSN_SEARCH_START_IDX_MASK) |
          (((uint64_t)(npsn)      << BNXT_RE_SQ_MSN_SEARCH_NEXT_PSN_SHIFT)
                                            & BNXT_RE_SQ_MSN_SEARCH_NEXT_PSN_MASK)  |
          (((uint64_t)(start_psn) << BNXT_RE_SQ_MSN_SEARCH_START_PSN_SHIFT)
                                            & BNXT_RE_SQ_MSN_SEARCH_START_PSN_MASK));
}

__device__ __forceinline__ void QueuePairBNXT::acquire_lock(uint32_t* lock) {
  uint32_t expected;

  do {
    expected = 0;
  } while (0 == __hip_atomic_compare_exchange_strong(lock, &expected, 1,
                                                     __ATOMIC_ACQUIRE,
                                                     __ATOMIC_ACQUIRE,
                                                     __HIP_MEMORY_SCOPE_SYSTEM));
}

__device__ __forceinline__ void QueuePairBNXT::release_lock(uint32_t* lock) {
  __hip_atomic_store(lock, 0, __ATOMIC_RELEASE, __HIP_MEMORY_SCOPE_SYSTEM);
}

#if defined(BUILD_DEBUG_DEVICE)
__device__ inline __noinline__ void QueuePairBNXT::print_cqe_error(uint8_t status) {
  switch (status) {
  case BNXT_RE_REQ_ST_BAD_RESP:
    LOGD_ERROR_ABORT("CQ error BAD_RESP (%x)", status);
    break;
  case BNXT_RE_REQ_ST_LOC_LEN:
    LOGD_ERROR_ABORT("CQ error LOC_LEN (%x)", status);
    break;
  case BNXT_RE_REQ_ST_LOC_QP_OP:
    LOGD_ERROR_ABORT("CQ error LOC_QP_OP (%x)", status);
    break;
  case BNXT_RE_REQ_ST_PROT:
    LOGD_ERROR_ABORT("CQ error PROT (%x)", status);
    break;
  case BNXT_RE_REQ_ST_MEM_OP:
    LOGD_ERROR_ABORT("CQ error MEM_OP (%x)", status);
    break;
  case BNXT_RE_REQ_ST_REM_INVAL:
    LOGD_ERROR_ABORT("CQ error REM_INVAL (%x)", status);
    break;
  case BNXT_RE_REQ_ST_REM_ACC:
    LOGD_ERROR_ABORT("CQ error REM_ACC (%x)", status);
    break;
  case BNXT_RE_REQ_ST_REM_OP:
    LOGD_ERROR_ABORT("CQ error REM_OP (%x)", status);
    break;
  case BNXT_RE_REQ_ST_RNR_NAK_XCED:
    LOGD_ERROR_ABORT("CQ error RNR_NAK_XCED (%x)", status);
    break;
  case BNXT_RE_REQ_ST_TRNSP_XCED:
    LOGD_ERROR_ABORT("CQ error TRNSP_XCED (%x)", status);
    break;
  case BNXT_RE_REQ_ST_WR_FLUSH:
    LOGD_ERROR_ABORT("CQ error WR_FLUSH (%x)", status);
    break;
  default:
    LOGD_ERROR_ABORT("CQ error unknown status (%x)", status);
    break;
  }
}
#endif  // BUILD_DEBUG_DEVICE

}  // namespace rocshmem

#endif  // LIBRARY_SRC_GDA_BNXT_QUEUE_PAIR_HPP_

