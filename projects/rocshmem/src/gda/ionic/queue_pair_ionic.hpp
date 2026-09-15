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

#ifndef LIBRARY_SRC_GDA_IONIC_QUEUE_PAIR_HPP_
#define LIBRARY_SRC_GDA_IONIC_QUEUE_PAIR_HPP_

#include <utility>

#include "log.hpp"
#include "util.hpp"

#include "gda/endian.hpp"
#include "gda/ionic/provider_gda_ionic.hpp"
#include "gda/queue_pair/queue_pair_device.hpp"

namespace rocshmem {

class QueuePairIONIC;

template <> struct QueuePairTraits<QueuePairIONIC> {
  enum class OpCode : uint8_t {
    RDMA_WRITE = IONIC_V2_OP_RDMA_WRITE,
    RDMA_READ  = IONIC_V2_OP_RDMA_READ,
    ATOMIC_CS  = IONIC_V2_OP_ATOMIC_CS,
    ATOMIC_FA  = IONIC_V2_OP_ATOMIC_FA,
  };

  /**
   * @brief ionic uses big-endian ordering
   */
  static constexpr endian::Order Endianness = endian::Order::Big;

  /**
   * @brief ionic inlining maximum is the WQE's payload segment, sizeof(ionic_v1_pld) = 32 bytes
   */
  static constexpr size_t InlineMax = sizeof(ionic_v1_pld);
  static_assert(InlineMax == 32, "ionic can send up to 32 bytes inline in a WQE");

  /*
   * @brief ionic preferred inlining threshold is the same as the maximum
   */
  static constexpr size_t InlineThreshold = InlineMax;
};

class QueuePairIONIC : public QueuePairDevice<QueuePairIONIC> {
private:
  ionic_device_sq sq;
  ionic_device_cq cq;

public:
  __host__ explicit QueuePairIONIC(uint32_t qpn, uintptr_t heap_laddr, uint32_t heap_lkey,
                                   uintptr_t heap_raddr, uint32_t heap_rkey, size_t heap_size,
                                   uint64_t *fetching_atomic, uint32_t fetching_atomic_lkey,
                                   uint64_t *nonfetching_atomic, uint32_t nonfetching_atomic_lkey,
                                   FreeList<uint64_t*> *fetching_atomic_freelist,
                                   const BufferInfo *local_buffers, size_t num_user_buffers,
                                   const SymmBufferInfo *symm_buffers, const int *symm_count,
                                   ionic_device_sq&& sq, ionic_device_cq&& cq)
    : QueuePairDevice{qpn, heap_laddr, heap_lkey, heap_raddr, heap_rkey, heap_size,
                      fetching_atomic, fetching_atomic_lkey,
                      nonfetching_atomic, nonfetching_atomic_lkey,
                      fetching_atomic_freelist,
                      local_buffers, num_user_buffers,
                      symm_buffers, symm_count},
      sq{std::move(sq)}, cq{std::move(cq)} { }

  __host__ explicit QueuePairIONIC(uint32_t qpn,
                                   uint64_t *nonfetching_atomic, uint32_t nonfetching_atomic_lkey,
                                   ionic_device_sq&& sq, ionic_device_cq&& cq)
    : QueuePairDevice{qpn, nonfetching_atomic, nonfetching_atomic_lkey},
      sq{std::move(sq)}, cq{std::move(cq)} { }

  __host__ QueuePairIONIC(const QueuePairIONIC& other)            = delete;
  __host__ QueuePairIONIC& operator=(const QueuePairIONIC& other) = delete;
  __host__ QueuePairIONIC(QueuePairIONIC&& other) noexcept        = default;
  __host__ QueuePairIONIC& operator=(QueuePairIONIC&& other)      = default;
  __host__ ~QueuePairIONIC()                                      = default;

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
  /**
   * @brief Reserve space in the sq to post this many wqes.
   * @param num_wqes number of sq wqes to reserve for this wave.
   * @return position of my_tid=0's wqe.
   */
  __device__ uint32_t reserve_sq(uint32_t num_wqes);

  /**
   * @brief Ring the sq doorbell maintaining order between waves.
   * @param my_sq_prod position of my_tid=0's wqe.
   * @param num_wqes number of sq wqes posted in this wave.
   * @param wqe this thread's wqe.
   * @return doorbell producer index.
   */
  __device__ uint32_t commit_sq(uint32_t my_sq_prod, uint32_t num_wqes);

  __device__ void ring_doorbell(uint32_t pos);

  /**
   * @brief Helper method to drain completion queue entries.
   * @param cons wait for sq.msn to catch up to this position.
   */
  __device__ void poll_cq_until(uint32_t cons);

  enum ionic_v1_cqe_qtf_bits_be : uint32_t {
    IONIC_V1_CQE_COLOR_BE = endian::to_be<uint32_t>(IONIC_V1_CQE_COLOR),
    IONIC_V1_CQE_ERROR_BE = endian::to_be<uint32_t>(IONIC_V1_CQE_ERROR),
  };

  enum ionic_v1_op_be : uint16_t {
    IONIC_V1_FLAG_INL_BE   = endian::to_be<uint16_t>(IONIC_V1_FLAG_INL),
    IONIC_V1_FLAG_SIG_BE   = endian::to_be<uint16_t>(IONIC_V1_FLAG_SIG),
    IONIC_V1_FLAG_COLOR_BE = endian::to_be<uint16_t>(IONIC_V1_FLAG_COLOR),
  };
};

// can be called with all active lanes using any number of different QPs, don't assume anything
template <QueuePairIONIC::OpCode Op, typename... Options>
__device__ __noinline__ void QueuePairIONIC::post_wqe_rma(
    uintptr_t laddr, uint32_t lkey, uintptr_t raddr, uint32_t rkey, size_t size,
    const ActiveWFInfo& wf_info, PostOpt<Options...>) {
  //using PostOptions = PostOpt<Options...>;
  uint32_t num_wqes = wf_info.num_pe_group_lanes;
  uint32_t my_sq_prod = 0;
  if (wf_info.is_pe_group_first) {
    my_sq_prod = reserve_sq(num_wqes);
  }
  my_sq_prod = __shfl(my_sq_prod, wf_info.pe_group_first_phys_lane_id);
  uint32_t my_sq_pos = my_sq_prod + wf_info.pe_group_logical_lane_id;
  struct ionic_v1_wqe *wqe = &sq.buf[my_sq_pos & sq.mask];
  uint16_t wqe_flags = 0;

  if (!(my_sq_pos & (sq.mask + 1))) {
    wqe_flags |= IONIC_V1_FLAG_COLOR_BE;
  }

  if (wf_info.is_pe_group_last) {
    wqe_flags |= IONIC_V1_FLAG_SIG_BE;
  }

  // TODO why is this needed?
  if constexpr (Op == OpCode::RDMA_WRITE) {
    if (size && !laddr) {
      size = 1;
    }
  }

  wqe->base.wqe_idx = my_sq_pos;
  wqe->base.op = static_cast<uint8_t>(Op);
  wqe->base.num_sge_key = size ? 1 : 0;
  wqe->base.imm_data_key = 0;

  wqe->common.rdma.remote_va_high = endian::to_be<uint32_t>(raddr >> 32);
  wqe->common.rdma.remote_va_low  = endian::to_be<uint32_t>(raddr);
  wqe->common.rdma.remote_rkey    = rkey;
  wqe->common.length              = endian::to_be<uint32_t>(size);

  if (size) {
    bool send_inline = can_inline<Op>(size);
    if (send_inline) {
      wqe_flags |= IONIC_V1_FLAG_INL_BE;
      wqe->base.num_sge_key = 0;
      if (!laddr) {
        // TODO why is this needed?
        wqe->common.pld.data[0] = 1;
      } else {
        memcpy(wqe->common.pld.data, reinterpret_cast<const void*>(laddr), size);
      }
    } else {
      wqe->common.pld.sgl[0].va   = endian::to_be<uint64_t>(laddr);
      wqe->common.pld.sgl[0].len  = endian::to_be<uint32_t>(size);
      wqe->common.pld.sgl[0].lkey = lkey;
    }
  }

  __hip_atomic_store(&wqe->base.flags, wqe_flags, __ATOMIC_RELEASE,
    __HIP_MEMORY_SCOPE_AGENT);

  if (wf_info.is_pe_group_first) {
    commit_sq(my_sq_prod, num_wqes);
  }
}

// precondition: called with all active lanes using different QPs
template <QueuePairIONIC::OpCode Op, typename... Options>
__device__ __noinline__ void QueuePairIONIC::post_wqe_rma_single(
    uintptr_t laddr, uint32_t lkey, uintptr_t raddr, uint32_t rkey, size_t size, PostOpt<Options...>) {
  //using PostOptions = PostOpt<Options...>;
  static constexpr uint32_t num_wqes = 1;
  uint32_t my_sq_prod = reserve_sq(num_wqes);
  uint32_t my_sq_pos = my_sq_prod;
  struct ionic_v1_wqe *wqe = &sq.buf[my_sq_pos & sq.mask];
  uint16_t wqe_flags = 0;

  if (!(my_sq_pos & (sq.mask + 1))) {
    wqe_flags |= IONIC_V1_FLAG_COLOR_BE;
  }

  wqe_flags |= IONIC_V1_FLAG_SIG_BE;

  // TODO why is this needed?
  if constexpr (Op == OpCode::RDMA_WRITE) {
    if (size && !laddr) {
      size = 1;
    }
  }

  wqe->base.wqe_idx = my_sq_pos;
  wqe->base.op = static_cast<uint8_t>(Op);
  wqe->base.num_sge_key = size ? 1 : 0;
  wqe->base.imm_data_key = 0;

  wqe->common.rdma.remote_va_high = endian::to_be<uint32_t>(raddr >> 32);
  wqe->common.rdma.remote_va_low  = endian::to_be<uint32_t>(raddr);
  wqe->common.rdma.remote_rkey    = rkey;
  wqe->common.length              = endian::to_be<uint32_t>(size);

  if (size) {
    bool send_inline = can_inline<Op>(size);
    if (send_inline) {
      wqe_flags |= IONIC_V1_FLAG_INL_BE;
      wqe->base.num_sge_key = 0;
      if (!laddr) {
        // TODO why is this needed?
        wqe->common.pld.data[0] = 1;
      } else {
        memcpy(wqe->common.pld.data, reinterpret_cast<const void*>(laddr), size);
      }
    } else {
      wqe->common.pld.sgl[0].va   = endian::to_be<uint64_t>(laddr);
      wqe->common.pld.sgl[0].len  = endian::to_be<uint32_t>(size);
      wqe->common.pld.sgl[0].lkey = lkey;
    }
  }

  __hip_atomic_store(&wqe->base.flags, wqe_flags, __ATOMIC_RELEASE, __HIP_MEMORY_SCOPE_AGENT);

  commit_sq(my_sq_prod, num_wqes);
}

// can be called with all active lanes using any number of different QPs, don't assume anything
template <QueuePairIONIC::OpCode Op, AMOFetchType Fetch, typename... Options>
__device__ __noinline__ QueuePairIONIC::amo_ret_t<Fetch> QueuePairIONIC::post_wqe_amo(
    uintptr_t raddr, uint32_t rkey, uint64_t swap_add, uint64_t compare,
    const ActiveWFInfo& wf_info, PostOpt<Options...>) {
  static_assert(Fetch != AMOFetchType::NonBlocking, "non-blocking AMOs not yet implemented");
  //using PostOptions = PostOpt<Options...>;
  uint32_t num_wqes = wf_info.num_pe_group_lanes;
  uint32_t my_sq_prod = 0;
  uint64_t* wave_fetch_atomic = nullptr;

  if (wf_info.is_pe_group_first) {
    my_sq_prod = reserve_sq(num_wqes);
    if constexpr (Fetch == AMOFetchType::Blocking) {
      using PopBackResult = decltype(fetching_atomic_freelist->pop_front());
      PopBackResult res{nullptr, false};
      do {
        res = fetching_atomic_freelist->pop_front();
      } while (!res.success);
      wave_fetch_atomic = res.value;
    }
  }

  if constexpr (Fetch == AMOFetchType::Blocking) {
    wave_fetch_atomic =
        reinterpret_cast<uint64_t*>(__shfl(reinterpret_cast<uintptr_t>(wave_fetch_atomic),
                                           wf_info.pe_group_first_phys_lane_id));
  }

  my_sq_prod = __shfl(my_sq_prod, wf_info.pe_group_first_phys_lane_id);
  uint32_t my_sq_pos = my_sq_prod + wf_info.pe_group_logical_lane_id;
  struct ionic_v1_wqe *wqe = &sq.buf[my_sq_pos & sq.mask];
  uint16_t wqe_flags = 0;

  if (!(my_sq_pos & (sq.mask + 1))) {
    wqe_flags |= IONIC_V1_FLAG_COLOR_BE;
  }

  if (wf_info.is_pe_group_last) {
    wqe_flags |= IONIC_V1_FLAG_SIG_BE;
  }

  wqe->base.wqe_idx = my_sq_pos;
  wqe->base.op = static_cast<uint8_t>(Op);
  wqe->base.num_sge_key = 1;
  wqe->base.imm_data_key = 0;

  wqe->atomic_v2.remote_va_high = endian::to_be<uint32_t>(raddr >> 32);
  wqe->atomic_v2.remote_va_low  = endian::to_be<uint32_t>(raddr);
  wqe->atomic_v2.remote_rkey    = rkey;
  wqe->atomic_v2.swap_add_high  = endian::to_be<uint32_t>(swap_add >> 32);
  wqe->atomic_v2.swap_add_low   = endian::to_be<uint32_t>(swap_add);
  wqe->atomic_v2.compare_high   = endian::to_be<uint32_t>(compare >> 32);
  wqe->atomic_v2.compare_low    = endian::to_be<uint32_t>(compare);

  uint64_t* atomic_laddr = nonfetching_atomic;
  if constexpr (Fetch == AMOFetchType::Blocking) {
    atomic_laddr = wave_fetch_atomic + wf_info.pe_group_logical_lane_id;
  }
  wqe->atomic_v2.local_va = endian::to_be(reinterpret_cast<uintptr_t>(atomic_laddr));
  wqe->atomic_v2.lkey     = get_atomic_lkey<Fetch>();

  __hip_atomic_store(&wqe->base.flags, wqe_flags, __ATOMIC_RELEASE,
    __HIP_MEMORY_SCOPE_AGENT);

  if (wf_info.is_pe_group_first) {
    uint32_t cons = commit_sq(my_sq_prod, num_wqes);
    if constexpr (Fetch == AMOFetchType::Blocking) {
      poll_cq_until(cons);
    }
  }

  if constexpr (Fetch == AMOFetchType::Blocking) {
    uint64_t ret = wave_fetch_atomic[wf_info.pe_group_logical_lane_id];
    __atomic_signal_fence(__ATOMIC_SEQ_CST);
    if (wf_info.is_pe_group_first) {
      fetching_atomic_freelist->push_back(wave_fetch_atomic);
    }
    return ret;
  }
}

// precondition: called with all active lanes using different QPs
template <QueuePairIONIC::OpCode Op, AMOFetchType Fetch, typename... Options>
__device__ __noinline__ QueuePairIONIC::amo_ret_t<Fetch> QueuePairIONIC::post_wqe_amo_single(
    uintptr_t raddr, uint32_t rkey, uint64_t swap_add, uint64_t compare, PostOpt<Options...>) {
  static_assert(Fetch != AMOFetchType::NonBlocking, "non-blocking AMOs not yet implemented");
  //using PostOptions = PostOpt<Options...>;
  static constexpr uint32_t num_wqes = 1;
  uint32_t my_sq_prod = reserve_sq(num_wqes);
  uint64_t* wave_fetch_atomic = nullptr;

  if constexpr (Fetch == AMOFetchType::Blocking) {
    using PopBackResult = decltype(fetching_atomic_freelist->pop_front());
    PopBackResult res{nullptr, false};
    do {
      res = fetching_atomic_freelist->pop_front();
    } while (!res.success);
    wave_fetch_atomic = res.value;
  }

  uint32_t my_sq_pos = my_sq_prod;
  struct ionic_v1_wqe *wqe = &sq.buf[my_sq_pos & sq.mask];
  uint16_t wqe_flags = 0;

  if (!(my_sq_pos & (sq.mask + 1))) {
    wqe_flags |= IONIC_V1_FLAG_COLOR_BE;
  }

  wqe_flags |= IONIC_V1_FLAG_SIG_BE;

  wqe->base.wqe_idx = my_sq_pos;
  wqe->base.op = static_cast<uint8_t>(Op);
  wqe->base.num_sge_key = 1;
  wqe->base.imm_data_key = 0;

  wqe->atomic_v2.remote_va_high = endian::to_be<uint32_t>(raddr >> 32);
  wqe->atomic_v2.remote_va_low  = endian::to_be<uint32_t>(raddr);
  wqe->atomic_v2.remote_rkey    = rkey;
  wqe->atomic_v2.swap_add_high  = endian::to_be<uint32_t>(swap_add >> 32);
  wqe->atomic_v2.swap_add_low   = endian::to_be<uint32_t>(swap_add);
  wqe->atomic_v2.compare_high   = endian::to_be<uint32_t>(compare >> 32);
  wqe->atomic_v2.compare_low    = endian::to_be<uint32_t>(compare);

  uint64_t* atomic_laddr = nonfetching_atomic;
  if constexpr (Fetch == AMOFetchType::Blocking) {
    atomic_laddr = wave_fetch_atomic;
  }
  wqe->atomic_v2.local_va = endian::to_be(reinterpret_cast<uintptr_t>(atomic_laddr));
  wqe->atomic_v2.lkey     = get_atomic_lkey<Fetch>();

  __hip_atomic_store(&wqe->base.flags, wqe_flags, __ATOMIC_RELEASE, __HIP_MEMORY_SCOPE_AGENT);

  uint32_t cons = commit_sq(my_sq_prod, num_wqes);

  if constexpr (Fetch == AMOFetchType::Blocking) {
    poll_cq_until(cons);
    uint64_t ret = wave_fetch_atomic[0];
    __atomic_signal_fence(__ATOMIC_SEQ_CST);
    fetching_atomic_freelist->push_back(wave_fetch_atomic);
    return ret;
  }
}

// precondition: called with all active lanes using different QPs
__device__ inline __noinline__ void QueuePairIONIC::quiet_single() {
  poll_cq_until(sq.pos);
}

// precondition: called with all active lanes using different QPs
__device__ __forceinline__ uint32_t QueuePairIONIC::reserve_sq(uint32_t num_wqes) {
  // reserve space for wqes in sq
  uint32_t my_sq_prod = __hip_atomic_fetch_add(&sq.pos, num_wqes,
                                               __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT);
  // wait for that space to be available
  poll_cq_until(my_sq_prod + num_wqes - sq.mask);
  return my_sq_prod;
}

// precondition: called with all active lanes using different QPs
__device__ __forceinline__ uint32_t QueuePairIONIC::commit_sq(
    uint32_t my_sq_prod, uint32_t num_wqes) {
  uint32_t dbprod = my_sq_prod + num_wqes;

  spin_lock_acquire_unique(&sq.lock);
  if ((sq.dbpos - dbprod) & (1u << 31)) {
    sq.dbpos = dbprod;

    ring_doorbell(dbprod);
  }
  spin_lock_release_unique(&sq.lock);

  return dbprod;
}

// precondition: called with all active lanes using different QPs
__device__ __forceinline__ void QueuePairIONIC::ring_doorbell(uint32_t pos) {
  /*
   * Notes on ionic doorbell ringing:
   *   - The size of doorbell page is 4 KiB.
   *   - The size of a doorbell is 8 B.
   *   - There are 8 queue types (qtype0, qtype1, qtype2, ..., qtype7).
   *   - Each queue type has a separate doorbell.
   *   - Doorbells repeat across the entire page,
   *     i.e. doorbell for qtype7 is followed by qtype0 again:
   *     [db0, db1, db2, db3, db4, db5, db6, db7, db0, db1, ..., db7, db0, db1, ...]
   *   - A queue can be signalled using any doorbell for that queue type.
   *
   * In conclusion:
   * A doorbell page has 64 doorbells for each queue type, strided by 8 doorbells (64 B).
   * Since wf_size <= 64, each lane can ring using a different doorbell in the doorbell page.
   * This avoids the missed-write issue when multiple lanes write to the same address.
   */
  static constexpr size_t num_qtypes = 8;
  size_t lane_db_offset = __lane_id() * num_qtypes;

  uint64_t dbval = sq.dbval | (pos & sq.mask);
  __hip_atomic_store(&sq.dbreg[lane_db_offset], dbval,
                     __ATOMIC_RELEASE, __HIP_MEMORY_SCOPE_SYSTEM);
}

// precondition: called with all active lanes using different QPs
__device__ inline void QueuePairIONIC::poll_cq_until(uint32_t cons) {
  volatile struct ionic_v1_cqe *cqe = &cq.buf[0];
  uint32_t qtf_be = cqe->qid_type_flags;
  uint32_t msn = endian::from_be(cqe->send.msg_msn);
  while ((msn - cons) & 0x800000) {
    if (!!(qtf_be & IONIC_V1_CQE_ERROR_BE)) {
      break;
    }

    qtf_be = cqe->qid_type_flags;
    msn = endian::from_be(cqe->send.msg_msn);
  }

  if (!!(qtf_be & IONIC_V1_CQE_ERROR_BE)) {
#if defined(BUILD_DEBUG_DEVICE)
    uint32_t qtf = endian::from_be(qtf_be);
    uint32_t qid = qtf >> IONIC_V1_CQE_QID_SHIFT;
    uint32_t type = (qtf >> IONIC_V1_CQE_TYPE_SHIFT) & IONIC_V1_CQE_TYPE_MASK;
    uint32_t flag = qtf & 0xf;
    uint32_t status = endian::from_be(cqe->status_length);
    uint64_t npg = cqe->send.npg_wqe_idx_timestamp & IONIC_V1_CQE_WQE_IDX_MASK;

    LOGD_ERROR("QUIET ERROR (CCQE): qid %u type %u flag %#x status %u msn %u npg %lu",
               qid, type, flag, status, msn, npg);
#endif
    /* No other way to signal an error, so just crash. */
    abort();
  }
}

}  // namespace rocshmem

#endif  // LIBRARY_SRC_GDA_IONIC_QUEUE_PAIR_HPP_
