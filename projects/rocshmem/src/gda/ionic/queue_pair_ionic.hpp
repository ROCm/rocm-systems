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

#include "gda/endian.hpp"
#include "gda/ionic/provider_gda_ionic.hpp"
#include "gda/queue_pair.hpp"

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

  static constexpr size_t InlineThreshold = sizeof(ionic_v1_pld);
  static_assert(InlineThreshold == 32, "ionic can send up to 32 bytes inline in a WQE");
};

class QueuePairIONIC : public QueuePairBase<QueuePairIONIC> {
public:
  __host__ explicit QueuePairIONIC(uint32_t qpn, uintptr_t heap_laddr, uint32_t heap_lkey,
                                   uintptr_t heap_raddr, uint32_t heap_rkey, size_t heap_size,
                                   const QpSymmEntry *symm_entries, const int *symm_count,
                                   struct ibv_pd* pd,
                                   ionic_device_sq&& sq, ionic_device_cq&& cq);

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
   * @param wf_info Wavefront information.
   * @param num_wqes number of sq wqes to reserve for this wave.
   * @return position of my_tid=0's wqe.
   */
  __device__ uint32_t reserve_sq(const ActiveWFInfo& wf_info, uint32_t num_wqes);
  __device__ uint32_t reserve_sq_single(uint32_t num_wqes);

  /**
   * @brief Ring the sq doorbell maintaining order between waves.
   * @param wf_info Wavefront information.
   * @param my_sq_prod position of my_tid=0's wqe.
   * @param num_wqes number of sq wqes posted in this wave.
   * @param wqe this thread's wqe.
   * @return doorbell producer index.
   */
  __device__ uint32_t commit_sq(const ActiveWFInfo& wf_info, uint32_t my_sq_prod,
                                uint32_t num_wqes);
  __device__ uint32_t commit_sq_single(uint32_t my_sq_prod, uint32_t num_wqes);

  __device__ void ring_doorbell(uint32_t pos);
  __device__ void ring_doorbell_single(uint32_t pos);

  /**
   * @brief Helper method to poll the next completion queue entry.
   */
  __device__ void poll_wave_cqes(uint64_t activemask);

  /**
   * @brief Helper method to drain completion queue entries.
   * @param wf_info Wavefront information.
   * @param cons wait for sq.msn to catch up to this position.
   */
  __device__ void quiet_internal_ccqe(const ActiveWFInfo& wf_info, uint32_t cons);
  __device__ void quiet_internal_ccqe_single(uint32_t cons);

  /**
   * @brief Helper method to drain completion queue entries.
   * @param wf_info Wavefront information.
   * @param cons wait for sq.msn to catch up to this position.
   */
  __device__ void quiet_internal(const ActiveWFInfo& wf_info, uint32_t cons);

  ionic_device_sq sq;
  ionic_device_cq cq;

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
  uint32_t num_wqes = 1;
  if (wf_info.scope == ThreadScope::thread) {
    num_wqes = wf_info.num_pe_group_lanes;
  }

  uint32_t my_sq_prod = reserve_sq(wf_info, num_wqes);

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

  commit_sq(wf_info, my_sq_prod, num_wqes);
}

// precondition: called with all active lanes using different QPs
template <QueuePairIONIC::OpCode Op, typename... Options>
__device__ __noinline__ void QueuePairIONIC::post_wqe_rma_single(
    uintptr_t laddr, uint32_t lkey, uintptr_t raddr, uint32_t rkey, size_t size, PostOpt<Options...>) {
  //using PostOptions = PostOpt<Options...>;
  uint32_t num_wqes = 1;
  uint32_t my_sq_prod = reserve_sq_single(num_wqes);
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

  commit_sq_single(my_sq_prod, num_wqes);
}

// can be called with all active lanes using any number of different QPs, don't assume anything
template <QueuePairIONIC::OpCode Op, AMOFetchType Fetch, typename... Options>
__device__ __noinline__ QueuePairIONIC::amo_ret_t<Fetch> QueuePairIONIC::post_wqe_amo(
    uintptr_t raddr, uint32_t rkey, uint64_t swap_add, uint64_t compare,
    const ActiveWFInfo& wf_info, PostOpt<Options...>) {
  static_assert(Fetch != AMOFetchType::NonBlocking);
  //using PostOptions = PostOpt<Options...>;
  uint32_t num_wqes = wf_info.num_pe_group_lanes;
  uint32_t my_sq_prod = reserve_sq(wf_info, num_wqes);
  uint32_t my_sq_pos = my_sq_prod + wf_info.pe_group_logical_lane_id;
  struct ionic_v1_wqe *wqe = &sq.buf[my_sq_pos & sq.mask];
  uint16_t wqe_flags = 0;
  uint32_t cons;

  uint64_t* wave_fetch_atomic{nullptr};
  if constexpr (Fetch == AMOFetchType::Blocking) {
    if (wf_info.is_pe_group_first) {
      auto res = fetching_atomic_freelist->pop_front();
      while (!res.success) {
        res = fetching_atomic_freelist->pop_front();
      }
      wave_fetch_atomic = res.value;
    }
    wave_fetch_atomic = (uint64_t*)__shfl((uint64_t)wave_fetch_atomic,
                         wf_info.pe_group_first_phys_lane_id);
  }

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

  cons = commit_sq(wf_info, my_sq_prod, num_wqes);

  if constexpr (Fetch == AMOFetchType::Blocking) {
    quiet_internal(wf_info, cons);
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
  static_assert(Fetch != AMOFetchType::NonBlocking);
  //using PostOptions = PostOpt<Options...>;
  uint32_t num_wqes = 1;
  uint32_t my_sq_prod = reserve_sq_single(num_wqes);
  uint32_t my_sq_pos = my_sq_prod;
  struct ionic_v1_wqe *wqe = &sq.buf[my_sq_pos & sq.mask];
  uint16_t wqe_flags = 0;
  uint32_t cons;

  uint64_t* wave_fetch_atomic{nullptr};
  if constexpr (Fetch == AMOFetchType::Blocking) {
    auto res = fetching_atomic_freelist->pop_front();
    while (!res.success) {
      res = fetching_atomic_freelist->pop_front();
    }
    wave_fetch_atomic = res.value;
  }

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

  cons = commit_sq_single(my_sq_prod, num_wqes);

  if constexpr (Fetch == AMOFetchType::Blocking) {
    quiet_internal_ccqe_single(cons);
    uint64_t ret = wave_fetch_atomic[0];
    __atomic_signal_fence(__ATOMIC_SEQ_CST);
    fetching_atomic_freelist->push_back(wave_fetch_atomic);
    return ret;
  }
}

}  // namespace rocshmem

#endif  // LIBRARY_SRC_GDA_IONIC_QUEUE_PAIR_HPP_
