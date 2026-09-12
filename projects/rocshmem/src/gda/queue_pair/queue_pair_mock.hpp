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

#ifndef LIBRARY_SRC_GDA_QUEUE_PAIR_MOCK_HPP_
#define LIBRARY_SRC_GDA_QUEUE_PAIR_MOCK_HPP_

#include <hip/hip_runtime.h>

#include "gda/endian.hpp"

#include "queue_pair_common.hpp"
#include "queue_pair_interface.hpp"
#include "queue_pair_option.hpp"
#include "queue_pair_shmem.hpp"

namespace rocshmem {

class QueuePairMock;

template <> struct QueuePairTraits<QueuePairMock> {
  /**
   * @brief QueuePair mock object OpCode enumeration.
   */
  enum class OpCode {
    RDMA_WRITE,
    RDMA_READ,
    ATOMIC_CS,
    ATOMIC_FA,
  };

  /**
   * @brief QueuePair mock object uses native ordering.
   */
  static constexpr endian::Order Endianness = endian::Order::Native;

  /**
   * @brief QueuePair mock object inlining maximum is 16 bytes.
   */
  static constexpr size_t InlineMax = sizeof(__uint128_t);


  /*
   * @brief QueuePair mock object preferred inlining threshold is 8 bytes.
   */
  static constexpr size_t InlineThreshold = sizeof(uint64_t);
};

class QueuePairMock : public QueuePairSHMEM<QueuePairMock> {
public:
  static __device__ inline size_t rma_count{0};
  static __device__ inline size_t rma_inline_count{0};
  static __device__ inline size_t amo_count{0};
  static __device__ inline size_t quiet_count{0};

public:
  __host__ QueuePairMock()                                      = default;
  __host__ QueuePairMock(const QueuePairMock& other)            = delete;
  __host__ QueuePairMock& operator=(const QueuePairMock& other) = delete;
  __host__ QueuePairMock(QueuePairMock&& other) noexcept        = default;
  __host__ QueuePairMock& operator=(QueuePairMock&& other)      = default;
  __host__ ~QueuePairMock()                                     = default;

public:
  template <OpCode Op, typename... Options>
  __device__ __forceinline__
  void post_wqe_rma(uintptr_t laddr, uint32_t lkey,
                    uintptr_t raddr, uint32_t rkey, size_t size,
                    const ActiveWFInfo& wf_info, PostOpt<Options...> = {});

  template <OpCode Op, typename... Options>
  __device__ __forceinline__
  void post_wqe_rma_single(uintptr_t laddr, uint32_t lkey,
                           uintptr_t raddr, uint32_t rkey, size_t size,
                           PostOpt<Options...> = {});

  template <OpCode Op, AMOFetchType Fetch, typename... Options>
  __device__ __forceinline__
  amo_ret_t<Fetch> post_wqe_amo(uintptr_t raddr, uint32_t rkey,
                                uint64_t swap_add, uint64_t compare,
                                const ActiveWFInfo& wf_info, PostOpt<Options...> = {});

  template <OpCode Op, AMOFetchType Fetch, typename... Options>
  __device__ __forceinline__
  amo_ret_t<Fetch> post_wqe_amo_single(uintptr_t raddr, uint32_t rkey,
                                       uint64_t swap_add, uint64_t compare,
                                       PostOpt<Options...> = {});

  __device__ __forceinline__ void quiet_single();


  /**
   * @brief Resolve the local (origin) virtual address and LKey of a symmetric address.
   * QueuePair mock object returns the local (origin) virtual address unmodified and LKey = 0.
   *
   * @param[in] addr Symmetric address to resolve.
   *
   * @return {reinterpret_cast<uintptr_t>(addr), 0}
   */
  __device__ __forceinline__
  std::tuple<uintptr_t, uint32_t> get_laddr_info(const void *addr, bool inlined = false) const {
    uintptr_t laddr = reinterpret_cast<uintptr_t>(addr);
    return {laddr, 0};
  }

  /**
   * @brief Resolve the remote (target) virtual address and RKey of a symmetric address.
   * QueuePair mock object returns the remote (target) virtual address unmodified and LKey = 0.
   *
   * @param[in] addr Symmetric address to resolve.
   *
   * @return {reinterpret_cast<uintptr_t>(addr), 0}
   */
  __device__ __forceinline__
  std::tuple<uintptr_t, uint32_t> get_raddr_info(const void *addr) const {
    uintptr_t laddr = reinterpret_cast<uintptr_t>(addr);
    return {laddr, 0};
  }

  /*
   * @brief Query whether data can be inlined into a WQE.
   *
   * @param[in] size Size of data carried by the WQE.
   * @tparam Op OpCode for this WQE.
   *
   * @return True if size bytes of data can be inlined into a WQE with OpCode Op, else false.
   */
  template <OpCode Op>
  static __host__ __device__ __forceinline__ constexpr bool can_inline(size_t size) {
    if constexpr (Op == OpCode::RDMA_WRITE) {
      return size <= Traits::InlineThreshold;
    } else {
      return  false;
    }
  }

  /**
   * @brief Convert value to Endianness of provider, byteswapping if necessary.
   * QueuePair mock object returns the value unmodified: mock objects use native endian ordering.
   *
   * @param[in] val Value to convert.
   * @tparam T Type of val.
   *
   * @return val
   */
  template <typename T>
  static __host__ __device__ __forceinline__ constexpr T to_provider_endianness(T val) {
    return val;
  }
};



template <QueuePairMock::OpCode Op, typename... Options>
__device__ __forceinline__ void QueuePairMock::post_wqe_rma(
    uintptr_t laddr, uint32_t lkey, uintptr_t raddr, uint32_t rkey, size_t size,
    const ActiveWFInfo& wf_info, PostOpt<Options...> options) {
  if (!laddr || !raddr) {
    return;
  }

  if (wf_info.is_pe_group_first) {
    __hip_atomic_fetch_add(&rma_count, wf_info.num_pe_group_lanes,
                           __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT);
  }

  if (can_inline<Op>(size)) {
    __hip_atomic_fetch_add(&rma_inline_count, 1, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT);
  }

  if constexpr (Op == OpCode::RDMA_WRITE) {
    // RDMA_WRITE: dest is raddr, src is laddr
    memcpy(reinterpret_cast<void*>(raddr), reinterpret_cast<const void*>(laddr), size);
  } else if constexpr (Op == OpCode::RDMA_READ) {
    // RDMA_READ:  dest is laddr, src is raddr
    memcpy(reinterpret_cast<void*>(laddr), reinterpret_cast<const void*>(raddr), size);
  } else {
    static_assert(false, "invalid OpCode: post_wqe_rma only supports RDMA_WRITE and RDMA_READ");
  }
}

template <QueuePairMock::OpCode Op, typename... Options>
__device__ __forceinline__ void QueuePairMock::post_wqe_rma_single(
    uintptr_t laddr, uint32_t lkey, uintptr_t raddr, uint32_t rkey, size_t size,
    PostOpt<Options...> options) {
  if (!laddr || !raddr) {
    return;
  }

  __hip_atomic_fetch_add(&rma_count, 1, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT);

  if (can_inline<Op>(size)) {
    __hip_atomic_fetch_add(&rma_inline_count, 1, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT);
  }

  if constexpr (Op == OpCode::RDMA_WRITE) {
    // RDMA_WRITE: dest is raddr, src is laddr
    memcpy(reinterpret_cast<void*>(raddr), reinterpret_cast<const void*>(laddr), size);
  } else if constexpr (Op == OpCode::RDMA_READ) {
    // RDMA_READ:  dest is laddr, src is raddr
    memcpy(reinterpret_cast<void*>(laddr), reinterpret_cast<const void*>(raddr), size);
  } else {
    static_assert(false, "invalid OpCode: post_wqe_rma_single only supports RDMA_WRITE and RDMA_READ");
  }
}

template <QueuePairMock::OpCode Op, AMOFetchType Fetch, typename... Options>
__device__ __forceinline__ QueuePairMock::amo_ret_t<Fetch> QueuePairMock::post_wqe_amo(
    uintptr_t raddr, uint32_t rkey, uint64_t swap_add, uint64_t compare,
    const ActiveWFInfo& wf_info, PostOpt<Options...> options) {
  static_assert(Fetch != AMOFetchType::NonBlocking, "non-blocking AMOs not yet implemented");
  if (!raddr) {
    // Returns void{}, a prvalue of type void; or uint64_t{}, which performs zero-initialization
    return amo_ret_t<Fetch>{};
  }

  if (wf_info.is_pe_group_first) {
    __hip_atomic_fetch_add(&amo_count, wf_info.num_pe_group_lanes,
                           __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT);
  }

  if constexpr (Op == OpCode::ATOMIC_CS) {
    // ATOMIC_CS: obj is raddr, expected is &compare, desired is swap_add
    __hip_atomic_compare_exchange_strong(reinterpret_cast<uint64_t*>(raddr), &compare, swap_add,
                                         __ATOMIC_RELAXED, __ATOMIC_RELAXED,
                                         __HIP_MEMORY_SCOPE_AGENT);
  } else if constexpr (Op == OpCode::ATOMIC_FA) {
    // ATOMIC_FA: obj is raddr, arg is swap_add; compare is otherwise unused, use for fetch val
    assert(compare == 0);
    compare = __hip_atomic_fetch_add(reinterpret_cast<uint64_t*>(raddr), swap_add,
                                     __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT);
  } else {
    static_assert(false, "invalid OpCode: post_wqe_amo only supports ATOMIC_CS and ATOMIC_FA");
  }

  if constexpr (Fetch == AMOFetchType::Blocking) {
    quiet(wf_info);
    /* compare either:
     *   a) unmodified, if ATOMIC_CS and atomic_compare_exchange succeeded (i.e. same as *raddr)
     *   b) load raddr, if ATOMIC_CS and atomic_compare_exchange failed
     *   c) prior value at raddr, if ATOMIC_FA */
    return compare;
  }
}

template <QueuePairMock::OpCode Op, AMOFetchType Fetch, typename... Options>
__device__ __forceinline__ QueuePairMock::amo_ret_t<Fetch> QueuePairMock::post_wqe_amo_single(
    uintptr_t raddr, uint32_t rkey, uint64_t swap_add, uint64_t compare,
    PostOpt<Options...> options) {
  static_assert(Fetch != AMOFetchType::NonBlocking, "non-blocking AMOs not yet implemented");
  if (!raddr) {
    // Returns void{}, a prvalue of type void; or uint64_t{}, which performs zero-initialization
    return amo_ret_t<Fetch>{};
  }

  __hip_atomic_fetch_add(&amo_count, 1, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT);

  if constexpr (Op == OpCode::ATOMIC_CS) {
    // ATOMIC_CS: obj is raddr, expected is &compare, desired is swap_add
    __hip_atomic_compare_exchange_strong(reinterpret_cast<uint64_t*>(raddr), &compare, swap_add,
                                         __ATOMIC_RELAXED, __ATOMIC_RELAXED,
                                         __HIP_MEMORY_SCOPE_AGENT);
  } else if constexpr (Op == OpCode::ATOMIC_FA) {
    // ATOMIC_FA: obj is raddr, arg is swap_add; compare is otherwise unused, use for fetch val
    assert(compare == 0);
    compare = __hip_atomic_fetch_add(reinterpret_cast<uint64_t*>(raddr), swap_add,
                                     __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT);
  } else {
    static_assert(false, "invalid OpCode: post_wqe_amo_single only supports ATOMIC_CS and ATOMIC_FA");
  }

  if constexpr (Fetch == AMOFetchType::Blocking) {
    quiet_single();
    /* compare either:
     *   a) unmodified, if ATOMIC_CS and atomic_compare_exchange succeeded (i.e. same as *raddr)
     *   b) load raddr, if ATOMIC_CS and atomic_compare_exchange failed
     *   c) prior value at raddr, if ATOMIC_FA */
    return compare;
  }
}

__device__ __forceinline__ void QueuePairMock::quiet_single() {
  // Release ensures that prior stores and RMW occur before the increment of quiet_count
  __hip_atomic_fetch_add(&quiet_count, 1, __ATOMIC_RELEASE, __HIP_MEMORY_SCOPE_AGENT);
}

}  // namespace rocshmem

#endif  // LIBRARY_SRC_GDA_QUEUE_PAIR_MOCK_HPP_
