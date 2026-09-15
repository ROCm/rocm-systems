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

#ifndef LIBRARY_SRC_GDA_QUEUE_PAIR_SHMEM_HPP_
#define LIBRARY_SRC_GDA_QUEUE_PAIR_SHMEM_HPP_

/**
 * @file queue_pair_shmem.hpp
 *
 * @section DESCRIPTION
 * An IB QueuePair (SQ and CQ) that the device can use to perform network
 * operations. Most important rocSHMEM operations are performed by this
 * class.
 */

#include <cstddef>
#include <cstdint>

#include <type_traits>

#include <hip/hip_runtime.h>

#include "queue_pair_common.hpp"
#include "queue_pair_interface.hpp"
#include "queue_pair_option.hpp"

namespace rocshmem {

/*
 * @brief CRTP mixin class supplying a common SHMEM-like interface to Queue Pair implementations.
 */
template <typename Provider>
class QueuePairSHMEM : public QueuePairInterface<Provider> {
/**
 * @name Inherited Members
 *
 * Members that are inherited from QueuePairInterface<Provider>.
 *
 * @{
 */
public:
  using typename QueuePairInterface<Provider>::Traits;
  using typename QueuePairInterface<Provider>::OpCode;

  using QueuePairInterface<Provider>::put_nbi;
  using QueuePairInterface<Provider>::put_nbi_single;
  using QueuePairInterface<Provider>::get_nbi;
  using QueuePairInterface<Provider>::get_nbi_single;
  using QueuePairInterface<Provider>::atomic_fetch_add;
  using QueuePairInterface<Provider>::atomic_fetch_add_single;
  using QueuePairInterface<Provider>::atomic_add;
  using QueuePairInterface<Provider>::atomic_add_single;
  using QueuePairInterface<Provider>::atomic_compare_swap;
  using QueuePairInterface<Provider>::atomic_compare_swap_single;
  using QueuePairInterface<Provider>::quiet;
/**@}*/



/**
 * @name Internal
 *
 * @{
 */
protected:
  /*
   * @brief Provider is a friend of QueuePairSHMEM<Provider>.
   */
  friend Provider;

  __host__  QueuePairSHMEM() = default;
  __device__ QueuePairSHMEM() = delete;

private:
  __device__ Provider& provider() {
    return static_cast<Provider&>(*this);
  }
/**@}*/



/**
 * @name Remote Memory Access (RMA)
 *
 * @{
 */
public:
  /**
   * @brief Create and enqueue a non-blocking put work queue entry (WQE).
   *
   * All address arguments are symmetric addresses in the default or user symmetric heaps.
   * Virtual addresses, LKeys, and RKeys are resolved internally.
   *
   * @param[in] dest Destination symmetric address for data transmission.
   * @param[in] source Source symmetric address for data transmission.
   * @param[in] nelems Size in bytes of data transmission.
   * @param[in] wf_info Wavefront information.
   *
   * @tparam Options Options to use when posting these WQEs.
   */
  template <typename... PostOptions>
  __device__ void put_nbi(void *dest, const void *source, size_t nelems,
                          const ActiveWFInfo& wf_info,
                          PostOpt<PostOptions...> post_options = {}) {
    bool inlined_wqe = Provider::template can_inline<OpCode::RDMA_WRITE>(nelems);
    auto [d_raddr, d_rkey] = provider().get_raddr_info(dest);
    auto [s_laddr, s_lkey] = provider().get_laddr_info(source, inlined_wqe);
    provider().put_nbi(d_raddr, d_rkey, s_laddr, s_lkey, nelems, wf_info, post_options);
  }
  template <typename... PostOptions>
  __device__ void put_nbi_single(void *dest, const void *source, size_t nelems,
                                 PostOpt<PostOptions...> post_options = {}) {
    bool inlined_wqe = Provider::template can_inline<OpCode::RDMA_WRITE>(nelems);
    auto [d_raddr, d_rkey] = provider().get_raddr_info(dest);
    auto [s_laddr, s_lkey] = provider().get_laddr_info(source, inlined_wqe);
    provider().put_nbi_single(d_raddr, d_rkey, s_laddr, s_lkey, nelems, post_options);
  }

  /**
   * @brief Create and enqueue a non-blocking get work queue entry (WQE).
   *
   * All address arguments are symmetric addresses in the default or user symmetric heaps.
   * Virtual addresses, LKeys, and RKeys are resolved internally.
   *
   * @param[in] dest Destination symmetric address for data transmission.
   * @param[in] source Source symmetric address for data transmission.
   * @param[in] nelems Size in bytes of data transmission.
   * @param[in] wf_info Wavefront information.
   *
   * @tparam Options Options to use when posting these WQEs.
   */
  template <typename... PostOptions>
  __device__ void get_nbi(void *dest, const void *source, size_t nelems,
                          const ActiveWFInfo& wf_info,
                          PostOpt<PostOptions...> post_options = {}) {
    auto [d_laddr, d_lkey] = provider().get_laddr_info(dest);
    auto [s_raddr, s_rkey] = provider().get_raddr_info(source);
    provider().get_nbi(d_laddr, d_lkey, s_raddr, s_rkey, nelems, wf_info, post_options);
  }
  template <typename... PostOptions>
  __device__ void get_nbi_single(void *dest, const void *source, size_t nelems,
                                 PostOpt<PostOptions...> post_options = {}) {
    auto [d_laddr, d_lkey] = provider().get_laddr_info(dest);
    auto [s_raddr, s_rkey] = provider().get_raddr_info(source);
    provider().get_nbi_single(d_laddr, d_lkey, s_raddr, s_rkey, nelems, post_options);
  }
/**@}*/



/**
 * @name Atomic Memory Operations (AMO)
 *
 * @{
 */
public:
  /**
   * @brief Create and enqueue a blocking atomic fetch-and-add work queue entry (WQE).
   *
   * All address arguments are symmetric addresses in the default or user symmetric heaps.
   * Virtual addresses, LKeys, and RKeys are resolved internally.
   *
   * @param[in] dest Destination symmetric address for data transmission.
   * @param[in] value Data value for the atomic operation.
   * @param[in] wf_info Wavefront information.
   *
   * @tparam Options Options to use when posting these WQEs.
   *
   * @return An atomic value.
   */
  template <typename... PostOptions>
  __device__ uint64_t atomic_fetch_add(void *dest, uint64_t value,
                                       const ActiveWFInfo& wf_info,
                                       PostOpt<PostOptions...> post_options = {}) {
    auto [d_raddr, d_rkey] = provider().get_raddr_info(dest);
    return provider().atomic_fetch_add(d_raddr, d_rkey, value, wf_info, post_options);
  }
  template <typename... PostOptions>
  __device__ uint64_t atomic_fetch_add_single(void *dest, uint64_t value,
                                              PostOpt<PostOptions...> post_options = {}) {
    auto [d_raddr, d_rkey] = provider().get_raddr_info(dest);
    return provider().atomic_fetch_add_single(d_raddr, d_rkey, value, post_options);
  }

#if 0 // NON-BLOCKING AMOs NOT YET IMPLEMENTED
  /**
   * @brief Create and enqueue a non-blocking atomic fetch-and-add work queue entry (WQE).
   *
   * All address arguments are symmetric addresses in the default or user symmetric heaps.
   * Virtual addresses, LKeys, and RKeys are resolved internally.
   *
   * @param[in] fetch Symmetric address for fetched value.
   * @param[in] dest Destination symmetric address for data transmission.
   * @param[in] value Data value for the atomic operation.
   * @param[in] wf_info Wavefront information.
   *
   * @tparam Options Options to use when posting these WQEs.
   *
   * @return An atomic value.
   */
  template <typename... PostOptions>
  __device__ void atomic_fetch_add_nbi(uint64_t *fetch, void *dest, uint64_t value,
                                       const ActiveWFInfo& wf_info,
                                       PostOpt<PostOptions...> post_options = {}) {
    auto [f_laddr, f_lkey] = provider().get_laddr_info(fetch);
    auto [d_raddr, d_rkey] = provider().get_raddr_info(dest);
    provider().atomic_fetch_add_nbi(
        f_laddr, f_lkey, d_raddr, d_rkey, value, wf_info, post_options);
  }
  template <typename... PostOptions>
  __device__ void atomic_fetch_add_nbi_single(uint64_t *fetch, void *dest, uint64_t value,
                                              PostOpt<PostOptions...> post_options = {}) {
    auto [f_laddr, f_lkey] = provider().get_laddr_info(fetch);
    auto [d_raddr, d_rkey] = provider().get_raddr_info(dest);
    provider().atomic_fetch_add_nbi_single(
        f_laddr, f_lkey, d_raddr, d_rkey, value, post_options);
  }
#endif

  /**
   * @brief Create and enqueue a non-fetching atomic add work queue entry (WQE).
   *
   * All address arguments are symmetric addresses in the default or user symmetric heaps.
   * Virtual addresses, LKeys, and RKeys are resolved internally.
   *
   * @param[in] dest Destination symmetric address for data transmission.
   * @param[in] value Data value for the atomic operation.
   * @param[in] wf_info Wavefront information.
   *
   * @tparam Options Options to use when posting these WQEs.
   */
  template <typename... PostOptions>
  __device__ void atomic_add(void *dest, uint64_t value,
                             const ActiveWFInfo& wf_info,
                             PostOpt<PostOptions...> post_options = {}) {
    auto [d_raddr, d_rkey] = provider().get_raddr_info(dest);
    provider().atomic_add(d_raddr, d_rkey, value, wf_info, post_options);
  }
  template <typename... PostOptions>
  __device__ void atomic_add_single(void *dest, uint64_t value,
                                    PostOpt<PostOptions...> post_options = {}) {
    auto [d_raddr, d_rkey] = provider().get_raddr_info(dest);
    provider().atomic_add_single(d_raddr, d_rkey, value, post_options);
  }

  /**
   * @brief Create and enqueue a blocking atomic compare-and-swap work queue entry (WQE).
   *
   * All address arguments are symmetric addresses in the default or user symmetric heaps.
   * Virtual addresses, LKeys, and RKeys are resolved internally.
   *
   * @param[in] dest Destination symmetric address for data transmission.
   * @param[in] cond Used in atomic comparisons.
   * @param[in] value Data value for the atomic operation.
   * @param[in] wf_info Wavefront information.
   *
   * @tparam Options Options to use when posting these WQEs.
   *
   * @return An atomic value.
   */
  template <typename... PostOptions>
  __device__ uint64_t atomic_compare_swap(void *dest, uint64_t cond, uint64_t value,
                                          const ActiveWFInfo& wf_info,
                                          PostOpt<PostOptions...> post_options = {}) {
    auto [d_raddr, d_rkey] = provider().get_raddr_info(dest);
    return provider().atomic_compare_swap(d_raddr, d_rkey, cond, value, wf_info, post_options);
  }
  template <typename... PostOptions>
  __device__ uint64_t atomic_compare_swap_single(void *dest, uint64_t cond, uint64_t value,
                                                 PostOpt<PostOptions...> post_options = {}) {
    auto [d_raddr, d_rkey] = provider().get_raddr_info(dest);
    return provider().atomic_compare_swap_single(d_raddr, d_rkey, cond, value, post_options);
  }

#if 0 // NON-BLOCKING AMOs NOT YET IMPLEMENTED
  /**
   * @brief Create and enqueue a non-blocking atomic compare-and-swap work queue entry (WQE).
   *
   * All address arguments are symmetric addresses in the default or user symmetric heaps.
   * Virtual addresses, LKeys, and RKeys are resolved internally.
   *
   * @param[in] fetch Symmetric address for fetched value.
   * @param[in] dest Destination symmetric address for data transmission.
   * @param[in] cond Used in atomic comparisons.
   * @param[in] value Data value for the atomic operation.
   * @param[in] wf_info Wavefront information.
   *
   * @tparam Options Options to use when posting these WQEs.
   *
   * @return An atomic value.
   */
  template <typename... PostOptions>
  __device__ void atomic_compare_swap_nbi(uint64_t *fetch, void *dest,
                                          uint64_t cond, uint64_t value,
                                          const ActiveWFInfo& wf_info,
                                          PostOpt<PostOptions...> post_options = {}) {
    auto [f_laddr, f_lkey] = provider().get_laddr_info(fetch);
    auto [d_raddr, d_rkey] = provider().get_raddr_info(dest);
    provider().atomic_compare_swap_nbi(
        f_laddr, f_lkey, d_raddr, d_rkey, cond, value, wf_info, post_options);
  }
  template <typename... PostOptions>
  __device__ void atomic_compare_swap_nbi_single(uint64_t *fetch, void *dest,
                                                 uint64_t cond, uint64_t value,
                                                 PostOpt<PostOptions...> post_options = {}) {
    auto [f_laddr, f_lkey] = provider().get_laddr_info(fetch);
    auto [d_raddr, d_rkey] = provider().get_raddr_info(dest);
    provider().atomic_compare_swap_nbi_single(
        f_laddr, f_lkey, d_raddr, d_rkey, cond, value, post_options);
  }
#endif

  /**
   * @brief Create and enqueue a non-fetching atomic compare-and-swap work queue entry (WQE).
   *
   * All address arguments are symmetric addresses in the default or user symmetric heaps.
   * Virtual addresses, LKeys, and RKeys are resolved internally.
   *
   * @param[in] dest Destination symmetric address for data transmission.
   * @param[in] cond Used in atomic comparisons.
   * @param[in] value Data value for the atomic operation.
   * @param[in] wf_info Wavefront information.
   *
   * @tparam Options Options to use when posting these WQEs.
   *
   *
   * @return An atomic value.
   */
  template <typename... PostOptions>
  __device__ void atomic_compare_swap_nofetch(void *dest, uint64_t cond, uint64_t value,
                                              const ActiveWFInfo& wf_info,
                                              PostOpt<PostOptions...> post_options = {}) {
    auto [d_raddr, d_rkey] = provider().get_raddr_info(dest);
    /* QueuePairInterface doesn't provide atomic_compare_swap_nofetch */
    provider().template post_wqe_amo<OpCode::ATOMIC_CS, AMOFetchType::NonFetching>(
        d_raddr, d_rkey, value, cond, wf_info, post_options);
  }
  template <typename... PostOptions>
  __device__ void atomic_compare_swap_nofetch_single(void *dest, uint64_t cond, uint64_t value,
                                                     PostOpt<PostOptions...> post_options = {}) {
    auto [d_raddr, d_rkey] = provider().get_raddr_info(dest);
    /* QueuePairInterface doesn't provide atomic_compare_swap_nofetch_single */
    provider().template post_wqe_amo_single<OpCode::ATOMIC_CS, AMOFetchType::NonFetching>(
        d_raddr, d_rkey, value, cond, post_options);
  }
/**@}*/
};

}  // namespace rocshmem

#endif  // LIBRARY_SRC_GDA_QUEUE_PAIR_SHMEM_HPP_
