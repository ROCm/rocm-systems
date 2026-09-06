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

#ifndef LIBRARY_SRC_GDA_QUEUE_PAIR_INTERFACE_HPP_
#define LIBRARY_SRC_GDA_QUEUE_PAIR_INTERFACE_HPP_

/**
 * @file queue_pair_interface.hpp
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
#include "queue_pair_option.hpp"

namespace rocshmem {

/*
 * @brief CRTP mixin class supplying a common mid-level interface to Queue Pair implementations.
 */
template <typename Provider>
class QueuePairInterface {
/**
 * @name Provider-Defined Members
 *
 * Members that are defined based on specializations of QueuePairTraits<Provider>.
 * Each Provider must supply their own definitions.
 *
 * @{
 */
public:
  /**
   * @brief Type alias for QueuePairTraits<Provider>.
   */
  using Traits = QueuePairTraits<Provider>;

  /* Check that Traits provides a valid inlining threshold */
  static_assert(Traits::InlineThreshold <= Traits::InlineMax,
                "Inlining threshold must be less than the inlining maximum.");

  /**
   * @brief Enumeration of the opcodes for Write, Read, Fetch-Add, and Compare-and-Swap.
   *
   * A definition must be provided by the QueuePairTraits<Provider> specialization
   * of each Provider subclass.
   */
  using OpCode = typename Traits::OpCode;
/**@}*/



/**
 * @name Member Types
 *
 * @{
 */
public:
  /**
   * @brief Helper alias for defining post_wqe_amo and post_wqe_amo_single in subclasses
   */
  template <AMOFetchType Fetch>
  using amo_ret_t = std::conditional_t<Fetch == AMOFetchType::Blocking, uint64_t, void>;
/**@}*/



/**
 * @name Internal
 *
 * @{
 */
protected:
  /*
   * @brief Provider is a friend of QueuePairInterface<Provider>.
   */
  friend Provider;

  __host__  QueuePairInterface() = default;
  __device__ QueuePairInterface() = delete;

private:
  __device__ Provider& provider() {
    static_assert(std::is_trivially_destructible_v<Provider>,
                  "QueuePair providers should have a trivial destructor: "
                  "memory is managed externally by QueuePairHost.");
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
   * All address arguments are virtual addresses at origin or target,
   * as appropriate for the direction of data transfer.
   *
   * @param[in] dest Destination address for data transmission. Remote virtual address at target.
   * @param[in] dest_rkey RKey of destination address.
   * @param[in] source Source address for data transmission. Local virtual address at origin.
   * @param[in] source_lkey LKey of source address.
   * @param[in] nelems Size in bytes of data transmission.
   * @param[in] wf_info Wavefront information.
   *
   * @tparam Options Options to use when posting these WQEs.
   */
  template <typename... PostOptions>
  __device__ void put_nbi(uintptr_t dest, uint32_t dest_rkey,
                          uintptr_t source, uint32_t source_lkey, size_t nelems,
                          const ActiveWFInfo& wf_info,
                          PostOpt<PostOptions...> post_options = {}) {
    provider().template post_wqe_rma<OpCode::RDMA_WRITE>(
        source, source_lkey, dest, dest_rkey, nelems, wf_info, post_options);
  }
  template <typename... PostOptions>
  __device__ void put_nbi_single(uintptr_t dest, uint32_t dest_rkey,
                                 uintptr_t source, uint32_t source_lkey, size_t nelems,
                                 PostOpt<PostOptions...> post_options = {}) {
    provider().template post_wqe_rma_single<OpCode::RDMA_WRITE>(
        source, source_lkey, dest, dest_rkey, nelems, post_options);
  }

  /**
   * @brief Create and enqueue a non-blocking get work queue entry (WQE).
   *
   * All address arguments are virtual addresses at origin or target,
   * as appropriate for the direction of data transfer.
   *
   * @param[in] dest Destination address for data transmission. Local virtual address at origin.
   * @param[in] dest_lkey LKey of destination address.
   * @param[in] source Source address for data transmission. Remote virtual address at target.
   * @param[in] source_rkey RKey of source address.
   * @param[in] nelems Size in bytes of data transmission.
   * @param[in] wf_info Wavefront information.
   *
   * @tparam Options Options to use when posting these WQEs.
   */
  template <typename... PostOptions>
  __device__ void get_nbi(uintptr_t dest, uint32_t dest_lkey,
                          uintptr_t source, uint32_t source_rkey, size_t nelems,
                          const ActiveWFInfo& wf_info,
                          PostOpt<PostOptions...> post_options = {}) {
    provider().template post_wqe_rma<OpCode::RDMA_READ>(
        dest, dest_lkey, source, source_rkey, nelems, wf_info, post_options);
  }
  template <typename... PostOptions>
  __device__ void get_nbi_single(uintptr_t dest, uint32_t dest_lkey,
                                 uintptr_t source, uint32_t source_rkey, size_t nelems,
                                 PostOpt<PostOptions...> post_options = {}) {
    provider().template post_wqe_rma_single<OpCode::RDMA_READ>(
        dest, dest_lkey, source, source_rkey, nelems, post_options);
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
   * All address arguments are virtual addresses at origin or target,
   * as appropriate for the direction of data transfer.
   *
   * @param[in] dest Destination address for data transmission. Remote virtual address at target.
   * @param[in] dest_rkey RKey of destination address.
   * @param[in] value Data value for the atomic operation.
   * @param[in] wf_info Wavefront information.
   *
   * @tparam Options Options to use when posting these WQEs.
   *
   * @return An atomic value.
   */
  template <typename... PostOptions>
  __device__ uint64_t atomic_fetch_add(uintptr_t dest, uint32_t dest_rkey, uint64_t value,
                                       const ActiveWFInfo& wf_info,
                                       PostOpt<PostOptions...> post_options = {}) {
    return provider().template post_wqe_amo<OpCode::ATOMIC_FA, AMOFetchType::Blocking>(
        dest, dest_rkey, value, 0, wf_info, post_options);
  }
  template <typename... PostOptions>
  __device__ uint64_t atomic_fetch_add_single(uintptr_t dest, uint32_t dest_rkey, uint64_t value,
                                              PostOpt<PostOptions...> post_options = {}) {
    return provider().template post_wqe_amo_single<OpCode::ATOMIC_FA, AMOFetchType::Blocking>(
        dest, dest_rkey, value, 0, post_options);
  }

#if 0 // NON-BLOCKING AMOs NOT YET IMPLEMENTED
  /**
   * @brief Create and enqueue a non-blocking atomic fetch-and-add work queue entry (WQE).
   *
   * All address arguments are virtual addresses at origin or target,
   * as appropriate for the direction of data transfer.
   *
   * @param[in] fetch Address for fetched value. Local virtual address at origin.
   * @param[in] fetch_lkey LKey of address for fetched value.
   * @param[in] dest Destination address for data transmission. Remote virtual address at target.
   * @param[in] dest_rkey RKey of destination address.
   * @param[in] value Data value for the atomic operation.
   * @param[in] wf_info Wavefront information.
   *
   * @tparam Options Options to use when posting these WQEs.
   *
   * @return An atomic value.
   */
  template <typename... PostOptions>
  __device__ void atomic_fetch_add_nbi(uintptr_t fetch, uint32_t fetch_lkey,
                                       uintptr_t dest, uint32_t dest_rkey, uint64_t value,
                                       const ActiveWFInfo& wf_info,
                                       PostOpt<PostOptions...> post_options = {}) {
    provider().template post_wqe_amo<OpCode::ATOMIC_FA, AMOFetchType::NonBlocking>(
        fetch, fetch_lkey, dest, dest_rkey, value, 0, wf_info, post_options);
  }
  template <typename... PostOptions>
  __device__ void atomic_fetch_add_nbi_single(uintptr_t fetch, uint32_t fetch_lkey,
                                              uintptr_t dest, uint32_t dest_rkey, uint64_t value,
                                              PostOpt<PostOptions...> post_options = {}) {
    provider().template post_wqe_amo_single<OpCode::ATOMIC_FA, AMOFetchType::NonBlocking>(
        fetch, fetch_lkey, dest, dest_rkey, value, 0, post_options);
  }
#endif

  /**
   * @brief Create and enqueue a non-fetching atomic add work queue entry (WQE).
   *
   * All address arguments are virtual addresses at origin or target,
   * as appropriate for the direction of data transfer.
   *
   * @param[in] dest Destination address for data transmission. Remote virtual address at target.
   * @param[in] dest_rkey RKey of destination address.
   * @param[in] value Data value for the atomic operation.
   * @param[in] wf_info Wavefront information.
   *
   * @tparam Options Options to use when posting these WQEs.
   */
  template <typename... PostOptions>
  __device__ void atomic_add(uintptr_t dest, uint32_t dest_rkey, uint64_t value,
                             const ActiveWFInfo& wf_info,
                             PostOpt<PostOptions...> post_options = {}) {
    provider().template post_wqe_amo<OpCode::ATOMIC_FA, AMOFetchType::NonFetching>(
        dest, dest_rkey, value, 0, wf_info, post_options);
  }
  template <typename... PostOptions>
  __device__ void atomic_add_single(uintptr_t dest, uint32_t dest_rkey, uint64_t value,
                                    PostOpt<PostOptions...> post_options = {}) {
    provider().template post_wqe_amo_single<OpCode::ATOMIC_FA, AMOFetchType::NonFetching>(
        dest, dest_rkey, value, 0, post_options);
  }

  /**
   * @brief Create and enqueue a blocking atomic compare-and-swap work queue entry (WQE).
   *
   * All address arguments are virtual addresses at origin or target,
   * as appropriate for the direction of data transfer.
   *
   * @param[in] dest Destination address for data transmission. Remote virtual address at target.
   * @param[in] dest_rkey RKey of destination address.
   * @param[in] cond Used in atomic comparisons.
   * @param[in] value Data value for the atomic operation.
   * @param[in] wf_info Wavefront information.
   *
   * @tparam Options Options to use when posting these WQEs.
   *
   * @return An atomic value.
   */
  template <typename... PostOptions>
  __device__ uint64_t atomic_compare_swap(uintptr_t dest, uint32_t dest_rkey,
                                          uint64_t cond, uint64_t value,
                                          const ActiveWFInfo& wf_info,
                                          PostOpt<PostOptions...> post_options = {}) {
    return provider().template post_wqe_amo<OpCode::ATOMIC_CS, AMOFetchType::Blocking>(
        dest, dest_rkey, value, cond, wf_info, post_options);
  }
  template <typename... PostOptions>
  __device__ uint64_t atomic_compare_swap_single(uintptr_t dest, uint32_t dest_rkey,
                                                 uint64_t cond, uint64_t value,
                                                 PostOpt<PostOptions...> post_options = {}) {
    return provider().template post_wqe_amo_single<OpCode::ATOMIC_CS, AMOFetchType::Blocking>(
        dest, dest_rkey, value, cond, post_options);
  }

#if 0 // NON-BLOCKING AMOs NOT YET IMPLEMENTED
  /**
   * @brief Create and enqueue a non-blocking atomic compare-and-swap work queue entry (WQE).
   *
   * All address arguments are virtual addresses at origin or target,
   * as appropriate for the direction of data transfer.
   *
   * @param[in] fetch Address for fetched value. Local virtual address at origin.
   * @param[in] fetch_lkey LKey of address for fetched value.
   * @param[in] dest Destination address for data transmission. Remote virtual address at target.
   * @param[in] dest_rkey RKey of destination address.
   * @param[in] cond Used in atomic comparisons.
   * @param[in] value Data value for the atomic operation.
   * @param[in] wf_info Wavefront information.
   *
   * @tparam Options Options to use when posting these WQEs.
   *
   * @return An atomic value.
   */
  template <typename... PostOptions>
  __device__ void atomic_compare_swap_nbi(uintptr_t fetch, uint32_t fetch_lkey,
                                          uintptr_t dest, uint32_t dest_rkey,
                                          uint64_t cond, uint64_t value,
                                          const ActiveWFInfo& wf_info,
                                          PostOpt<PostOptions...> post_options = {}) {
    provider().template post_wqe_amo<OpCode::ATOMIC_CS, AMOFetchType::NonBlocking>(
        fetch, fetch_lkey, dest, dest_rkey, value, cond, wf_info, post_options);
  }
  template <typename... PostOptions>
  __device__ void atomic_compare_swap_nbi_single(uintptr_t fetch, uint32_t fetch_lkey,
                                                 uintptr_t dest, uint32_t dest_rkey,
                                                 uint64_t cond, uint64_t value,
                                                 PostOpt<PostOptions...> post_options = {}) {
    provider().template post_wqe_amo_single<OpCode::ATOMIC_CS, AMOFetchType::NonBlocking>(
        fetch, fetch_lkey, dest, dest_rkey, value, cond, post_options);
  }
#endif
/**@}*/



/**
 * @name Completion and Ordering.
 *
 * @{
 */
public:
  /**
   * @brief Empty all completions from the completion queue.
   *
   * @param[in] wf_info Wavefront information.
   */
  __device__ void quiet(const ActiveWFInfo& wf_info) {
    if (wf_info.is_pe_group_first) {
      provider().quiet_single();
    }
  }
/**@}*/
};

}  // namespace rocshmem

#endif  // LIBRARY_SRC_GDA_QUEUE_PAIR_INTERFACE_HPP_
