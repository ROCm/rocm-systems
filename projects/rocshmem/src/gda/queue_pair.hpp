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

#ifndef LIBRARY_SRC_GDA_QUEUE_PAIR_HPP_
#define LIBRARY_SRC_GDA_QUEUE_PAIR_HPP_

/**
 * @file queue_pair.hpp
 *
 * @section DESCRIPTION
 * An IB QueuePair (SQ and CQ) that the device can use to perform network
 * operations. Most important rocSHMEM operations are performed by this
 * class.
 */

#include <cassert>
#include <cstddef>
#include <cstdint>

#include <algorithm>
#include <functional>
#include <iterator>
#include <limits>
#include <tuple>
#include <type_traits>
#include <unordered_map>
#include <vector>

#include <hip/hip_runtime.h>

#include "rocshmem/rocshmem_config.h"  // NOLINT(build/include_subdir)
#include "rocshmem/rocshmem.hpp"
#include "endian.hpp"
#include "constants.hpp"
#include "util.hpp"

#include "ibv_wrapper.hpp"

#include "containers/free_list_impl.hpp"
#include "memory/hip_allocator.hpp"

namespace rocshmem {

struct BufferInfo {
  uintptr_t addr;
  size_t    length;
  uint32_t  lkey;
};

struct BufferHostInfo {
  struct ibv_mr* mr;
  BufferInfo*    info_ptr;

  BufferHostInfo(struct ibv_mr* mr, BufferInfo* info_ptr) : mr{mr}, info_ptr{info_ptr} { }
};

template <typename Iter>
class iterator_freelist {
private:
  std::vector<Iter> container{};

  /* Reserve space in container for std::distance(first, last) elements */
  void reserve(Iter first, Iter last) {
    container.reserve(std::distance(first, last));
  }

  void init(Iter first, Iter last) {
    /* Push back iterators in range [first, last) to container in reverse order
     * Sorted so that container.back() always has the smallest iterator */
    auto N = std::distance(first, last);
    while (first != last--) {
      container.push_back(last);
    }
  }

public:
  using value_type      = Iter;
  using difference_type = typename std::iterator_traits<Iter>::difference_type;
  using size_type       = typename decltype(container)::size_type;

  iterator_freelist() = default;

  /* Initialize freelist with iterators in range [first, last) */
  iterator_freelist(Iter first, Iter last) {
    reserve(first, last);
    init(first, last);
  }

  /* Initialize freelist with iterators in range [first, std::next(first, count)) */
  iterator_freelist(Iter first, difference_type count)
    : iterator_freelist{first, std::next(first, count)} { }

  bool empty() const {
    return container.empty();
  }

  size_type size() const {
    return container.size();
  }

  Iter top() {
    return container.back();
  }

  void push(Iter value) {
    std::greater<Iter> compare{};
    if (container.empty() || compare(top(), value)) {
      /* value < top(), so push_back maintains sorted property */
      container.push_back(value);
    } else {
      /* Use binary search to find insert position for value */
      auto pos = std::upper_bound(container.cbegin(), container.cend(), value, compare);
      container.insert(pos, value);
    }
  }

  void pop() {
    container.pop_back();
  }

  void reset(Iter first, Iter last) {
    container.clear();
    reserve(first, last);
    init(first, last);
  }

  void reset(Iter first, difference_type count) {
    reset(first, std::next(first, count));
  }
};

/**
 * @brief Scope at which WQEs are issued and completed. This is used to
 * determine how to synchronize threads and when to poll the CQ for
 * completions.
 * thread: Each thread issues WQEs independently
 * wave: Thread 0 in each wave issues WQE
 * wg: Thread 0 of WAVE 0 issues WQE
 */

enum class ThreadScope: int {
  thread,
  wave,
  wg
};

class ActiveWFInfo {
 public:
  uint64_t    activemask{0};                  // Mask of active threads in the wavefront
  uint64_t    pe_group_mask{0};               // Mask of active threads with the same PE
  int         pe{-1};                         // PE for the threads in pe_group_mask
  int         num_pe_group_lanes{0};          // Number of active lanes in pe_group_mask
  int         pe_group_logical_lane_id{0};    // Logical lane id of this thread in pe_group_mask
  int         pe_group_first_phys_lane_id{0}; // Physical lane id of first thread in pe_group_mask
  int         pe_group_last_phys_lane_id{0};  // Physical lane id of last thread in pe_group_mask
  ThreadScope scope{ThreadScope::thread};     // Threading scope
  bool        is_pe_group_first{false};       // True if this is the first thread in pe_group_mask
  bool        is_pe_group_last{false};        // True if this is the last thread in pe_group_mask

  __device__ explicit ActiveWFInfo(int pe, ThreadScope scope = ThreadScope::thread)
      : pe(pe), scope(scope) {
    // Get active lane mask
    activemask = get_active_lane_mask();

    // Get mask of active lanes with the same PE
    switch (scope) {
      case ThreadScope::thread: {
        pe_group_mask       = __match_any_sync(activemask, pe);
        num_pe_group_lanes  = get_active_lane_count(pe_group_mask);
        pe_group_logical_lane_id = get_active_lane_num(pe_group_mask);
        pe_group_first_phys_lane_id = get_first_active_lane_id(pe_group_mask);
        pe_group_last_phys_lane_id  = get_last_active_lane_id(pe_group_mask);
        break;
      }
      // Only thread 0 issues the WQE, so the group is just that thread
      case ThreadScope::wave:
      case ThreadScope::wg: {
        pe_group_mask       = 1;
        num_pe_group_lanes  = 1;
        pe_group_logical_lane_id = get_active_lane_num(activemask);
        pe_group_first_phys_lane_id = 0;
        pe_group_last_phys_lane_id  = 0;
      }
    }
    is_pe_group_first = (pe_group_logical_lane_id == 0);
    is_pe_group_last  = (pe_group_logical_lane_id == num_pe_group_lanes - 1);
  }

  // used in CAS based atomic operations at thread scope
  __device__ void update(int _pe, ThreadScope _scope = ThreadScope::thread) {
    // Get active lane mask
    activemask          = get_active_lane_mask();
    pe_group_mask       = __match_any_sync(activemask, pe);
    num_pe_group_lanes  = get_active_lane_count(pe_group_mask);
    pe_group_logical_lane_id = get_active_lane_num(pe_group_mask);
    pe_group_first_phys_lane_id = get_first_active_lane_id(pe_group_mask);
    pe_group_last_phys_lane_id  = get_last_active_lane_id(pe_group_mask);
    is_pe_group_first   = (pe_group_logical_lane_id == 0);
    is_pe_group_last    = (pe_group_logical_lane_id == num_pe_group_lanes - 1);
    scope               = _scope;
    pe                  = _pe;
  }

  __device__ void printInfo() {
    printf("PE: %d, Scope: %d, activemask: %llx, "
           "pe_group_mask: %llx, num_pe_group_lanes: %d, "
           "thread_id: %u, pe_group_logical_lane_id: %d, "
           "is_pe_group_first: %d, pe_group_first_phys_lane_id: %d, "
           "is_pe_group_last: %d, pe_group_last_phys_lane_id: %d\n",
           pe, static_cast<int>(scope), static_cast<unsigned long long>(activemask),
           static_cast<unsigned long long>(pe_group_mask), num_pe_group_lanes,
           threadIdx.x, pe_group_logical_lane_id,
           static_cast<int>(is_pe_group_first), pe_group_first_phys_lane_id,
           static_cast<int>(is_pe_group_last), pe_group_last_phys_lane_id);
  }
};

/*
 * @struct QueuePairTraits<Provider>
 * @brief Defines Provider-specific types and constants.
 *
 * Each Provider subclass of QueuePairBase<Provider> should also define a specialization
 * for QueuePairTraits<Provider> that defines the documented members:
 *   - QueuePairTraits<Provider>::OpCode
 *   - QueuePairTraits<Provider>::Endianness
 *   - QueuePairTraits<Provider>::InlineThreshold
 *
 * Sample specialization code for a QueuePairProvider subclass
 * of QueuePairBase<QueuePairProvider>:
 * @code
 * class QueuePairProvider;
 * template <> struct QueuePairTraits<QueuePairProvider> {
 *   enum class OpCode : uint8_t {
 *     RDMA_WRITE = ...,
 *     RDMA_READ  = ...,
 *     ATOMIC_CS  = ...,
 *     ATOMIC_FA  = ...,
 *   };
 *
 *   static constexpr endian::Order Endianness = endian::Order::...;
 *
 *   static constexpr size_t InlineThreshold = ...;
 * };
 * @endcode
 *
 * @tparam Provider Name of Provider class.
 */

/*
 * @enum QueuePairTraits<Provider>::OpCode
 * @brief Enumeration of the Provider-specific opcodes.
 *
 * @var QueuePairTraits<Provider>::OpCode::RDMA_WRITE
 * Provider-specific RDMA Write opcode.
 *
 * @var QueuePairTraits<Provider>::OpCode::RDMA_READ
 * Provider-specific RDMA Read opcode.
 *
 * @var QueuePairTraits<Provider>::OpCode::ATOMIC_CS
 * Provider-specific Compare-and-Swap opcode.
 *
 * @var QueuePairTraits<Provider>::OpCode::ATOMIC_FA
 * Provider-specific Fetch-Add opcode.
 */

/*
 * @var endian::Order QueuePairTraits<Provider>::Endianness
 * @brief Endianness order of data stored by the hardware for Provider.
 *
 * @qualifier static
 * @qualifier constexpr
 */

/*
 * @var size_t QueuePairTraits<Provider>::InlineThreshold
 * @brief Maximum number of bytes that can be sent inline in a WQE.
 *
 * @qualifer static
 * @qualifier constexpr
 */

template <typename Provider>
struct QueuePairTraits;

namespace QueuePairOption {
  /*
   * @brief Helper alias for option tag types before C++26 std::constant_wrapper.
   *
   * Options are defined as types derived from constant_t<V>
   * so that type-based tag dispatching can be used.
   */
  template <auto V>
  using constant_t = std::integral_constant<decltype(V), V>;

  enum class UpdateThread {
    All,
    Last,
    None,
  };

  /*
   * @brief Option: whether the send queue doorbell will be rung.
   */
  template <bool ring_db>
  struct ring_db_tag : constant_t<ring_db> { };

  /*
   * @brief Option: whether thread safety will be enforced.
   */
  template <bool thread_safe>
  struct thread_safe_tag : constant_t<thread_safe> { };

  /*
   * @brief Option: whether the number of available send queue entries will be checked.
   */
  template <bool check_sq>
  struct check_sq_tag : constant_t<check_sq> { };

  /*
   * @brief Option: which threads' WQEs will generate CQEs.
   */
  template <UpdateThread update_cq>
  struct update_cq_tag : constant_t<update_cq> { };

  /* forward declaration */
  template <typename... Options> struct PostOpt;

  /* deduction guide, required before C++20 */
  template <typename... Options> __host__ __device__ PostOpt(Options...) -> PostOpt<Options...>;

  /* Base case with all options defined */
  template <bool ring_db, bool thread_safe, bool check_sq, UpdateThread update_cq>
  struct PostOpt<ring_db_tag<ring_db>,
                 thread_safe_tag<thread_safe>,
                 check_sq_tag<check_sq>,
                 update_cq_tag<update_cq>> {
    /* empty constructor for type deduction from tags */
    template <typename... Options> __host__ __device__ constexpr PostOpt(Options...) { }

    /* static constexpr data members to simplify option access */
    static constexpr auto RingDB     = ring_db;
    static constexpr auto ThreadSafe = thread_safe;
    static constexpr auto CheckSQ    = check_sq;
    static constexpr auto UpdateCQ   = update_cq;

    static __device__ constexpr inline bool signal_completion(const ActiveWFInfo& wf_info) {
      if constexpr (UpdateCQ == UpdateThread::All) {
        // all WQEs update the CQ
        return true;
      } else if constexpr (UpdateCQ == UpdateThread::Last) {
        // only the last WQE in each group updates the CQ
        return wf_info.is_pe_group_last;
      } else {
        // no WQEs update the CQ
        return false;
      }
    }

    static __device__ constexpr inline bool signal_completion_single() {
      if constexpr (UpdateCQ == UpdateThread::All) {
        // all WQEs update the CQ
        return true;
      } else if constexpr (UpdateCQ == UpdateThread::Last) {
        // singleton groups, so all threads are "last": all WQEs update the CQ
        return true;
      } else {
        // no WQEs update the CQ
        return false;
      }
    }
  };

  /* Extraneous parameters,
   * else matches PostOpt<ring_db_tag, thread_safe_tag, check_sq_tag, update_cq_tag> */
  template <bool ring_db, bool thread_safe, bool check_sq, UpdateThread update_cq, typename... Options>
  struct PostOpt<ring_db_tag<ring_db>,
                 thread_safe_tag<thread_safe>,
                 check_sq_tag<check_sq>,
                 update_cq_tag<update_cq>,
                 Options...> {
    static_assert(sizeof...(Options) == 0, "Too many or invalid options");
  };

  /* Missing update_cq_tag,
   * else matches PostOpt<ring_db_tag, thread_safe_tag, check_sq_tag, update_cq_tag, Options...> */
  template <bool ring_db, bool thread_safe, bool check_sq, typename... Options>
  struct PostOpt<ring_db_tag<ring_db>,
                 thread_safe_tag<thread_safe>,
                 check_sq_tag<check_sq>,
                 Options...>
       : PostOpt<ring_db_tag<ring_db>,
                 thread_safe_tag<thread_safe>,
                 check_sq_tag<check_sq>,
                 update_cq_tag</* default: all WQEs update the CQ */ UpdateThread::All>,
                 Options...> {
    /* inherit constructor */
    using PostOpt<ring_db_tag<ring_db>,
                  thread_safe_tag<thread_safe>,
                  check_sq_tag<check_sq>,
                  update_cq_tag<UpdateThread::All>,
                  Options...
                 >::PostOpt;
  };

  /* Missing check_sq_tag,
   * else matches PostOpt<ring_db_tag, thread_safe_tag, check_sq_tag, Options...> */
  template <bool ring_db, bool thread_safe, typename... Options>
  struct PostOpt<ring_db_tag<ring_db>,
                 thread_safe_tag<thread_safe>,
                 Options...>
       : PostOpt<ring_db_tag<ring_db>,
                 thread_safe_tag<thread_safe>,
                 check_sq_tag</* default: DO check the SQ */ true>,
                 Options...> {
    /* inherit constructor */
    using PostOpt<ring_db_tag<ring_db>,
                  thread_safe_tag<thread_safe>,
                  check_sq_tag<true>,
                  Options...
                 >::PostOpt;
  };

  /* Missing thread_safe_tag,
   * else matches PostOpt<ring_db_tag, thread_safe_tag, Options...> */
  template <bool ring_db, typename... Options>
  struct PostOpt<ring_db_tag<ring_db>,
                 Options...>
       : PostOpt<ring_db_tag<ring_db>,
                 thread_safe_tag</* default: DO use thread safety */ true>,
                 Options...> {
    /* inherit constructor */
    using PostOpt<ring_db_tag<ring_db>,
                  thread_safe_tag<true>,
                  Options...
                 >::PostOpt;
  };

  /* Missing ring_db_tag,
   * else matches PostOpt<ring_db_tag, Options...> */
  template <typename... Options>
  struct PostOpt
       : PostOpt<ring_db_tag</* default: DO ring the doorbell */ true>,
                 Options...> {
    /* inherit constructor */
    using PostOpt<ring_db_tag<true>,
                  Options...
                 >::PostOpt;
  };
}

/*
 * @brief Type alias helper for WQE posting options.
 */
using QueuePairOption::PostOpt;

/* bring QueuePairOption::UpdateThread into scope */
using QueuePairOption::UpdateThread;

/* constexpr variable templates to simplify usage */
template <auto V> constexpr inline auto RingDB     = QueuePairOption::ring_db_tag<V>{};
template <auto V> constexpr inline auto ThreadSafe = QueuePairOption::thread_safe_tag<V>{};
template <auto V> constexpr inline auto CheckSQ    = QueuePairOption::check_sq_tag<V>{};
template <auto V> constexpr inline auto UpdateCQ   = QueuePairOption::update_cq_tag<V>{};

/*
 * @brief Atomic Memory Operation fetching behavior.
 */
enum class AMOFetchType {
  Blocking,
  NonBlocking,
  NonFetching,
};



/*
 * @brief CRTP mixin class supplying a common SHMEM-like interface to Queue Pair implementations.
 */
template <typename Provider>
class QueuePairSHMEM {
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
 * @name Remote Memory Access (RMA)
 *
 * @{
 */
public:
  /**
   * @brief Create and enqueue a non-blocking put work queue entry (WQE).
   *
   * @param[in] dest Destination address for data transmission.
   * @param[in] source Source address for data transmission.
   * @param[in] nelems Size in bytes of data transmission.
   * @param[in] wf_info Wavefront information.
   *
   * @tparam Options Options to use when posting these WQEs.
   */
  template <typename... PostOptions>
  __device__ void put_nbi(void *dest, const void *source, size_t nelems,
                          const ActiveWFInfo& wf_info, PostOpt<PostOptions...> = {});
  template <typename... PostOptions>
  __device__ void put_nbi_single(void *dest, const void *source, size_t nelems,
                                 PostOpt<PostOptions...> = {});

  /**
   * @brief Create and enqueue a non-blocking get work queue entry (WQE).
   *
   * @param[in] dest Destination address for data transmission.
   * @param[in] source Source address for data transmission.
   * @param[in] nelems Size in bytes of data transmission.
   * @param[in] wf_info Wavefront information.
   *
   * @tparam Options Options to use when posting these WQEs.
   */
  template <typename... PostOptions>
  __device__ void get_nbi(void *dest, const void *source, size_t nelems,
                          const ActiveWFInfo& wf_info, PostOpt<PostOptions...> = {});
  template <typename... PostOptions>
  __device__ void get_nbi_single(void *dest, const void *source, size_t nelems,
                                 PostOpt<PostOptions...> = {});
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
   * @param[in] dest Destination address for data transmission.
   * @param[in] value Data value for the atomic operation.
   * @param[in] wf_info Wavefront information.
   *
   * @tparam Options Options to use when posting these WQEs.
   *
   * @return An atomic value.
   */
  template <typename... PostOptions>
  __device__ uint64_t atomic_fetch_add(void *dest, uint64_t value,
                                       const ActiveWFInfo& wf_info, PostOpt<PostOptions...> = {});
  template <typename... PostOptions>
  __device__ uint64_t atomic_fetch_add_single(void *dest, uint64_t value,
                                              PostOpt<PostOptions...> = {});

#if 0
  /**
   * @brief Create and enqueue a non-blocking atomic fetch-and-add work queue entry (WQE).
   *
   * @param[in] fetch Address for fetched value.
   * @param[in] dest Destination address for data transmission.
   * @param[in] value Data value for the atomic operation.
   * @param[in] wf_info Wavefront information.
   *
   * @tparam Options Options to use when posting these WQEs.
   *
   * @return An atomic value.
   */
  template <typename... PostOptions>
  __device__ void atomic_fetch_add_nbi(uint64_t *fetch, void *dest, uint64_t value,
                                       const ActiveWFInfo& wf_info, PostOpt<PostOptions...> = {});
  template <typename... PostOptions>
  __device__ void atomic_fetch_add_nbi_single(uint64_t *fetch, void *dest, uint64_t value,
                                              PostOpt<PostOptions...> = {});
#endif

  /**
   * @brief Create and enqueue a non-fetching atomic add work queue entry (WQE).
   *
   * @param[in] dest Destination address for data transmission.
   * @param[in] value Data value for the atomic operation.
   * @param[in] wf_info Wavefront information.
   *
   * @tparam Options Options to use when posting these WQEs.
   */
  template <typename... PostOptions>
  __device__ void atomic_add(void *dest, uint64_t value,
                             const ActiveWFInfo& wf_info, PostOpt<PostOptions...> = {});
  template <typename... PostOptions>
  __device__ void atomic_add_single(void *dest, uint64_t value,
                                    PostOpt<PostOptions...> = {});

  /**
   * @brief Create and enqueue a blocking atomic compare-and-swap work queue entry (WQE).
   *
   * @param[in] dest Destination address for data transmission.
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
                                          const ActiveWFInfo& wf_info, PostOpt<PostOptions...> = {});
  template <typename... PostOptions>
  __device__ uint64_t atomic_compare_swap_single(void *dest, uint64_t cond, uint64_t value,
                                                 PostOpt<PostOptions...> = {});

#if 0
  /**
   * @brief Create and enqueue a non-blocking atomic compare-and-swap work queue entry (WQE).
   *
   * @param[in] fetch Address for fetched value.
   * @param[in] dest Destination address for data transmission.
   * @param[in] cond Used in atomic comparisons.
   * @param[in] value Data value for the atomic operation.
   * @param[in] wf_info Wavefront information.
   *
   * @tparam Options Options to use when posting these WQEs.
   *
   * @return An atomic value.
   */
  template <typename... PostOptions>
  __device__ void atomic_compare_swap_nbi(uint64_t *fetch, void *dest, uint64_t cond, uint64_t value,
                                          const ActiveWFInfo& wf_info, PostOpt<PostOptions...> = {});
  template <typename... PostOptions>
  __device__ void atomic_compare_swap_nbi_single(uint64_t *fetch, void *dest, uint64_t cond, uint64_t value,
                                                 PostOpt<PostOptions...> = {});
#endif

  /**
   * @brief Create and enqueue a non-fetching atomic compare-and-swap work queue entry (WQE).
   *
   * @param[in] dest Destination address for data transmission.
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
                                              const ActiveWFInfo& wf_info, PostOpt<PostOptions...> = {});
  template <typename... PostOptions>
  __device__ void atomic_compare_swap_nofetch_single(void *dest, uint64_t cond, uint64_t value,
                                                     PostOpt<PostOptions...> = {});
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
  __device__ void quiet(const ActiveWFInfo& wf_info);
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
};



template <typename Provider>
template <typename... PostOptions>
__device__ void QueuePairSHMEM<Provider>::put_nbi(
    void *dest, const void *source, size_t nelems,
    const ActiveWFInfo& wf_info, PostOpt<PostOptions...> post_options) {
  static constexpr OpCode Op = OpCode::RDMA_WRITE;
  uintptr_t laddr = reinterpret_cast<uintptr_t>(source);
  uintptr_t raddr = reinterpret_cast<uintptr_t>(dest);
  /* only need lkey when RDMA_WRITE can't be inlined */
  uint32_t lkey = !Provider::template can_inline<Op>(nelems) ? provider().get_lkey(laddr) : 0;
  uint32_t rkey = provider().get_rkey(raddr);
  provider().template post_wqe_rma<Op>(laddr, lkey, raddr, rkey, nelems, wf_info, post_options);
}

template <typename Provider>
template <typename... PostOptions>
__device__ void QueuePairSHMEM<Provider>::put_nbi_single(
    void *dest, const void *source, size_t nelems,
    PostOpt<PostOptions...> post_options) {
  static constexpr OpCode Op = OpCode::RDMA_WRITE;
  uintptr_t laddr = reinterpret_cast<uintptr_t>(source);
  uintptr_t raddr = reinterpret_cast<uintptr_t>(dest);
  /* only need lkey when RDMA_WRITE can't be inlined */
  uint32_t lkey = !Provider::template can_inline<Op>(nelems) ? provider().get_lkey(laddr) : 0;
  uint32_t rkey = provider().get_rkey(raddr);
  provider().template post_wqe_rma_single<Op>(laddr, lkey, raddr, rkey, nelems, post_options);
}

template <typename Provider>
template <typename... PostOptions>
__device__ void QueuePairSHMEM<Provider>::get_nbi(
    void *dest, const void *source, size_t nelems,
    const ActiveWFInfo& wf_info, PostOpt<PostOptions...> post_options) {
  static constexpr OpCode Op = OpCode::RDMA_READ;
  uintptr_t laddr = reinterpret_cast<uintptr_t>(dest);
  uintptr_t raddr = reinterpret_cast<uintptr_t>(source);
  uint32_t lkey = provider().get_lkey(laddr);
  uint32_t rkey = provider().get_rkey(raddr);
  provider().template post_wqe_rma<Op>(laddr, lkey, raddr, rkey, nelems, wf_info, post_options);
}

template <typename Provider>
template <typename... PostOptions>
__device__ void QueuePairSHMEM<Provider>::get_nbi_single(
    void *dest, const void *source, size_t nelems,
    PostOpt<PostOptions...> post_options) {
  static constexpr OpCode Op = OpCode::RDMA_READ;
  uintptr_t laddr = reinterpret_cast<uintptr_t>(dest);
  uintptr_t raddr = reinterpret_cast<uintptr_t>(source);
  uint32_t lkey = provider().get_lkey(laddr);
  uint32_t rkey = provider().get_rkey(raddr);
  provider().template post_wqe_rma_single<Op>(laddr, lkey, raddr, rkey, nelems, post_options);
}

#if 0
template <typename Provider>
template <typename... PostOptions>
__device__ uint64_t QueuePairSHMEM<Provider>::atomic_fetch_add(
    void *dest, uint64_t value,
    const ActiveWFInfo& wf_info, PostOpt<PostOptions...> post_options) {
  static constexpr OpCode Op = OpCode::ATOMIC_FA;
  static constexpr AMOFetchType Fetch = AMOFetchType::Blocking;
  uintptr_t laddr = /* TODO */
  uintptr_t raddr = reinterpret_cast<uintptr_t>(dest);
  uint32_t lkey = provider().get_lkey(laddr);
  uint32_t rkey = provider().get_rkey(raddr);
  provider().template post_wqe_amo<Op, Fetch>(laddr, lkey, raddr, rkey, value, 0, wf_info, post_options);
  quiet(wf_info);
  return *reinterpret_cast<uint64_t*>(laddr);
}

template <typename Provider>
template <typename... PostOptions>
__device__ uint64_t QueuePairSHMEM<Provider>::atomic_fetch_add_single(
    void *dest, uint64_t value,
    PostOpt<PostOptions...> post_options) {
  static constexpr OpCode Op = OpCode::ATOMIC_FA;
  static constexpr AMOFetchType Fetch = AMOFetchType::Blocking;
  uintptr_t laddr = /* TODO */
  uintptr_t raddr = reinterpret_cast<uintptr_t>(dest);
  uint32_t lkey = provider().get_lkey(laddr);
  uint32_t rkey = provider().get_rkey(raddr);
  provider().template post_wqe_amo_single<Op, Fetch>(laddr, lkey, raddr, rkey, value, 0, post_options);
  provider().quiet_single();
  return *reinterpret_cast<uint64_t*>(laddr);
}

template <typename Provider>
template <typename... PostOptions>
__device__ void QueuePairSHMEM<Provider>::atomic_fetch_add_nbi(
    uint64_t *fetch, void *dest, uint64_t value,
    const ActiveWFInfo& wf_info, PostOpt<PostOptions...> post_options) {
  static constexpr OpCode Op = OpCode::ATOMIC_FA;
  static constexpr AMOFetchType Fetch = AMOFetchType::NonBlocking;
  uintptr_t laddr = reinterpret_cast<uintptr_t>(fetch);
  uintptr_t raddr = reinterpret_cast<uintptr_t>(dest);
  uint32_t lkey = provider().get_lkey(laddr);
  uint32_t rkey = provider().get_rkey(raddr);
  provider().template post_wqe_amo<Op, Fetch>(laddr, lkey, raddr, rkey, value, 0, wf_info, post_options);
}

template <typename Provider>
template <typename... PostOptions>
__device__ void QueuePairSHMEM<Provider>::atomic_fetch_add_nbi_single(
    uint64_t *fetch, void *dest, uint64_t value,
    PostOpt<PostOptions...> post_options) {
  static constexpr OpCode Op = OpCode::ATOMIC_FA;
  static constexpr AMOFetchType Fetch = AMOFetchType::NonBlocking;
  uintptr_t laddr = reinterpret_cast<uintptr_t>(fetch);
  uintptr_t raddr = reinterpret_cast<uintptr_t>(dest);
  uint32_t lkey = provider().get_lkey(laddr);
  uint32_t rkey = provider().get_rkey(raddr);
  provider().template post_wqe_amo_single<Op, Fetch>(laddr, lkey, raddr, rkey, value, 0, post_options);
}

template <typename Provider>
template <typename... PostOptions>
__device__ void QueuePairSHMEM<Provider>::atomic_add(
    void *dest, uint64_t value,
    const ActiveWFInfo& wf_info, PostOpt<PostOptions...> post_options) {
  static constexpr OpCode Op = OpCode::ATOMIC_FA;
  static constexpr AMOFetchType Fetch = AMOFetchType::NonFetching;
  uintptr_t laddr = reinterpret_cast<uintptr_t>(nonfetching_atomic);
  uintptr_t raddr = reinterpret_cast<uintptr_t>(dest);
  uint32_t lkey = provider().get_lkey(laddr);
  uint32_t rkey = provider().get_rkey(raddr);
  provider().template post_wqe_amo<Op, Fetch>(laddr, lkey, raddr, rkey, value, 0, wf_info, post_options);
}

template <typename Provider>
template <typename... PostOptions>
__device__ void QueuePairSHMEM<Provider>::atomic_add_single(
    void *dest, uint64_t value,
    PostOpt<PostOptions...> post_options) {
  static constexpr OpCode Op = OpCode::ATOMIC_FA;
  static constexpr AMOFetchType Fetch = AMOFetchType::NonFetching;
  uintptr_t laddr = reinterpret_cast<uintptr_t>(nonfetching_atomic);
  uintptr_t raddr = reinterpret_cast<uintptr_t>(dest);
  uint32_t lkey = provider().get_lkey(laddr);
  uint32_t rkey = provider().get_rkey(raddr);
  provider().template post_wqe_amo_single<Op, Fetch>(laddr, lkey, raddr, rkey, value, 0, post_options);
}

template <typename Provider>
template <typename... PostOptions>
__device__ uint64_t QueuePairSHMEM<Provider>::atomic_compare_swap(
    void *dest, uint64_t cond, uint64_t value,
    const ActiveWFInfo& wf_info, PostOpt<PostOptions...> post_options) {
  static constexpr OpCode Op = OpCode::ATOMIC_CS;
  static constexpr AMOFetchType Fetch = AMOFetchType::Blocking;
  uintptr_t laddr = /* TODO */
  uintptr_t raddr = reinterpret_cast<uintptr_t>(dest);
  uint32_t lkey = provider().get_lkey(laddr);
  uint32_t rkey = provider().get_rkey(raddr);
  provider().template post_wqe_amo<Op, Fetch>(laddr, lkey, raddr, rkey, value, cond, wf_info, post_options);
  quiet(wf_info);
  return *reinterpret_cast<uint64_t*>(laddr);
}

template <typename Provider>
template <typename... PostOptions>
__device__ uint64_t QueuePairSHMEM<Provider>::atomic_compare_swap_single(
    void *dest, uint64_t cond, uint64_t value,
    PostOpt<PostOptions...> post_options) {
  static constexpr OpCode Op = OpCode::ATOMIC_CS;
  static constexpr AMOFetchType Fetch = AMOFetchType::Blocking;
  uintptr_t laddr = /* TODO */
  uintptr_t raddr = reinterpret_cast<uintptr_t>(dest);
  uint32_t lkey = provider().get_lkey(laddr);
  uint32_t rkey = provider().get_rkey(raddr);
  provider().template post_wqe_amo_single<Op, Fetch>(laddr, lkey, raddr, rkey, value, cond, post_options);
  quiet_single();
  return *reinterpret_cast<uint64_t*>(laddr);
}

template <typename Provider>
template <typename... PostOptions>
__device__ void QueuePairSHMEM<Provider>::atomic_compare_swap_nbi(
    uint64_t *fetch, void *dest, uint64_t cond, uint64_t value,
    const ActiveWFInfo& wf_info, PostOpt<PostOptions...> post_options) {
  static constexpr OpCode Op = OpCode::ATOMIC_CS;
  static constexpr AMOFetchType Fetch = AMOFetchType::NonBlocking;
  uintptr_t laddr = reinterpret_cast<uintptr_t>(fetch);
  uintptr_t raddr = reinterpret_cast<uintptr_t>(dest);
  uint32_t lkey = provider().get_lkey(laddr);
  uint32_t rkey = provider().get_rkey(raddr);
  provider().template post_wqe_amo<Op, Fetch>(laddr, lkey, raddr, rkey, value, cond, wf_info, post_options);
}

template <typename Provider>
template <typename... PostOptions>
__device__ void QueuePairSHMEM<Provider>::atomic_compare_swap_nbi_single(
    uint64_t *fetch, void *dest, uint64_t cond, uint64_t value,
    PostOpt<PostOptions...> post_options) {
  static constexpr OpCode Op = OpCode::ATOMIC_CS;
  static constexpr AMOFetchType Fetch = AMOFetchType::NonBlocking;
  uintptr_t laddr = reinterpret_cast<uintptr_t>(fetch);
  uintptr_t raddr = reinterpret_cast<uintptr_t>(dest);
  uint32_t lkey = provider().get_lkey(laddr);
  uint32_t rkey = provider().get_rkey(raddr);
  provider().template post_wqe_amo_single<Op, Fetch>(laddr, lkey, raddr, rkey, value, cond, post_options);
}

template <typename Provider>
template <typename... PostOptions>
__device__ void QueuePairSHMEM<Provider>::atomic_compare_swap_nofetch(
    void *dest, uint64_t cond, uint64_t value,
    const ActiveWFInfo& wf_info, PostOpt<PostOptions...> post_options) {
  static constexpr OpCode Op = OpCode::ATOMIC_CS;
  static constexpr AMOFetchType Fetch = AMOFetchType::NonFetching;
  uintptr_t laddr = reinterpret_cast<uintptr_t>(nonfetching_atomic);
  uintptr_t raddr = reinterpret_cast<uintptr_t>(dest);
  uint32_t lkey = provider().get_lkey(laddr);
  uint32_t rkey = provider().get_rkey(raddr);
  provider().template post_wqe_amo<Op, Fetch>(laddr, lkey, raddr, rkey, value, cond, wf_info, post_options);
}

template <typename Provider>
template <typename... PostOptions>
__device__ void QueuePairSHMEM<Provider>::atomic_compare_swap_nofetch_single(
    void *dest, uint64_t cond, uint64_t value,
    PostOpt<PostOptions...> post_options) {
  static constexpr OpCode Op = OpCode::ATOMIC_CS;
  static constexpr AMOFetchType Fetch = AMOFetchType::NonFetching;
  uintptr_t laddr = reinterpret_cast<uintptr_t>(nonfetching_atomic);
  uintptr_t raddr = reinterpret_cast<uintptr_t>(dest);
  uint32_t lkey = provider().get_lkey(laddr);
  uint32_t rkey = provider().get_rkey(raddr);
  provider().template post_wqe_amo_single<Op, Fetch>(laddr, lkey, raddr, rkey, value, cond, post_options);
}
#endif

template <typename Provider>
template <typename... PostOptions>
__device__ uint64_t QueuePairSHMEM<Provider>::atomic_fetch_add(
    void *dest, uint64_t value,
    const ActiveWFInfo& wf_info, PostOpt<PostOptions...> post_options) {
  static constexpr OpCode Op = OpCode::ATOMIC_FA;
  static constexpr AMOFetchType Fetch = AMOFetchType::Blocking;
  uintptr_t raddr = reinterpret_cast<uintptr_t>(dest);
  uint32_t rkey = provider().get_rkey(raddr);
  return provider().template post_wqe_amo<Op, Fetch>(raddr, rkey, value, 0, wf_info, post_options);
}

template <typename Provider>
template <typename... PostOptions>
__device__ uint64_t QueuePairSHMEM<Provider>::atomic_fetch_add_single(
    void *dest, uint64_t value,
    PostOpt<PostOptions...> post_options) {
  static constexpr OpCode Op = OpCode::ATOMIC_FA;
  static constexpr AMOFetchType Fetch = AMOFetchType::Blocking;
  uintptr_t raddr = reinterpret_cast<uintptr_t>(dest);
  uint32_t rkey = provider().get_rkey(raddr);
  return provider().template post_wqe_amo_single<Op, Fetch>(raddr, rkey, value, 0, post_options);
}

template <typename Provider>
template <typename... PostOptions>
__device__ void QueuePairSHMEM<Provider>::atomic_add(
    void *dest, uint64_t value,
    const ActiveWFInfo& wf_info, PostOpt<PostOptions...> post_options) {
  static constexpr OpCode Op = OpCode::ATOMIC_FA;
  static constexpr AMOFetchType Fetch = AMOFetchType::NonFetching;
  uintptr_t raddr = reinterpret_cast<uintptr_t>(dest);
  uint32_t rkey = provider().get_rkey(raddr);
  provider().template post_wqe_amo<Op, Fetch>(raddr, rkey, value, 0, wf_info, post_options);
}

template <typename Provider>
template <typename... PostOptions>
__device__ void QueuePairSHMEM<Provider>::atomic_add_single(
    void *dest, uint64_t value,
    PostOpt<PostOptions...> post_options) {
  static constexpr OpCode Op = OpCode::ATOMIC_FA;
  static constexpr AMOFetchType Fetch = AMOFetchType::NonFetching;
  uintptr_t raddr = reinterpret_cast<uintptr_t>(dest);
  uint32_t rkey = provider().get_rkey(raddr);
  provider().template post_wqe_amo_single<Op, Fetch>(raddr, rkey, value, 0, post_options);
}

template <typename Provider>
template <typename... PostOptions>
__device__ uint64_t QueuePairSHMEM<Provider>::atomic_compare_swap(
    void *dest, uint64_t cond, uint64_t value,
    const ActiveWFInfo& wf_info, PostOpt<PostOptions...> post_options) {
  static constexpr OpCode Op = OpCode::ATOMIC_CS;
  static constexpr AMOFetchType Fetch = AMOFetchType::Blocking;
  uintptr_t raddr = reinterpret_cast<uintptr_t>(dest);
  uint32_t rkey = provider().get_rkey(raddr);
  return provider().template post_wqe_amo<Op, Fetch>(raddr, rkey, value, cond, wf_info, post_options);
}

template <typename Provider>
template <typename... PostOptions>
__device__ uint64_t QueuePairSHMEM<Provider>::atomic_compare_swap_single(
    void *dest, uint64_t cond, uint64_t value,
    PostOpt<PostOptions...> post_options) {
  static constexpr OpCode Op = OpCode::ATOMIC_CS;
  static constexpr AMOFetchType Fetch = AMOFetchType::Blocking;
  uintptr_t raddr = reinterpret_cast<uintptr_t>(dest);
  uint32_t rkey = provider().get_rkey(raddr);
  return provider().template post_wqe_amo_single<Op, Fetch>(raddr, rkey, value, cond, post_options);
}

template <typename Provider>
template <typename... PostOptions>
__device__ void QueuePairSHMEM<Provider>::atomic_compare_swap_nofetch(
    void *dest, uint64_t cond, uint64_t value,
    const ActiveWFInfo& wf_info, PostOpt<PostOptions...> post_options) {
  static constexpr OpCode Op = OpCode::ATOMIC_CS;
  static constexpr AMOFetchType Fetch = AMOFetchType::NonFetching;
  uintptr_t raddr = reinterpret_cast<uintptr_t>(dest);
  uint32_t rkey = provider().get_rkey(raddr);
  provider().template post_wqe_amo<Op, Fetch>(raddr, rkey, value, cond, wf_info, post_options);
}

template <typename Provider>
template <typename... PostOptions>
__device__ void QueuePairSHMEM<Provider>::atomic_compare_swap_nofetch_single(
    void *dest, uint64_t cond, uint64_t value,
    PostOpt<PostOptions...> post_options) {
  static constexpr OpCode Op = OpCode::ATOMIC_CS;
  static constexpr AMOFetchType Fetch = AMOFetchType::NonFetching;
  uintptr_t raddr = reinterpret_cast<uintptr_t>(dest);
  uint32_t rkey = provider().get_rkey(raddr);
  provider().template post_wqe_amo_single<Op, Fetch>(raddr, rkey, value, cond, post_options);
}

template <typename Provider>
__device__ void QueuePairSHMEM<Provider>::quiet(const ActiveWFInfo& wf_info) {
  if (wf_info.is_pe_group_first) {
    provider().quiet_single();
  }
}



/*
 * @brief CRTP base class for Provider-specific Queue Pair implementations.
 */
template <typename Provider>
class QueuePairBase : public QueuePairSHMEM<Provider> {
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

  /**
   * @brief Enumeration of the opcodes for Write, Read, Fetch-Add, and Compare-and-Swap.
   *
   * A definition must be provided by the QueuePairTraits<Provider> specialization
   * of each Provider subclass.
   */
  using OpCode = typename Traits::OpCode;

  /**
   * @brief Constant defining the endianness required by the provider. Used for e.g. lkey and rkey.
   *
   * A definition must be provided by the QueuePairTraits<Provider> specialization
   * of each Provider subclass.
   */
  static constexpr endian::Order ProviderEndianness = Traits::Endianness;
/**@}*/



/**
 * @name Constructors and Destructors
 *
 * @{
 */
public:
  /**
   * @brief Constructor.
   *
   * @param[in] qpn Queue Pair number.
   * @param[in] base_heap Base address of local heap.
   * @param[in] heap_size Size of heap, in bytes.
   * @param[in] lkey LKey of local heap.
   * @param[in] rkey RKey of remote heap.
   * @param[in] pd IBVerbs Protection Domain for registering additional buffers and heaps.
   */
  __host__ explicit QueuePairBase(uint32_t qpn, void *base_heap, size_t heap_size,
                                  uint32_t lkey, uint32_t rkey, struct ibv_pd* pd);

protected:
  /**
   * @brief Copy and Move Constructors and Assignment Operators.
   */
  /* NOTE: do we need to create definitions for these? define as deleted? */
  /* move-only type? */

  /*
   * @brief Copy constructor is deleted.
   */
  __host__ QueuePairBase(const QueuePairBase& other) = delete;

  /*
   * @brief Copy assignment operator is deleted.
   */
  __host__ QueuePairBase& operator=(const QueuePairBase& other) = delete;

  /*
   * @brief Move constructor.
   *
   * Performs a member-wise move of all data members from other to *this,
   * then resets allocated data members of other so that it can be safely reused or destroyed.
   *
   * @param[in,out] other QueuePairBase object to move from.
   */
  __host__ QueuePairBase(QueuePairBase&& other) noexcept;

  /*
   * @brief Move assignment operator.
   *
   * Cleans up all resources allocated by *this,
   * performs a member-wise move of all data members from other to *this,
   * then resets allocated data members of other so that it can be safely reused or destroyed.
   *
   * @param[in,out] other QueuePairBase object to move from.
   * @return *this
   */
  __host__ QueuePairBase& operator=(QueuePairBase&& other);

  /**
   * @brief Destructor.
   *
   * Cleans up all resources allocated by *this.
   */
  __host__ ~QueuePairBase();
/**@}*/



/**
 * @name Buffer Registration
 *
 * @{
 */
public:
  /**
   * @brief Register buffer for use as local address in rocSHMEM routines
   *
   * @param[in] addr Base address of buffer.
   * @param[in] length Length of buffer.
   *
   * @retval ROCSHMEM_SUCCESS Buffer registered successfully.
   * @retval ROCSHMEM_ERROR Buffer could not be registered.
   */
  __host__ int buffer_register(void *addr, size_t length);

  /**
   * @brief Unregister buffer.
   * Buffer must have previously been registered with buffer_register(void *, size_t).
   *
   * @param[in] addr Base address of buffer.
   *
   * @retval ROCSHMEM_SUCCESS Buffer unregistered successfully.
   */
  __host__ int buffer_unregister(void *addr);

  /**
   * @brief Unregister all registered buffers.
   *
   * @retval ROCSHMEM_SUCCESS Buffers unregistered successfully.
   */
  __host__ int buffer_unregister_all();

  /**
   * @brief Retrieve LKey for address.
   *
   * @param[in] addr Address to lookup LKey of.
   *
   * @return LKey for addr or std::numeric_limits<uint32_t>::max() if not found.
   * Endianness of returned LKey value is ProviderEndianness.
   */
  __device__ __forceinline__ uint32_t get_lkey(uintptr_t addr);

  /**
   * @brief Retrieve RKey for address.
   *
   * @param[in] addr Address to lookup RKey of.
   *
   * @return RKey for addr or std::numeric_limits<uint32_t>::max() if not found.
   * Endianness of returned RKey value is ProviderEndianness.
   */
  __device__ __forceinline__ uint32_t get_rkey(uintptr_t addr);

  /*
   * @brief Query whether data can be inlined into a WQE.
   *
   * @param[in] size Size of data carried by the WQE.
   * @tparam Op OpCode for this WQE.
   *
   * @return True if size bytes of data can be inlined into a WQE with OpCode Op, else false.
   */
  template <OpCode Op>
  static __device__ constexpr bool can_inline(size_t size);

  /**
   * @brief Convert value to ProviderEndianness, byteswapping if necessary.
   *
   * @param[in] val Value to convert.
   * @tparam T Type of val.
   *
   * @return endian::from_native<ProviderEndianness, T>(val)
   */
  template <typename T>
  static __host__ __device__ constexpr T to_provider_endianness(T val);
/**@}*/



/**
 * @name Internal
 *
 * @{
 */
protected:
  /*
   * @brief Provider is a friend of QueuePairBase<Provider>.
   */
  friend Provider;

  template <AMOFetchType Fetch>
  __device__ constexpr uint64_t* get_atomic_addr();

  template <AMOFetchType Fetch>
  __device__ constexpr uint32_t get_atomic_lkey();

private:
  __device__ Provider& provider() {
    return static_cast<Provider&>(*this);
  }

  template <typename T>
  __host__ std::tuple<T*, struct ibv_mr*, uint32_t> allocate_and_register(size_t size, int access);
/**@}*/



/**
 * @name Non-static Data Members
 *
 * @{
 */
public:
  uintptr_t base_heap;
  size_t heap_size;

protected:
  // Used in most WQEs
  uint32_t qp_num;
  uint32_t lkey;
  uint32_t rkey;
  uint32_t fetching_atomic_idx{0};

private:
  // Used by get_lkey
  size_t num_user_buffers{0};
  BufferInfo* buffer_info{nullptr};

protected:
  // Used in atomic WQEs
  uint64_t* fetching_atomic{nullptr};
  uint64_t* nonfetching_atomic{nullptr};

  uint32_t fetching_atomic_lkey{0};
  uint32_t nonfetching_atomic_lkey{0};

  static constexpr size_t FETCHING_ATOMIC_CNT{1024};
  static_assert(FETCHING_ATOMIC_CNT % WF_SIZE == 0);
  using FreeListT = FreeList<uint64_t*>;
  FreeListT* fetching_atomic_freelist{nullptr};

  HIPAllocator allocator{};

private:
  // Used by host in buffer_register, buffer_unregister
  std::unordered_map<void*, BufferHostInfo> buffer_info_map{};
  iterator_freelist<BufferInfo*> buffer_info_freelist{};
  struct ibv_pd* pd;
  struct ibv_mr* fetching_atomic_mr{nullptr};
  struct ibv_mr* nonfetching_atomic_mr{nullptr};
/**@}*/
};



template <typename Provider>
__host__ QueuePairBase<Provider>::QueuePairBase(uint32_t qpn, void *base_heap, size_t heap_size,
                                                uint32_t lkey, uint32_t rkey, struct ibv_pd* pd)
  : base_heap{reinterpret_cast<uintptr_t>(base_heap)},
    heap_size{heap_size},
    qp_num{qpn},
    lkey{to_provider_endianness(lkey)},
    rkey{to_provider_endianness(rkey)},
    pd{pd} {
  int access = IBV_ACCESS_LOCAL_WRITE
             | IBV_ACCESS_REMOTE_WRITE
             | IBV_ACCESS_REMOTE_READ
             | IBV_ACCESS_REMOTE_ATOMIC;

  if (envvar::gda::pcie_relaxed_ordering) {
    access |= IBV_ACCESS_RELAXED_ORDERING;
  }

  // Allocate and register the fetching and nonfetching atomics arrays
  std::tie(nonfetching_atomic, nonfetching_atomic_mr, nonfetching_atomic_lkey)
      = allocate_and_register<uint64_t>(1, access);
  std::tie(fetching_atomic, fetching_atomic_mr, fetching_atomic_lkey)
      = allocate_and_register<uint64_t>(FETCHING_ATOMIC_CNT, access);

  allocator.allocate(reinterpret_cast<void**>(&fetching_atomic_freelist), sizeof(FreeListT));
  new (fetching_atomic_freelist) FreeListT{allocator};

  int deviceId;
  CHECK_HIP(hipGetDevice(&deviceId));
  int wf_size = get_wf_size(deviceId);
  for (size_t i = 0; i < FETCHING_ATOMIC_CNT; i += wf_size) {
    fetching_atomic_freelist->push_back(&fetching_atomic[i]);
  }

  /* Setup User Buffer Registration Mechanism */
  num_user_buffers = envvar::gda::num_user_buffers;

  CHECK_HIP(hipMalloc(&buffer_info, sizeof(BufferInfo) * num_user_buffers));
  CHECK_HIP(hipMemset(buffer_info, 0, sizeof(BufferInfo) * num_user_buffers));

  /* Reserve memory to register up to num_user_buffers */
  buffer_info_map.reserve(num_user_buffers);

  /* Create buffer registration info freelist */
  buffer_info_freelist = iterator_freelist<BufferInfo*>{buffer_info, num_user_buffers};
}

template <typename Provider>
__host__ QueuePairBase<Provider>::QueuePairBase(QueuePairBase&& other) noexcept
  : base_heap               {std::move(other.base_heap)},
    heap_size               {std::move(other.heap_size)},
    qp_num                  {std::move(other.qp_num)},
    lkey                    {std::move(other.lkey)},
    rkey                    {std::move(other.rkey)},
    fetching_atomic_idx     {std::move(other.fetching_atomic_idx)},
    num_user_buffers        {std::move(other.num_user_buffers)},
    buffer_info             {std::move(other.buffer_info)},
    fetching_atomic         {std::move(other.fetching_atomic)},
    nonfetching_atomic      {std::move(other.nonfetching_atomic)},
    fetching_atomic_lkey    {std::move(other.fetching_atomic_lkey)},
    nonfetching_atomic_lkey {std::move(other.nonfetching_atomic_lkey)},
    fetching_atomic_freelist{std::move(other.fetching_atomic_freelist)},
    allocator               {std::move(other.allocator)},
    buffer_info_map         {std::move(other.buffer_info_map)},
    buffer_info_freelist    {std::move(other.buffer_info_freelist)},
    pd                      {std::move(other.pd)},
    fetching_atomic_mr      {std::move(other.fetching_atomic_mr)},
    nonfetching_atomic_mr   {std::move(other.nonfetching_atomic_mr)} {
  other.buffer_info              = nullptr;
  other.fetching_atomic          = nullptr;
  other.nonfetching_atomic       = nullptr;
  other.fetching_atomic_freelist = nullptr;
  other.fetching_atomic_mr       = nullptr;
  other.nonfetching_atomic_mr    = nullptr;
}

template <typename Provider>
__host__ QueuePairBase<Provider>& QueuePairBase<Provider>::operator=(QueuePairBase&& other) {
  int err = 0;

  /* Step 1: ensure all resources in *this are deallocated */
  if (!buffer_info_map.empty()) {
    LOG_WARN("Unmatched buffer_register detected: "
             "move assignment operator %s called, but buffer registration map is not empty!",
             __PRETTY_FUNCTION__ );
    /* Deregister every memory region registered with this QP */
    for (auto&& [addr, host_info] : buffer_info_map) {
      err = ibv.dereg_mr(host_info.mr);
      CHECK_ZERO(err, "ibv_dereg_mr (QueuePairBase<Provider>::operator=(QueuePairBase&&))");
    }
  }

  if (buffer_info) {
    CHECK_HIP(hipFree(buffer_info));
  }

  if (fetching_atomic_freelist) {
    fetching_atomic_freelist->~FreeListT();
    allocator.deallocate(static_cast<void*>(fetching_atomic_freelist));
  }

  if (fetching_atomic_mr) {
    err = ibv.dereg_mr(fetching_atomic_mr);
    CHECK_ZERO(err, "ibv_dereg_mr (fetching_atomic)");
  }

  if (fetching_atomic) {
    allocator.deallocate(static_cast<void*>(fetching_atomic));
  }

  if (nonfetching_atomic_mr) {
    err = ibv.dereg_mr(nonfetching_atomic_mr);
    CHECK_ZERO(err, "ibv_dereg_mr (nonfetching_atomic)");
  }

  if (nonfetching_atomic) {
    allocator.deallocate(static_cast<void*>(nonfetching_atomic));
  }

  /* Step 2: member-wise move of all data members from other to *this */
  base_heap                = std::move(other.base_heap);
  heap_size                = std::move(other.heap_size);
  qp_num                   = std::move(other.qp_num);
  lkey                     = std::move(other.lkey);
  rkey                     = std::move(other.rkey);
  fetching_atomic_idx      = std::move(other.fetching_atomic_idx);
  num_user_buffers         = std::move(other.num_user_buffers);
  buffer_info              = std::move(other.buffer_info);
  fetching_atomic          = std::move(other.fetching_atomic);
  nonfetching_atomic       = std::move(other.nonfetching_atomic);
  fetching_atomic_lkey     = std::move(other.fetching_atomic_lkey);
  nonfetching_atomic_lkey  = std::move(other.nonfetching_atomic_lkey);
  fetching_atomic_freelist = std::move(other.fetching_atomic_freelist);
  allocator                = std::move(other.allocator);
  buffer_info_map          = std::move(other.buffer_info_map);
  buffer_info_freelist     = std::move(other.buffer_info_freelist);
  pd                       = std::move(other.pd);
  fetching_atomic_mr       = std::move(other.fetching_atomic_mr);
  nonfetching_atomic_mr    = std::move(other.nonfetching_atomic_mr);

  /* Step 3: reset allocated data members of other so that it can be safely reused or destroyed */
  other.buffer_info              = nullptr;
  other.fetching_atomic          = nullptr;
  other.nonfetching_atomic       = nullptr;
  other.fetching_atomic_freelist = nullptr;
  other.fetching_atomic_mr       = nullptr;
  other.nonfetching_atomic_mr    = nullptr;

  /* Step 4: return *this */
  return *this;
}

template <typename Provider>
__host__ QueuePairBase<Provider>::~QueuePairBase() {
  int err = 0;

  if (!buffer_info_map.empty()) {
    LOG_WARN("Unmatched buffer_register detected: "
             "destructor %s called, but buffer registration map is not empty!",
             __PRETTY_FUNCTION__ );
    /* Deregister every memory region registered with this QP */
    for (auto&& [addr, host_info] : buffer_info_map) {
      err = ibv.dereg_mr(host_info.mr);
      CHECK_ZERO(err, "ibv_dereg_mr (QueuePairBase<Provider>::~QueuePairBase)");
    }
  }

  if (buffer_info) {
    CHECK_HIP(hipFree(buffer_info));
  }

  if (fetching_atomic_freelist) {
    fetching_atomic_freelist->~FreeListT();
    allocator.deallocate(static_cast<void*>(fetching_atomic_freelist));
  }

  if (fetching_atomic_mr) {
    err = ibv.dereg_mr(fetching_atomic_mr);
    CHECK_ZERO(err, "ibv_dereg_mr (fetching_atomic)");
  }

  if (fetching_atomic) {
    allocator.deallocate(static_cast<void*>(fetching_atomic));
  }

  if (nonfetching_atomic_mr) {
    err = ibv.dereg_mr(nonfetching_atomic_mr);
    CHECK_ZERO(err, "ibv_dereg_mr (nonfetching_atomic)");
  }

  if (nonfetching_atomic) {
    allocator.deallocate(static_cast<void*>(nonfetching_atomic));
  }
}

template <typename Provider>
template <typename T>
__host__ std::tuple<T*, struct ibv_mr*, uint32_t>
QueuePairBase<Provider>::allocate_and_register(size_t count, int access) {
  void* ptr = nullptr;
  size_t size = sizeof(T) * count;
  allocator.allocate(&ptr, size);
  CHECK_HIP(hipMemset(ptr, 0, size));
  struct ibv_mr* mr = ibv.reg_mr(pd, ptr, size, access, &allocator);
  CHECK_NNULL(mr, "ibv_reg_mr");
  return {static_cast<T*>(ptr), mr, to_provider_endianness(mr->lkey)};
}

template <typename Provider>
__host__ int QueuePairBase<Provider>::buffer_register(void *addr, size_t length) {
  if (buffer_info_freelist.empty()) {
    assert(buffer_info_map.size() == num_user_buffers);
    LOG_WARN("Unable to register user buffer (%p, %zu) with QP. "
             "Please increase the value of %s.",
             addr, length, envvar::gda::num_user_buffers.get_name().c_str());
    return ROCSHMEM_ERROR;
  }

  /* Register addr */
  int access = IBV_ACCESS_LOCAL_WRITE
             | IBV_ACCESS_REMOTE_WRITE
             | IBV_ACCESS_REMOTE_READ
             | IBV_ACCESS_REMOTE_ATOMIC;

  if (envvar::gda::pcie_relaxed_ordering) {
    access |= IBV_ACCESS_RELAXED_ORDERING;
  }

  struct ibv_mr* mr = ibv.reg_mr(pd, addr, length, access, &allocator);
  CHECK_NNULL(mr, "ibv_reg_mr (buffer_register)");

  /* Get pointer to next free buffer_info entry */
  BufferInfo* info_ptr = buffer_info_freelist.top();

  /* Try inserting buffer host info into buffer_info_map; detects whether this is a duplicate */
  auto&& [it, inserted] = buffer_info_map.try_emplace(addr, mr, info_ptr);

  if (inserted) [[likely]] {
    /* Insertion succceeded, pop the buffer_info entry from the freelist */
    buffer_info_freelist.pop();
    /* Copy buffer info to device */
    BufferInfo info{reinterpret_cast<uintptr_t>(addr), length, to_provider_endianness(mr->lkey)};
    CHECK_HIP(hipMemcpy(info_ptr, &info, sizeof(BufferInfo), hipMemcpyHostToDevice));
    return ROCSHMEM_SUCCESS;
  } else {
    auto&& [key, host_info] = *it;
    LOG_WARN("Unable to register user buffer (%p, %zu) with QP: "
             "already registered with length=%zu, lkey=%u, rkey=%u.",
             addr, length, host_info.mr->length, host_info.mr->lkey, host_info.mr->rkey);
    /* Deregister addr */
    int err = ibv.dereg_mr(mr);
    CHECK_ZERO(err, "ibv_dereg_mr (buffer_register)");
    return ROCSHMEM_ERROR;
  }
}

template <typename Provider>
__host__ int QueuePairBase<Provider>::buffer_unregister(void *addr) {
  /* Lookup buffer host info */
  auto it = buffer_info_map.find(addr);

  if (it != buffer_info_map.end()) [[likely]] {
    auto&& [key, host_info] = *it;
    /* Reset buffer_info entry and push to freelist */
    CHECK_HIP(hipMemset(host_info.info_ptr, 0, sizeof(BufferInfo)));
    buffer_info_freelist.push(host_info.info_ptr);
    /* Deregister addr */
    int err = ibv.dereg_mr(host_info.mr);
    CHECK_ZERO(err, "ibv_dereg_mr (buffer_unregister)");
    /* Remove from map */
    buffer_info_map.erase(it);
    return ROCSHMEM_SUCCESS;
  } else {
    LOG_WARN("Unable to unregister user buffer (%p) with this QP: "
             "user buffer not registered.", addr);
    return ROCSHMEM_ERROR;
  }
}

template <typename Provider>
__host__ int QueuePairBase<Provider>::buffer_unregister_all() {
  int err = 0;

  /* Deregister every memory region registered with this QP */
  for (auto&& [addr, host_info] : buffer_info_map) {
    err = ibv.dereg_mr(host_info.mr);
    CHECK_ZERO(err, "ibv_dereg_mr (QueuePairBase<Provider>::~QueuePairBase)");
  }
  buffer_info_map.clear();

  /* Clear all buffer_info slots */
  CHECK_HIP(hipMemset(buffer_info, 0, sizeof(BufferInfo) * num_user_buffers));

  /* Reset buffer_info_freelist */
  buffer_info_freelist.reset(buffer_info, num_user_buffers);
  return ROCSHMEM_SUCCESS;
}

template <typename Provider>
__device__ __forceinline__ uint32_t QueuePairBase<Provider>::get_lkey(uintptr_t addr) {
  /* Check if in heap */
  if (is_ptr_in_range(base_heap, heap_size, addr)) [[likely]] {
    return lkey;
  }

  /* Get the correct lkey for the user buffer */
  for (size_t i = 0; i < num_user_buffers; i++) {
    if (is_ptr_in_range(buffer_info[i].addr, buffer_info[i].length, addr)) {
      return buffer_info[i].lkey;
    }
  }

  LOGD_ERROR_ABORT("Valid lkey for address %p not found", reinterpret_cast<void*>(addr));
  return std::numeric_limits<uint32_t>::max();
}

template <typename Provider>
__device__ __forceinline__ uint32_t QueuePairBase<Provider>::get_rkey([[maybe_unused]] uintptr_t addr) {
  return rkey;
}

template <typename Provider>
template <typename QueuePairBase<Provider>::OpCode Op>
__device__ constexpr bool QueuePairBase<Provider>::can_inline(size_t size) {
  if constexpr (Op == OpCode::RDMA_WRITE) {
    return size <= Traits::InlineThreshold;
  } else {
    return  false;
  }
}

template <typename Provider>
template <typename T>
__host__ __device__ constexpr T QueuePairBase<Provider>::to_provider_endianness(T val) {
  return endian::from_native<ProviderEndianness, T>(val);
}

template <typename Provider>
template <AMOFetchType Fetch>
__device__ constexpr uint64_t* QueuePairBase<Provider>::get_atomic_addr() {
  static_assert(Fetch != AMOFetchType::NonBlocking);
  if constexpr (Fetch == AMOFetchType::Blocking) {
    return fetching_atomic;
  } else if constexpr (Fetch == AMOFetchType::NonFetching) {
    return nonfetching_atomic;
  }
}

template <typename Provider>
template <AMOFetchType Fetch>
__device__ constexpr uint32_t QueuePairBase<Provider>::get_atomic_lkey() {
  static_assert(Fetch != AMOFetchType::NonBlocking);
  if constexpr (Fetch == AMOFetchType::Blocking) {
    return fetching_atomic_lkey;
  } else if constexpr (Fetch == AMOFetchType::NonFetching) {
    return nonfetching_atomic_lkey;
  }
}

}  // namespace rocshmem

#endif  // LIBRARY_SRC_GDA_QUEUE_PAIR_HPP_
