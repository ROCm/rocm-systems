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

#ifndef LIBRARY_SRC_GDA_QUEUE_PAIR_OPTION_HPP_
#define LIBRARY_SRC_GDA_QUEUE_PAIR_OPTION_HPP_

/**
 * @file queue_pair_option.hpp
 *
 * @section DESCRIPTION
 * An IB QueuePair (SQ and CQ) that the device can use to perform network
 * operations. Most important rocSHMEM operations are performed by this
 * class.
 */

#include <type_traits>

#include <hip/hip_runtime.h>

namespace rocshmem {

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

  /* Clang versions < 22 implicitly treats deduction guides as __host__ functions
   * unless marked otherwise by explicit attributes.
   *
   * Clang versions >= 22 implicitly treats all deduction guides as __host__ __device__
   * and issues a warning when they have explicit attributes.
   * See https://clang.llvm.org/docs/HIPSupport.html#deduction-guides for details.
   *
   * Disable this warning so that Clang >= 22 doesn't cause issues;
   * remove the attributes once we no longer support older compiler versions. */
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wdeprecated-attributes"
  /* deduction guide, required before C++20 */
  template <typename... Options> __host__ __device__ PostOpt(Options...) -> PostOpt<Options...>;
#pragma clang diagnostic pop

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

}  // namespace rocshmem

#endif  // LIBRARY_SRC_GDA_QUEUE_PAIR_OPTION_HPP_
