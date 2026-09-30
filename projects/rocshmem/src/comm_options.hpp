// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#ifndef LIBRARY_SRC_COMM_OPTIONS_HPP_
#define LIBRARY_SRC_COMM_OPTIONS_HPP_

/**
 * @file comm_options.hpp
 *
 * @section DESCRIPTION
 * Backend-agnostic, compile-time options for communication operations.
 *
 * Each option is a tag describing a property of a communication call. Backends
 * consume the tags they understand and ignore the rest, so the same CommOpt can
 * be threaded uniformly through every API communication call. These options
 * were historically the GDA queue-pair "PostOpt" options; the tag names are
 * preserved here (a later commit gives them backend-agnostic semantic names).
 */

#include <type_traits>

#include <hip/hip_runtime.h>

// ActiveWFInfo (used by the CQ-update helper). It only depends on util.hpp, so
// including it here does not couple CommOpt to any GDA transport.
#include "gda/queue_pair/queue_pair_common.hpp"

namespace rocshmem {

namespace CommOption {
  /*
   * @brief Helper alias for option tag types before C++26 std::constant_wrapper.
   *
   * Options are defined as types derived from constant_t<V>
   * so that type-based tag dispatching can be used.
   */
  template <auto V>
  using constant_t = std::integral_constant<decltype(V), V>;

  /*
   * @brief Type trait for defining default values for options.
   */
  template <template<auto> typename option_tag> struct default_option { };

  /*
   * @brief Helper type alias for the default_option type trait.
   */
  template <template<auto> typename option_tag>
  using default_option_t = typename default_option<option_tag>::type;

  /*
   * @brief Helper variable for the default option value.
   */
  template <template<auto> typename option_tag>
  constexpr inline auto default_option_v = default_option_t<option_tag>::value;

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

  template <> struct default_option<ring_db_tag> {
    /* default: DO ring the doorbell */
    using type = ring_db_tag<true>;
  };

  /*
   * @brief Option: whether thread safety will be enforced.
   */
  template <bool thread_safe>
  struct thread_safe_tag : constant_t<thread_safe> { };

  template <> struct default_option<thread_safe_tag> {
    /* default: DO use thread safety */
    using type = thread_safe_tag<true>;
  };

  /*
   * @brief Option: whether the number of available send queue entries will be checked.
   */
  template <bool check_sq>
  struct check_sq_tag : constant_t<check_sq> { };

  template <> struct default_option<check_sq_tag> {
    /* default: DO check the SQ */
    using type = check_sq_tag<true>;
  };

  /*
   * @brief Option: which threads' WQEs will generate CQEs.
   */
  template <UpdateThread update_cq>
  struct update_cq_tag : constant_t<update_cq> { };

  template <> struct default_option<update_cq_tag> {
    /* default: all WQEs update the CQ */
    using type = update_cq_tag<UpdateThread::All>;
  };

  /* forward declaration */
  template <typename... Options> struct CommOpt;

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
  template <typename... Options> __host__ __device__ CommOpt(Options...) -> CommOpt<Options...>;
#pragma clang diagnostic pop

  /* Base case with all options defined */
  template <bool ring_db, bool thread_safe, bool check_sq, UpdateThread update_cq>
  struct CommOpt<ring_db_tag<ring_db>,
                 thread_safe_tag<thread_safe>,
                 check_sq_tag<check_sq>,
                 update_cq_tag<update_cq>> {
    /* explicitly-defaulted default constructor */
    __host__ __device__ constexpr CommOpt() = default;
    /* constructor for type deduction from tags */
    __host__ __device__ constexpr CommOpt(ring_db_tag<ring_db>,
                                          thread_safe_tag<thread_safe>,
                                          check_sq_tag<check_sq>,
                                          update_cq_tag<update_cq>) { }

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
   * else matches CommOpt<ring_db_tag, thread_safe_tag, check_sq_tag, update_cq_tag> */
  template <bool ring_db, bool thread_safe, bool check_sq, UpdateThread update_cq,
            typename... Options>
  struct CommOpt<ring_db_tag<ring_db>,
                 thread_safe_tag<thread_safe>,
                 check_sq_tag<check_sq>,
                 update_cq_tag<update_cq>,
                 Options...> {
    static_assert(sizeof...(Options) == 0, "Too many or invalid options");
  };

  /* Missing update_cq_tag,
   * else matches CommOpt<ring_db_tag, thread_safe_tag, check_sq_tag, update_cq_tag, Options...> */
  template <bool ring_db, bool thread_safe, bool check_sq, typename... Options>
  struct CommOpt<ring_db_tag<ring_db>,
                 thread_safe_tag<thread_safe>,
                 check_sq_tag<check_sq>,
                 Options...>
       : CommOpt<ring_db_tag<ring_db>,
                 thread_safe_tag<thread_safe>,
                 check_sq_tag<check_sq>,
                 default_option_t<update_cq_tag>,
                 Options...> {
    __host__ __device__ constexpr CommOpt(ring_db_tag<ring_db>,
                                          thread_safe_tag<thread_safe>,
                                          check_sq_tag<check_sq>,
                                          Options...) { }
    /* inherit constructor */
    using CommOpt<ring_db_tag<ring_db>,
                  thread_safe_tag<thread_safe>,
                  check_sq_tag<check_sq>,
                  default_option_t<update_cq_tag>,
                  Options...
                 >::CommOpt;
  };

  /* Missing check_sq_tag,
   * else matches CommOpt<ring_db_tag, thread_safe_tag, check_sq_tag, Options...> */
  template <bool ring_db, bool thread_safe, typename... Options>
  struct CommOpt<ring_db_tag<ring_db>,
                 thread_safe_tag<thread_safe>,
                 Options...>
       : CommOpt<ring_db_tag<ring_db>,
                 thread_safe_tag<thread_safe>,
                 default_option_t<check_sq_tag>,
                 Options...> {
    __host__ __device__ constexpr CommOpt(ring_db_tag<ring_db>,
                                          thread_safe_tag<thread_safe>,
                                          Options...) { }
    /* inherit constructor */
    using CommOpt<ring_db_tag<ring_db>,
                  thread_safe_tag<thread_safe>,
                  default_option_t<check_sq_tag>,
                  Options...
                 >::CommOpt;
  };

  /* Missing thread_safe_tag,
   * else matches CommOpt<ring_db_tag, thread_safe_tag, Options...> */
  template <bool ring_db, typename... Options>
  struct CommOpt<ring_db_tag<ring_db>,
                 Options...>
       : CommOpt<ring_db_tag<ring_db>,
                 default_option_t<thread_safe_tag>,
                 Options...> {
    __host__ __device__ constexpr CommOpt(ring_db_tag<ring_db>,
                                          Options...) { }
    /* inherit constructor */
    using CommOpt<ring_db_tag<ring_db>,
                  default_option_t<thread_safe_tag>,
                  Options...
                 >::CommOpt;
  };

  /* Missing ring_db_tag,
   * else matches CommOpt<ring_db_tag, Options...> */
  template <typename... Options>
  struct CommOpt
       : CommOpt<default_option_t<ring_db_tag>,
                 Options...> {
    __host__ __device__ constexpr CommOpt(Options...) { }
    /* inherit constructor */
    using CommOpt<default_option_t<ring_db_tag>,
                  Options...
                 >::CommOpt;
  };

  /* ensure default CommOpt<> uses all the default options */
  static_assert(CommOpt<>::RingDB     == default_option_v<ring_db_tag>     &&
                CommOpt<>::ThreadSafe == default_option_v<thread_safe_tag> &&
                CommOpt<>::CheckSQ    == default_option_v<check_sq_tag>    &&
                CommOpt<>::UpdateCQ   == default_option_v<update_cq_tag>);

}  // namespace CommOption

/*
 * @brief Type alias helper for communication options.
 */
using CommOption::CommOpt;

/* bring CommOption::UpdateThread into scope */
using CommOption::UpdateThread;

/* constexpr variable templates to simplify usage */
template <auto V> constexpr inline auto RingDB     = CommOption::ring_db_tag<V>{};
template <auto V> constexpr inline auto ThreadSafe = CommOption::thread_safe_tag<V>{};
template <auto V> constexpr inline auto CheckSQ    = CommOption::check_sq_tag<V>{};
template <auto V> constexpr inline auto UpdateCQ   = CommOption::update_cq_tag<V>{};

}  // namespace rocshmem

#endif  // LIBRARY_SRC_COMM_OPTIONS_HPP_
