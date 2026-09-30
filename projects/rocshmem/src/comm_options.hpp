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
 * Each option is a tag describing an app-level property/semantic of a
 * communication call (not a backend implementation detail). Backends consume
 * the tags they understand and ignore the rest, so the same CommOpt can be
 * threaded uniformly through every API communication call.
 *
 * These options were historically the GDA queue-pair "PostOpt" options; the
 * tags are named for the app-level semantics they express:
 *   - initiate     : launch the transfer now, or stage it for a later
 *                    initiating call (batched submission).
 *                    (GDA: ring the SQ doorbell.)
 *   - concurrent   : the issuing context may be accessed by multiple threads
 *                    concurrently and needs internal serialization; set false
 *                    when the app guarantees serialized access.
 *                    (GDA: take the SQ lock.)
 *   - flow_control : the library enforces capacity/backpressure; set false when
 *                    the app guarantees outstanding operations stay within
 *                    capacity. (GDA: poll the CQ for free SQ slots.)
 *   - completion   : completion-tracking granularity: notify per-op (Every),
 *                    per-group (Last), or not at all (None).
 *                    (GDA: set MLX5_WQE_CTRL_CQ_UPDATE.)
 */

#include <type_traits>

#include <hip/hip_runtime.h>

// ActiveWFInfo (used by the completion-granularity helper). It only depends on
// util.hpp, so including it here does not couple CommOpt to any GDA transport.
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

  /*
   * @brief Completion-tracking granularity.
   */
  enum class CompletionScope {
    Every,  // every operation is individually tracked to completion
    Last,   // only the last operation in a group is tracked
    None,   // fire-and-forget; no completion tracking
  };

  /*
   * @brief Option: whether this call initiates (launches) the transfer now, or
   * stages it to be launched together with a later initiating call.
   */
  template <bool initiate>
  struct initiate_tag : constant_t<initiate> { };

  template <> struct default_option<initiate_tag> {
    /* default: DO initiate the transfer now */
    using type = initiate_tag<true>;
  };

  /*
   * @brief Option: whether the issuing context may be accessed concurrently by
   * multiple threads/waves (requiring internal serialization).
   */
  template <bool concurrent>
  struct concurrent_tag : constant_t<concurrent> { };

  template <> struct default_option<concurrent_tag> {
    /* default: DO assume concurrent access (serialize internally) */
    using type = concurrent_tag<true>;
  };

  /*
   * @brief Option: whether the library enforces flow control
   * (capacity/backpressure) for this operation.
   */
  template <bool flow_control>
  struct flow_control_tag : constant_t<flow_control> { };

  template <> struct default_option<flow_control_tag> {
    /* default: DO enforce flow control */
    using type = flow_control_tag<true>;
  };

  /*
   * @brief Option: completion-tracking granularity for this operation.
   */
  template <CompletionScope completion>
  struct completion_tag : constant_t<completion> { };

  template <> struct default_option<completion_tag> {
    /* default: track every operation to completion */
    using type = completion_tag<CompletionScope::Every>;
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
  template <bool initiate, bool concurrent, bool flow_control, CompletionScope completion>
  struct CommOpt<initiate_tag<initiate>,
                 concurrent_tag<concurrent>,
                 flow_control_tag<flow_control>,
                 completion_tag<completion>> {
    /* explicitly-defaulted default constructor */
    __host__ __device__ constexpr CommOpt() = default;
    /* constructor for type deduction from tags */
    __host__ __device__ constexpr CommOpt(initiate_tag<initiate>,
                                          concurrent_tag<concurrent>,
                                          flow_control_tag<flow_control>,
                                          completion_tag<completion>) { }

    /* static constexpr data members to simplify option access */
    static constexpr auto Initiate    = initiate;
    static constexpr auto Concurrent  = concurrent;
    static constexpr auto FlowControl = flow_control;
    static constexpr auto Completion  = completion;

    static __device__ constexpr inline bool signal_completion(const ActiveWFInfo& wf_info) {
      if constexpr (Completion == CompletionScope::Every) {
        // every operation is tracked to completion
        return true;
      } else if constexpr (Completion == CompletionScope::Last) {
        // only the last operation in each group is tracked
        return wf_info.is_pe_group_last;
      } else {
        // no completion tracking
        return false;
      }
    }

    static __device__ constexpr inline bool signal_completion_single() {
      if constexpr (Completion == CompletionScope::Every) {
        // every operation is tracked to completion
        return true;
      } else if constexpr (Completion == CompletionScope::Last) {
        // singleton groups, so all threads are "last": track all
        return true;
      } else {
        // no completion tracking
        return false;
      }
    }
  };

  /* Extraneous parameters,
   * else matches CommOpt<initiate_tag, concurrent_tag, flow_control_tag, completion_tag> */
  template <bool initiate, bool concurrent, bool flow_control, CompletionScope completion,
            typename... Options>
  struct CommOpt<initiate_tag<initiate>,
                 concurrent_tag<concurrent>,
                 flow_control_tag<flow_control>,
                 completion_tag<completion>,
                 Options...> {
    static_assert(sizeof...(Options) == 0, "Too many or invalid options");
  };

  /* Missing completion_tag,
   * else matches CommOpt<initiate_tag, concurrent_tag, flow_control_tag, completion_tag, Options...> */
  template <bool initiate, bool concurrent, bool flow_control, typename... Options>
  struct CommOpt<initiate_tag<initiate>,
                 concurrent_tag<concurrent>,
                 flow_control_tag<flow_control>,
                 Options...>
       : CommOpt<initiate_tag<initiate>,
                 concurrent_tag<concurrent>,
                 flow_control_tag<flow_control>,
                 default_option_t<completion_tag>,
                 Options...> {
    __host__ __device__ constexpr CommOpt(initiate_tag<initiate>,
                                          concurrent_tag<concurrent>,
                                          flow_control_tag<flow_control>,
                                          Options...) { }
    /* inherit constructor */
    using CommOpt<initiate_tag<initiate>,
                  concurrent_tag<concurrent>,
                  flow_control_tag<flow_control>,
                  default_option_t<completion_tag>,
                  Options...
                 >::CommOpt;
  };

  /* Missing flow_control_tag,
   * else matches CommOpt<initiate_tag, concurrent_tag, flow_control_tag, Options...> */
  template <bool initiate, bool concurrent, typename... Options>
  struct CommOpt<initiate_tag<initiate>,
                 concurrent_tag<concurrent>,
                 Options...>
       : CommOpt<initiate_tag<initiate>,
                 concurrent_tag<concurrent>,
                 default_option_t<flow_control_tag>,
                 Options...> {
    __host__ __device__ constexpr CommOpt(initiate_tag<initiate>,
                                          concurrent_tag<concurrent>,
                                          Options...) { }
    /* inherit constructor */
    using CommOpt<initiate_tag<initiate>,
                  concurrent_tag<concurrent>,
                  default_option_t<flow_control_tag>,
                  Options...
                 >::CommOpt;
  };

  /* Missing concurrent_tag,
   * else matches CommOpt<initiate_tag, concurrent_tag, Options...> */
  template <bool initiate, typename... Options>
  struct CommOpt<initiate_tag<initiate>,
                 Options...>
       : CommOpt<initiate_tag<initiate>,
                 default_option_t<concurrent_tag>,
                 Options...> {
    __host__ __device__ constexpr CommOpt(initiate_tag<initiate>,
                                          Options...) { }
    /* inherit constructor */
    using CommOpt<initiate_tag<initiate>,
                  default_option_t<concurrent_tag>,
                  Options...
                 >::CommOpt;
  };

  /* Missing initiate_tag,
   * else matches CommOpt<initiate_tag, Options...> */
  template <typename... Options>
  struct CommOpt
       : CommOpt<default_option_t<initiate_tag>,
                 Options...> {
    __host__ __device__ constexpr CommOpt(Options...) { }
    /* inherit constructor */
    using CommOpt<default_option_t<initiate_tag>,
                  Options...
                 >::CommOpt;
  };

  /* ensure default CommOpt<> uses all the default options */
  static_assert(CommOpt<>::Initiate    == default_option_v<initiate_tag>     &&
                CommOpt<>::Concurrent  == default_option_v<concurrent_tag>   &&
                CommOpt<>::FlowControl == default_option_v<flow_control_tag> &&
                CommOpt<>::Completion  == default_option_v<completion_tag>);

}  // namespace CommOption

/*
 * @brief Type alias helper for communication options.
 */
using CommOption::CommOpt;

/* bring CommOption::CompletionScope into scope */
using CommOption::CompletionScope;

/* constexpr variable templates to simplify usage */
template <auto V> constexpr inline auto Initiate    = CommOption::initiate_tag<V>{};
template <auto V> constexpr inline auto Concurrent  = CommOption::concurrent_tag<V>{};
template <auto V> constexpr inline auto FlowControl = CommOption::flow_control_tag<V>{};
template <auto V> constexpr inline auto Completion  = CommOption::completion_tag<V>{};

}  // namespace rocshmem

#endif  // LIBRARY_SRC_COMM_OPTIONS_HPP_
