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

#ifndef LIBRARY_SRC_GDA_QUEUE_PAIR_MUX_HPP_
#define LIBRARY_SRC_GDA_QUEUE_PAIR_MUX_HPP_

#include <algorithm>
#include <tuple>
#include <type_traits>
#include <utility>

#include <hip/hip_runtime.h>

#include "rocshmem/rocshmem_config.h"  // NOLINT(build/include_subdir)

#include "constmem.hpp"
#include "gda_enums.hpp"
#include "log.hpp"
#include "util.hpp"

#include "gda/queue_pair/queue_pair_interface.hpp"
#include "gda/queue_pair/queue_pair_shmem.hpp"

#if defined(GDA_IONIC)
#include "gda/ionic/queue_pair_ionic.hpp"
#endif
#if defined(GDA_BNXT)
#include "gda/bnxt/queue_pair_bnxt.hpp"
#endif
#if defined(GDA_MLX5)
#include "gda/mlx5/queue_pair_mlx5.hpp"
#endif

namespace rocshmem {

class QueuePairMux;

template <> struct QueuePairTraits<QueuePairMux> {
  /**
   * @brief Mux OpCode enumeration
   */
  enum class OpCode {
    RDMA_WRITE,
    RDMA_READ,
    ATOMIC_CS,
    ATOMIC_FA,
  };

  static constexpr size_t InlineMax =
      std::min({
#if defined(GDA_IONIC)
                QueuePairTraits<QueuePairIONIC>::InlineMax,
#endif
#if defined(GDA_BNXT)
                QueuePairTraits<QueuePairBNXT>::InlineMax,
#endif
#if defined(GDA_MLX5)
                QueuePairTraits<QueuePairMLX5>::InlineMax,
#endif
                });

  static constexpr size_t InlineThreshold =
      std::min({
#if defined(GDA_IONIC)
                QueuePairTraits<QueuePairIONIC>::InlineThreshold,
#endif
#if defined(GDA_BNXT)
                QueuePairTraits<QueuePairBNXT>::InlineThreshold,
#endif
#if defined(GDA_MLX5)
                QueuePairTraits<QueuePairMLX5>::InlineThreshold,
#endif
                });
};

class QueuePairMux : public QueuePairSHMEM<QueuePairMux> {
private:
  union QueuePairUnion {
#if defined(GDA_IONIC)
    QueuePairIONIC ionic;
#endif
#if defined(GDA_BNXT)
    QueuePairBNXT  bnxt;
#endif
#if defined(GDA_MLX5)
    QueuePairMLX5  mlx5;
#endif
  };
  static_assert(std::is_aggregate_v<QueuePairUnion>,
                "QueuePairUnion must be an aggregate type.");
  static_assert(std::is_trivially_destructible_v<QueuePairUnion>,
                "QueuePairUnion must have a trivial destructor.");

  QueuePairUnion qp;

  /**
   * @brief Underlying GDA Provider.
   * Only used by __host__ code, __device__ code should use constmem.gda_provider instead.
   */
  static inline GDAProvider provider{GDAProvider::UNSET};

public:
#if defined(GDA_IONIC)
  __host__ QueuePairMux(QueuePairIONIC&& ionic)
    : qp{.ionic = std::move(ionic)} { provider = GDAProvider::IONIC; }
#endif
#if defined(GDA_BNXT)
  __host__ QueuePairMux(QueuePairBNXT&& bnxt)
    : qp{.bnxt = std::move(bnxt)}  { provider = GDAProvider::BNXT;  }
#endif
#if defined(GDA_MLX5)
  __host__ QueuePairMux(QueuePairMLX5&& mlx5)
    : qp{.mlx5 = std::move(mlx5)}  { provider = GDAProvider::MLX5;  }
#endif

  __host__ QueuePairMux(const QueuePairMux& other)            = delete;
  __host__ QueuePairMux& operator=(const QueuePairMux& other) = delete;
  __host__ QueuePairMux(QueuePairMux&& other) noexcept        = default;
  __host__ QueuePairMux& operator=(QueuePairMux&& other)      = default;
  __host__ ~QueuePairMux()                                    = default;

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
   *
   * Attempts to resolve the address in the following order of locations:
   *   1. The default symmetric heap.
   *   2. All user-registered local buffers.
   *   3. All user-registered symmetric buffers.
   *
   * @param[in] addr Symmetric address to resolve.
   *
   * @return {laddr, lkey} for addr or {0, std::numeric_limits<uint32_t>::max()} if not found.
   * Endianness of returned LKey value is ProviderEndianness.
   */
  __device__ __forceinline__
  std::tuple<uintptr_t, uint32_t> get_laddr_info(const void *addr, bool inlined = false) const;

  /**
   * @brief Resolve the remote (target) virtual address and RKey of a symmetric address.
   *
   * Attempts to resolve the address in the following order of locations:
   *   1. The default symmetric heap.
   *   2. All user-registered symmetric buffers.
   *
   * @param[in] addr Symmetric address to resolve.
   *
   * @return {raddr, rkey} for addr or {0, std::numeric_limits<uint32_t>::max()} if not found.
   * Endianness of returned RKey value is ProviderEndianness.
   */
  __device__ __forceinline__
  std::tuple<uintptr_t, uint32_t> get_raddr_info(const void *addr) const;

  /*
   * @brief Query whether data can be inlined into a WQE.
   *
   * @param[in] size Size of data carried by the WQE.
   * @tparam Op OpCode for this WQE.
   *
   * @return True if size bytes of data can be inlined into a WQE with OpCode Op, else false.
   */
  template <OpCode Op>
  static __host__ __device__ __forceinline__ constexpr bool can_inline(size_t size);

  /**
   * @brief Convert value to Endianness of selected provider, byteswapping if necessary.
   *
   * @param[in] val Value to convert.
   * @tparam T Type of val.
   *
   * @return QueuePairProvider::to_provider_endianness<T>(val)
   */
  template <typename T>
  static __host__ __device__ __forceinline__ T to_provider_endianness(T val);

private:
  static __host__   __forceinline__ GDAProvider get_provider() { return provider; }
  static __device__ __forceinline__ GDAProvider get_provider() { return constmem.gda_provider; }

  /*
   * @brief Convert from QueuePairMux::OpCode to the equivalent Provider::OpCode
   */
  template <OpCode Op, typename Provider>
  static __host__ __device__ constexpr typename Provider::OpCode provider_op();
};



template <QueuePairMux::OpCode Op, typename Provider>
__host__ __device__ constexpr typename Provider::OpCode QueuePairMux::provider_op() {
  if constexpr (Op == OpCode::RDMA_WRITE) {
    return Provider::OpCode::RDMA_WRITE;
  } else if constexpr (Op == OpCode::RDMA_READ) {
    return Provider::OpCode::RDMA_READ;
  } else if constexpr (Op == OpCode::ATOMIC_FA) {
    return Provider::OpCode::ATOMIC_FA;
  } else if constexpr (Op == OpCode::ATOMIC_CS) {
    return Provider::OpCode::ATOMIC_CS;
  } else {
    // might need some tricks if CWG2518 / P2593R1 isn't implemented by a supported compiler
    static_assert(false, "Invalid or unimplemented OpCode");
  }
}

template <QueuePairMux::OpCode Op, typename... Options>
__device__ __forceinline__ void QueuePairMux::post_wqe_rma(
    uintptr_t laddr, uint32_t lkey, uintptr_t raddr, uint32_t rkey, size_t size,
    const ActiveWFInfo& wf_info, PostOpt<Options...> options) {
  switch (get_provider()) {
#if defined(GDA_IONIC)
  case GDAProvider::IONIC:
    return qp.ionic.post_wqe_rma<provider_op<Op, QueuePairIONIC>()>(
                                 laddr, lkey, raddr, rkey, size, wf_info, options);
#endif
#if defined(GDA_BNXT)
  case GDAProvider::BNXT:
    return qp.bnxt.post_wqe_rma<provider_op<Op, QueuePairBNXT>()>(
                                laddr, lkey, raddr, rkey, size, wf_info, options);
#endif
#if defined(GDA_MLX5)
  case GDAProvider::MLX5:
    return qp.mlx5.post_wqe_rma<provider_op<Op, QueuePairMLX5>()>(
                                laddr, lkey, raddr, rkey, size, wf_info, options);
#endif
  default:
    assert(false /* invalid GDAProvider */);
    __builtin_unreachable();
  }
}

template <QueuePairMux::OpCode Op, typename... Options>
__device__ __forceinline__ void QueuePairMux::post_wqe_rma_single(
    uintptr_t laddr, uint32_t lkey, uintptr_t raddr, uint32_t rkey, size_t size,
    PostOpt<Options...> options) {
  switch (get_provider()) {
#if defined(GDA_IONIC)
  case GDAProvider::IONIC:
    return qp.ionic.post_wqe_rma_single<provider_op<Op, QueuePairIONIC>()>(
                                        laddr, lkey, raddr, rkey, size, options);
#endif
#if defined(GDA_BNXT)
  case GDAProvider::BNXT:
    return qp.bnxt.post_wqe_rma_single<provider_op<Op, QueuePairBNXT>()>(
                                       laddr, lkey, raddr, rkey, size, options);
#endif
#if defined(GDA_MLX5)
  case GDAProvider::MLX5:
    return qp.mlx5.post_wqe_rma_single<provider_op<Op, QueuePairMLX5>()>(
                                       laddr, lkey, raddr, rkey, size, options);
#endif
  default:
    assert(false /* invalid GDAProvider */);
    __builtin_unreachable();
  }
}

template <QueuePairMux::OpCode Op, AMOFetchType Fetch, typename... Options>
__device__ __forceinline__ QueuePairMux::amo_ret_t<Fetch> QueuePairMux::post_wqe_amo(
    uintptr_t raddr, uint32_t rkey, uint64_t swap_add, uint64_t compare,
    const ActiveWFInfo& wf_info, PostOpt<Options...> options) {
  static_assert(Fetch != AMOFetchType::NonBlocking, "non-blocking AMOs not yet implemented");
  switch (get_provider()) {
#if defined(GDA_IONIC)
  case GDAProvider::IONIC:
    return qp.ionic.post_wqe_amo<provider_op<Op, QueuePairIONIC>(), Fetch>(
                                 raddr, rkey, swap_add, compare, wf_info, options);
#endif
#if defined(GDA_BNXT)
  case GDAProvider::BNXT:
    return qp.bnxt.post_wqe_amo<provider_op<Op, QueuePairBNXT>(), Fetch>(
                                raddr, rkey, swap_add, compare, wf_info, options);
#endif
#if defined(GDA_MLX5)
  case GDAProvider::MLX5:
    return qp.mlx5.post_wqe_amo<provider_op<Op, QueuePairMLX5>(), Fetch>(
                                raddr, rkey, swap_add, compare, wf_info, options);
#endif
  default:
    assert(false /* invalid GDAProvider */);
    __builtin_unreachable();
  }
}

template <QueuePairMux::OpCode Op, AMOFetchType Fetch, typename... Options>
__device__ __forceinline__ QueuePairMux::amo_ret_t<Fetch> QueuePairMux::post_wqe_amo_single(
    uintptr_t raddr, uint32_t rkey, uint64_t swap_add, uint64_t compare,
    PostOpt<Options...> options) {
  static_assert(Fetch != AMOFetchType::NonBlocking, "non-blocking AMOs not yet implemented");
  switch (get_provider()) {
#if defined(GDA_IONIC)
  case GDAProvider::IONIC:
    return qp.ionic.post_wqe_amo_single<provider_op<Op, QueuePairIONIC>(), Fetch>(
                                        raddr, rkey, swap_add, compare, options);
#endif
#if defined(GDA_BNXT)
  case GDAProvider::BNXT:
    return qp.bnxt.post_wqe_amo_single<provider_op<Op, QueuePairBNXT>(), Fetch>(
                                       raddr, rkey, swap_add, compare, options);
#endif
#if defined(GDA_MLX5)
  case GDAProvider::MLX5:
    return qp.mlx5.post_wqe_amo_single<provider_op<Op, QueuePairMLX5>(), Fetch>(
                                       raddr, rkey, swap_add, compare, options);
#endif
  default:
    assert(false /* invalid GDAProvider */);
    __builtin_unreachable();
  }
}

__device__ __forceinline__ void QueuePairMux::quiet_single() {
  switch (get_provider()) {
#if defined(GDA_IONIC)
  case GDAProvider::IONIC:
    return qp.ionic.quiet_single();
#endif
#if defined(GDA_BNXT)
  case GDAProvider::BNXT:
    return qp.bnxt.quiet_single();
#endif
#if defined(GDA_MLX5)
  case GDAProvider::MLX5:
    return qp.mlx5.quiet_single();
#endif
  default:
    assert(false /* invalid GDAProvider */);
    __builtin_unreachable();
  }
}

__device__ __forceinline__ std::tuple<uintptr_t, uint32_t>
QueuePairMux::get_laddr_info(const void *addr, bool inlined) const {
  switch (get_provider()) {
#if defined(GDA_IONIC)
  case GDAProvider::IONIC:
    return qp.ionic.get_laddr_info(addr, inlined);
#endif
#if defined(GDA_BNXT)
  case GDAProvider::BNXT:
    return qp.bnxt.get_laddr_info(addr, inlined);
#endif
#if defined(GDA_MLX5)
  case GDAProvider::MLX5:
    return qp.mlx5.get_laddr_info(addr, inlined);
#endif
  default:
    assert(false /* invalid GDAProvider */);
    __builtin_unreachable();
  }
}

__device__ __forceinline__
std::tuple<uintptr_t, uint32_t> QueuePairMux::get_raddr_info(const void *addr) const {
  switch (get_provider()) {
#if defined(GDA_IONIC)
  case GDAProvider::IONIC:
    return qp.ionic.get_raddr_info(addr);
#endif
#if defined(GDA_BNXT)
  case GDAProvider::BNXT:
    return qp.bnxt.get_raddr_info(addr);
#endif
#if defined(GDA_MLX5)
  case GDAProvider::MLX5:
    return qp.mlx5.get_raddr_info(addr);
#endif
  default:
    assert(false /* invalid GDAProvider */);
    __builtin_unreachable();
  }
}

template <QueuePairMux::OpCode Op>
__host__ __device__ __forceinline__ constexpr bool QueuePairMux::can_inline(size_t size) {
  /*
   * We don't know which GDA provider will be selected at runtime,
   * so we define QueuePairTraits<QueuePairMux>::InlineThreshold
   * as the minimum InlineThreshold of the available GDA providers.
   *
   * This is a sufficient lower bound that can be used at compile-time
   * to e.g. allocate a buffer that can be inlined
   * regardless of which provider is selected at runtime.
   *
   * At runtime, however, we can do better:
   * once we know what provider was selected,
   * we can query QueuePairProvider::can_inline directly.
   *
   * To enable this behavior, QueuePairMux::can_inline must detect
   * whether it was called from within a manifestly constant-evaluated context.
   * This capability is not available in standard C++17.
   * In C++20, this can be done using std::is_constant_evaluated.
   * In C++23, this can be done using consteval-if.
   *
   * In C++17, we can backport the C++20 behavior using Clang's __builtin_is_constant_evaluated().
   * Once we enable C++20/C++23, move to standards-compliant behavior.
   */
  if (__builtin_is_constant_evaluated()) {
    // In a constant-evaluated context use the minimum InlineThreshold.
    if constexpr (Op == OpCode::RDMA_WRITE) {
      return size <= Traits::InlineThreshold;
    } else {
      return false;
    }
  } else {
    // At runtime, dispatch to the selected GDA provider.
    switch (get_provider()) {
#if defined(GDA_IONIC)
    case GDAProvider::IONIC:
      return QueuePairIONIC::can_inline<provider_op<Op, QueuePairIONIC>()>(size);
#endif
#if defined(GDA_BNXT)
    case GDAProvider::BNXT:
      return QueuePairBNXT::can_inline<provider_op<Op, QueuePairBNXT>()>(size);
#endif
#if defined(GDA_MLX5)
    case GDAProvider::MLX5:
      return QueuePairMLX5::can_inline<provider_op<Op, QueuePairMLX5>()>(size);
#endif
    default:
#ifdef __HIP_DEVICE_COMPILE__
    assert(false /* invalid GDAProvider */);
    __builtin_unreachable();
#else
    static_assert(std::is_same_v<std::underlying_type_t<GDAProvider>, int>);
    LOG_ERROR_ABORT("Invalid GDAProvider (%d)", static_cast<int>(get_provider()));
#endif
    }
  }
}

template <typename T>
__host__ __device__ __forceinline__ T QueuePairMux::to_provider_endianness(T val) {
  switch (get_provider()) {
#if defined(GDA_IONIC)
  case GDAProvider::IONIC:
    return QueuePairIONIC::to_provider_endianness<T>(val);
#endif
#if defined(GDA_BNXT)
  case GDAProvider::BNXT:
    return QueuePairBNXT::to_provider_endianness<T>(val);
#endif
#if defined(GDA_MLX5)
  case GDAProvider::MLX5:
    return QueuePairMLX5::to_provider_endianness<T>(val);
#endif
  default:
#ifdef __HIP_DEVICE_COMPILE__
    assert(false /* invalid GDAProvider */);
    __builtin_unreachable();
#else
    static_assert(std::is_same_v<std::underlying_type_t<GDAProvider>, int>);
    LOG_ERROR_ABORT("Invalid GDAProvider (%d)", static_cast<int>(get_provider()));
#endif
  }
}

}  // namespace rocshmem

#endif  // LIBRARY_SRC_GDA_QUEUE_PAIR_MUX_HPP_
