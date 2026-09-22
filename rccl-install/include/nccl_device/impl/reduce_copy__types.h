#include "hip/hip_runtime.h"
/*************************************************************************
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * See LICENSE.txt for more license information
 *************************************************************************/

#ifndef _NCCL_DEVICE_REDUCE_COPY__TYPES_H_
#define _NCCL_DEVICE_REDUCE_COPY__TYPES_H_

#include "vector__types.h"
#include "../utility.h"
#include "../coop.h"

#ifdef __HIPCC__
#include <cassert>
#include <type_traits>

namespace nccl {
namespace utility {

// Reduction Operators

template <typename T>
struct OpSum {
  using EltType = T;
  NCCL_DEVICE_INLINE T operator()(const T& a, const T& b) const {
    return a + b;
  }
};

// Helper trait to create accumulator reduction operator from RedOp
// Maps RedOp (e.g., OpSum<T>) to accumulator reduction operator (e.g., OpSum<AccEltType>)
template <typename RedOp, typename AccEltType>
struct AccRedOp {
  // Default: keep RedOp as-is (non-templated operators).
  using Type = RedOp;
};

// Rebind RedOp<T> to RedOp<AccEltType> when possible.
template <template <typename> typename Red, typename T, typename AccEltType>
struct AccRedOp<Red<T>, AccEltType> {
  using Type = Red<AccEltType>;
};

// Cooperation Level Helpers for compile-time stride resolution
template <typename Coop>
struct CoopStride {
  // Default: runtime determined (use sentinel 0 to indicate runtime)
  static constexpr int value = 0;
};

#ifdef __HIPCC__
// Specialization for warp: always 32
template <>
struct CoopStride<ncclCoopWarp> {
  static constexpr int value = 32;
};

// Specialization for CTA: use 32 for warp coalescing
template <>
struct CoopStride<ncclCoopCta> {
  static constexpr int value = 32;
};

// Specialization for thread: use 1
template <>
struct CoopStride<ncclCoopThread> {
  static constexpr int value = 1;
};
#endif

#if defined(__CUDA_FP8_TYPES_EXIST__)
// Specialization for FP8 types - convert to half, add, convert back
template <>
struct OpSum<__hip_fp8_e4m3_fnuz> {
  using EltType = __hip_fp8_e4m3_fnuz;
  NCCL_DEVICE_INLINE __hip_fp8_e4m3_fnuz operator()(const __hip_fp8_e4m3_fnuz& a, const __hip_fp8_e4m3_fnuz& b) const {
#if __CUDA_ARCH__ >= 800
      // Use native half addition on architectures that support it
    return __hip_fp8_e4m3_fnuz(__hadd(__half(a), __half(b)));
#else
      // Fallback: convert to float, add, convert back
    return __hip_fp8_e4m3_fnuz(float(a) + float(b));
#endif
  }
};

template <>
struct OpSum<__hip_fp8_e5m2_fnuz> {
  using EltType = __hip_fp8_e5m2_fnuz;
  NCCL_DEVICE_INLINE __hip_fp8_e5m2_fnuz operator()(const __hip_fp8_e5m2_fnuz& a, const __hip_fp8_e5m2_fnuz& b) const {
#if __CUDA_ARCH__ >= 800
      // Use native half addition on architectures that support it
    return __hip_fp8_e5m2_fnuz(__hadd(__half(a), __half(b)));
#else
      // Fallback: convert to float, add, convert back
    return __hip_fp8_e5m2_fnuz(float(a) + float(b));
#endif
  }
};
#endif

} // namespace utility
} // namespace nccl
#endif // __HIPCC__

#endif // _NCCL_DEVICE_REDUCE_COPY__TYPES_H_
