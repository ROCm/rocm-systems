/*************************************************************************
 * Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * Derived from Meta torchcomms comms/common/algorithms/CollCommon.cuh.
 * C++17: replaced C++20 concepts with constexpr type trait + static_assert.
 * See LICENSE.txt for license information.
 ************************************************************************/

#pragma once

#include <cstddef>
#include <type_traits>

#if defined(__HIP_PLATFORM_AMD__) || defined(__HIP_PLATFORM_HCC__)
#include <hip/hip_bfloat16.h>
#include <hip/hip_fp16.h>
#include <hip/hip_runtime.h>
#include <hip/amd_detail/amd_hip_bf16.h>
using bf16 = hip_bfloat16;
#else
#include <cuda.h>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
using bf16 = __nv_bfloat16;
using bf162 = __nv_bfloat162;
#endif

namespace meta::comms {

template <typename T>
inline constexpr bool is_supported_type_v =
    (std::is_same<T, float>::value || std::is_same<T, half>::value ||
     std::is_same<T, bf16>::value
#if defined(__HIP_PLATFORM_AMD__) || defined(__HIP_PLATFORM_HCC__)
     || std::is_same<T, __hip_bfloat16>::value
#endif
    );

template <typename T>
static inline __device__ uint32_t
vecElementAdd(const uint32_t& a, const uint32_t& b) {
  static_assert(is_supported_type_v<T>, "dda: unsupported element type");
  if constexpr (std::is_same<T, float>::value) {
    const float* x = reinterpret_cast<const float*>(&a);
    const float* y = reinterpret_cast<const float*>(&b);
    float z = x[0] + y[0];
    return (reinterpret_cast<uint32_t*>(&z))[0];
  } else if constexpr (std::is_same<T, half>::value) {
    const __half* x = reinterpret_cast<const __half*>(&a);
    const __half* y = reinterpret_cast<const __half*>(&b);
    __half2 p = __halves2half2(x[0], x[1]);
    __half2 q = __halves2half2(y[0], y[1]);
    __half2 z = __hadd2(p, q);
    return (reinterpret_cast<uint32_t*>(&z))[0];
  } else if constexpr (std::is_same<T, bf16>::value) {
    const bf16* x = reinterpret_cast<const bf16*>(&a);
    const bf16* y = reinterpret_cast<const bf16*>(&b);
#if defined(__HIP_PLATFORM_AMD__) || defined(__HIP_PLATFORM_HCC__)
    uint32_t out = 0;
    bf16* z = reinterpret_cast<bf16*>(&out);
    z[0] = x[0] + y[0];
    z[1] = x[1] + y[1];
    return out;
#else
    bf162 p = {x[0], x[1]};
    bf162 q = {y[0], y[1]};
    bf162 z = __hadd2(p, q);
    return (reinterpret_cast<uint32_t*>(&z))[0];
#endif
  }
  return 0;
}

template <typename T>
static inline __device__ uint4 vecElementAdd(const uint4& a, const uint4& b) {
  static_assert(is_supported_type_v<T>, "dda: unsupported element type");
  uint4 res{0, 0, 0, 0};
  res.x = vecElementAdd<T>(a.x, b.x);
  res.y = vecElementAdd<T>(a.y, b.y);
  res.z = vecElementAdd<T>(a.z, b.z);
  res.w = vecElementAdd<T>(a.w, b.w);
  return res;
}

template <typename T>
static inline __device__ void copyFromSrcToDest(
    const T* __restrict__ srcbuff,
    T* __restrict__ destbuff,
    const size_t idxStart,
    const size_t idxEnd,
    const size_t idxStride) {
  static_assert(is_supported_type_v<T>, "dda: unsupported element type");
  for (size_t idx = idxStart; idx < idxEnd; idx += idxStride) {
    *reinterpret_cast<uint4*>(&destbuff[idx]) =
        reinterpret_cast<const uint4*>(&srcbuff[idx])[0];
  }
}

template <typename T, int NRANKS, bool hasAcc>
static inline __device__ void reduceScatter(
    T* const* __restrict__ ipcbuffs,
    T* __restrict__ destbuff,
    const T* __restrict__ acc,
    int selfRank,
    const size_t idxStart,
    const size_t idxEnd,
    const size_t idxStride,
    int pattern) {
  static_assert(is_supported_type_v<T>, "dda: unsupported element type");
  for (size_t idx = idxStart; idx < idxEnd; idx += idxStride) {
    // pattern = 2 performs reduce (one-shot)
    // pattern = 1 performs reduce-scatter (two-shot)	  
    size_t srcIdx = (pattern == 2) ? idx : (idx + selfRank * idxEnd);
    size_t destIdx = (pattern == 1) ? (idx + selfRank * idxEnd) : idx;

    uint4 sum{0, 0, 0, 0};
    if constexpr (hasAcc) {
      sum = reinterpret_cast<const uint4*>(&acc[srcIdx])[0];
    }

    uint4 srcVals[2];
    *reinterpret_cast<uint4*>(&srcVals[0]) =
        reinterpret_cast<const uint4*>(&ipcbuffs[0][srcIdx])[0];
#pragma unroll NRANKS - 1
    for (int r = 0; r < NRANKS - 1; ++r) {
      *reinterpret_cast<uint4*>(&srcVals[(r + 1) & 1]) =
          reinterpret_cast<const uint4*>(
              &ipcbuffs[(r + 1) % NRANKS][srcIdx])[0];
      sum = vecElementAdd<T>(sum, srcVals[r & 1]);
    }
    sum = vecElementAdd<T>(sum, srcVals[(NRANKS - 1) & 1]);

    *reinterpret_cast<uint4*>(&destbuff[destIdx]) =
        *reinterpret_cast<const uint4*>(&sum);
  }
}

template <typename T, int NRANKS>
static inline __device__ void allGather(
    T* const* __restrict__ ipcbuffs,
    T* __restrict__ destbuff,
    int selfRank,
    const size_t idxStart,
    const size_t idxEnd,
    const size_t idxStride,
    bool enable_offset) {
  static_assert(is_supported_type_v<T>, "dda: unsupported element type");
  for (size_t idx = idxStart; idx < idxEnd; idx += idxStride) {
#pragma unroll NRANKS
    for (int r = 0; r < NRANKS; ++r) {
      int srcRank = (selfRank + r) % NRANKS;
      int destIdx = idx + srcRank * idxEnd;
      int srcIdx;
      if (enable_offset) {
        srcIdx = destIdx;
      } else {
        srcIdx = static_cast<int>(idx);
      }
      *reinterpret_cast<uint4*>(&destbuff[destIdx]) =
          reinterpret_cast<const uint4*>(&ipcbuffs[srcRank][srcIdx])[0];
    }
  }
}

} // namespace meta::comms
