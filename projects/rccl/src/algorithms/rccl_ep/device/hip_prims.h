/*************************************************************************
 * Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * Wave-level primitives for AMDGCN: lane and wave geometry, wave-wide scans
 * and reductions, and acquire/release accessors.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

#ifndef RCCL_EP_HIP_PRIMS_H_
#define RCCL_EP_HIP_PRIMS_H_

#include <cstdint>

#include <hip/hip_runtime.h>

namespace rccl_ep {

// ---------------------------------------------------------------------------
// Wave geometry.
//
// AMDGCN CDNA is wave64. Every mask is 64-bit. Do not introduce a literal 32 or
// 0xffffffff anywhere in this module: divide-by-32 index arithmetic fails
// *silently* on wave64, whereas 32-bit shuffle masks at least fail at compile
// time.
// ---------------------------------------------------------------------------
constexpr int kWarpSize = 64;
using wave_mask_t = uint64_t;
constexpr wave_mask_t kFullWaveMask = ~wave_mask_t{0};

__device__ __forceinline__ int get_lane_idx() {
  return __lane_id();
}

__device__ __forceinline__ int get_warp_idx() {
  return static_cast<int>(threadIdx.x) / kWarpSize;
}

__device__ __forceinline__ int get_num_warps() {
  return static_cast<int>(blockDim.x) / kWarpSize;
}

template <typename T>
__device__ __forceinline__ T reduce_add(T v) {
#pragma unroll
  for (int off = kWarpSize / 2; off > 0; off >>= 1) v += __shfl_xor(v, off, kWarpSize);
  return v;
}

// Inclusive prefix sum across the wave (Hillis-Steele).
template <typename T>
__device__ __forceinline__ T warp_inclusive_sum(T v) {
  const int lane = get_lane_idx();
#pragma unroll
  for (int off = 1; off < kWarpSize; off <<= 1) {
    T n = __shfl_up(v, off, kWarpSize);
    if (lane >= off) v += n;
  }
  return v;
}

template <typename T>
__device__ __forceinline__ T warp_exclusive_sum(T v) {
  return warp_inclusive_sum(v) - v;
}

template <typename T>
__device__ __forceinline__ T ld_acquire_sys(const T* p) {
  return __hip_atomic_load(p, __ATOMIC_ACQUIRE, __HIP_MEMORY_SCOPE_SYSTEM);
}

template <typename T>
__device__ __forceinline__ void st_release_sys(T* p, T v) {
  __hip_atomic_store(p, v, __ATOMIC_RELEASE, __HIP_MEMORY_SCOPE_SYSTEM);
}

}  // namespace rccl_ep

#endif  // RCCL_EP_HIP_PRIMS_H_
