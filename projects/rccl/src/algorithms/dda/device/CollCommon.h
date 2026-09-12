/*************************************************************************
 * Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * Derived from Meta torchcomms comms/common/algorithms/CollCommon.cuh.
 * C++17: replaced C++20 concepts with constexpr type trait + static_assert.
 * See LICENSE.txt for license information.
 ************************************************************************/

#pragma once

#include <algorithm>
#include <cstdint>
#include <utility>
#include <cstddef>
#include <type_traits>

#if defined(__HIP_PLATFORM_AMD__) || defined(__HIP_PLATFORM_HCC__)
// Do NOT include <hip/hip_bfloat16.h> (the old header).  It defines
// _HIP_BFLOAT16_H_ which conflicts with the RCCL bf16 workaround in
// device.h on ROCm 6.x, causing a hard #error regardless of include order.
// amd_hip_bf16.h already provides __hip_bfloat16 without touching those macros.
#include <hip/hip_fp16.h>
#include <hip/hip_runtime.h>
#include <hip/amd_detail/amd_hip_bf16.h>
using bf16 = __hip_bfloat16;
#else
#include <cuda.h>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
using bf16 = __nv_bfloat16;
using bf162 = __nv_bfloat162;
#endif

#include "algorithms/dda/device/dda_abort.h"
#include "nccl_device/rccl_ptr.h"

namespace dda::common {

template <typename T>
inline constexpr bool is_supported_type_v = (std::is_same<T, float>::value || std::is_same<T, half>::value ||
                                             std::is_same<T, int8_t>::value || std::is_same<T, bf16>::value
#if defined(__HIP_PLATFORM_AMD__) || defined(__HIP_PLATFORM_HCC__)
                                             || std::is_same<T, __hip_bfloat16>::value
#endif
);

template <typename T>
static inline __device__ uint32_t vecElementAdd(const uint32_t& a, const uint32_t& b) {
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
static inline __device__ void copyFromSrcToDest(const T* __restrict__ srcbuff, T* __restrict__ destbuff,
                                                const size_t idxStart, const size_t idxEnd, const size_t idxStride) {
  static_assert(is_supported_type_v<T>, "dda: unsupported element type");
  for (size_t idx = idxStart; idx < idxEnd; idx += idxStride) {
    *reinterpret_cast<uint4*>(&destbuff[idx]) = reinterpret_cast<const uint4*>(&srcbuff[idx])[0];
  }
}

// Unified reduce-scatter helper for both the IPC and fabric paths.
//   - NRANKS_CT > 0  : compile-time clique size; nRanks folds to a constant and
//                      the peer loop is fully unrolled (the IPC / fabric 4,8
//                      fast paths).
//   - NRANKS_CT == 0 : runtime fallback; nRanks comes from nRanksRuntime and the
//                      peer loop is partially unrolled 8-wide, covering any clique up to
//                      kDdaMaxNranks.
template <typename T, int NRANKS_CT, bool hasAcc>
static inline __device__ void reduceScatter(T* const* __restrict__ ipcbuffs, T* __restrict__ destbuff,
                                            const T* __restrict__ acc, int selfRank, int nRanksRuntime,
                                            const size_t idxStart, const size_t idxEnd, const size_t idxStride,
                                            int pattern) {
  static_assert(is_supported_type_v<T>, "dda: unsupported element type");
  const int nRanks = (NRANKS_CT > 0) ? NRANKS_CT : nRanksRuntime;
  constexpr int kUnroll = (NRANKS_CT > 0) ? NRANKS_CT : 8;
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
    *reinterpret_cast<uint4*>(&srcVals[0]) = reinterpret_cast<const uint4*>(&ipcbuffs[0][srcIdx])[0];
#pragma unroll kUnroll
    for (int r = 0; r < nRanks - 1; ++r) {
      *reinterpret_cast<uint4*>(&srcVals[(r + 1) & 1]) = reinterpret_cast<const uint4*>(&ipcbuffs[r + 1][srcIdx])[0];
      sum = vecElementAdd<T>(sum, srcVals[r & 1]);
    }
    sum = vecElementAdd<T>(sum, srcVals[(nRanks - 1) & 1]);

    *reinterpret_cast<uint4*>(&destbuff[destIdx]) = *reinterpret_cast<const uint4*>(&sum);
  }
}

// Unified all-gather helper for both the IPC and fabric paths. See reduceScatter
// above for the NRANKS_CT / nRanksRuntime / unroll semantics.
template <typename T, int NRANKS_CT>
static inline __device__ void allGather(T* const* __restrict__ ipcbuffs, T* __restrict__ destbuff, int selfRank,
                                        int nRanksRuntime, const size_t idxStart, const size_t idxEnd,
                                        const size_t idxStride, bool enable_offset) {
  static_assert(is_supported_type_v<T>, "dda: unsupported element type");
  const int nRanks = (NRANKS_CT > 0) ? NRANKS_CT : nRanksRuntime;
  constexpr int kUnroll = (NRANKS_CT > 0) ? NRANKS_CT : 8;
  for (size_t idx = idxStart; idx < idxEnd; idx += idxStride) {
#pragma unroll kUnroll
    for (int r = 0; r < nRanks; ++r) {
      int srcRank = (selfRank + r) % nRanks;
      int destIdx = idx + srcRank * idxEnd;
      int srcIdx;
      if (enable_offset) {
        srcIdx = destIdx;
      } else {
        srcIdx = static_cast<int>(idx);
      }
      *reinterpret_cast<uint4*>(&destbuff[destIdx]) = reinterpret_cast<const uint4*>(&ipcbuffs[srcRank][srcIdx])[0];
    }
  }
}

inline uint32_t divRoundUp(size_t a, size_t b) {
  uint32_t y = static_cast<uint32_t>((a + b - 1) / b);
  if (y == 0) {
    y = 1;
  }
  return y;
}

// True if per-rank data fits in one CUDA block (512 threads, 16-byte loads per thread).
// Same threshold as getGridAndBlockDims(). DDA AlltoAll uses it to choose in-kernel
// staging copy for small messages vs pre-kernel cudaMemcpyAsync for larger ones.
inline bool ddaAlltoAllSingleBlockGrid(size_t count, int typeSize) {
  constexpr uint32_t kThreadsPerBlock = 512;
  const uint32_t elementsPerThread = 16 / typeSize;
  return count < elementsPerThread * kThreadsPerBlock;
}

constexpr uint32_t calcBlockCount(size_t numThreads, size_t threadsPerBlock, size_t maxBlocks) {
  const auto uNumThreads = static_cast<uint64_t>(numThreads);
  const auto uThreadsPerBlock = static_cast<uint64_t>(threadsPerBlock);
  // Overflow safe variant of (a + b - 1) / b
  const uint64_t blocks = uNumThreads / uThreadsPerBlock + (uNumThreads % uThreadsPerBlock != 0);
  uint32_t y = static_cast<uint32_t>(std::min(blocks, maxBlocks));
  if (y == 0) {
    y = 1;
  }
  return y;
}

inline std::pair<dim3, dim3> getGridAndBlockDims(size_t count, int typeSize, size_t maxBlocks) {
  constexpr uint32_t kThreadsPerWarp = 64;
  constexpr uint32_t kThreadsPerBlock = 512;

  const uint32_t elementsPerThread = 16 / typeSize; // we do 16 Byte load in kernel

  const uint32_t elementsPerWarp = elementsPerThread * kThreadsPerWarp;

  dim3 threads(0, 1, 1);
  dim3 blocks(0, 1, 1);
  if (ddaAlltoAllSingleBlockGrid(count, typeSize)) {
    threads.x = divRoundUp(count, elementsPerWarp) * kThreadsPerWarp;
    blocks.x = 1;
  } else {
    auto warpsRequired = divRoundUp(count, elementsPerWarp);
    blocks.x = calcBlockCount(divRoundUp(count, elementsPerThread), kThreadsPerBlock, maxBlocks);
    auto warpsPerBlock = divRoundUp(warpsRequired, blocks.x);
    auto threadsPerBlock = std::min<uint32_t>(kThreadsPerBlock, warpsPerBlock * kThreadsPerWarp);
    threads.x = threadsPerBlock;
  }

  return std::make_pair(blocks, threads);
}

// Per-rank staging slot size in scratch bytes, fixed at compile time so the
// double-buffered layout is identical on every rank and call. The eligibility
// checks derive their per-message caps from it.
// Footprint = 2 banks * nRanks * (kDdaLLMaxBytes)
// example: 128 MiB at 16 MiB for 4 ranks
constexpr size_t kDdaLLMaxBytes = (size_t)(16) * 1024 * 1024;      // 16M

// 16-byte LL line: two (4B data, 4B flag) pairs carrying 8B of payload.
union LLPacket16 {
  struct {
    uint32_t data0;
    uint32_t flag0;
    uint32_t data1;
    uint32_t flag1;
  };
  uint4 raw;
};
static_assert(sizeof(LLPacket16) == 16, "LLPacket16 must be exactly 16 bytes");

__device__ __forceinline__ void ddaLLStoreLineB128(uint32_t* dst, uint32_t a0, uint32_t a1, uint32_t a2, uint32_t a3) {
#if RCCL_HAVE_GLOBAL_DWORDX4_BUILTINS
  union {
    v4u v;
    uint32_t w[4];
  } u;
  u.w[0] = a0;
  u.w[1] = a1;
  u.w[2] = a2;
  u.w[3] = a3;
  __builtin_amdgcn_global_store_b128((v4u_gptr)dst, u.v, RCCL_SYSTEM_SYNCSCOPE);
#else
  __builtin_nontemporal_store(a0, (u32_gptr)dst + 0);
  __builtin_nontemporal_store(a1, (u32_gptr)dst + 1);
  __builtin_nontemporal_store(a2, (u32_gptr)dst + 2);
  __builtin_nontemporal_store(a3, (u32_gptr)dst + 3);
#endif
  asm volatile("" ::: "memory");
}

__device__ __forceinline__ void ddaLLLoadLineB128(const uint32_t* src, uint32_t& o0, uint32_t& o1, uint32_t& o2,
                                                  uint32_t& o3) {
  asm volatile("" ::: "memory");
#if RCCL_HAVE_GLOBAL_DWORDX4_BUILTINS
  union {
    v4u v;
    uint32_t w[4];
  } u;
  u.v = __builtin_amdgcn_global_load_b128((v4u_gptr)src, RCCL_SYSTEM_SYNCSCOPE);
  o0 = u.w[0];
  o1 = u.w[1];
  o2 = u.w[2];
  o3 = u.w[3];
#else
  o0 = __builtin_nontemporal_load((u32_gptr)src + 0);
  o1 = __builtin_nontemporal_load((u32_gptr)src + 1);
  o2 = __builtin_nontemporal_load((u32_gptr)src + 2);
  o3 = __builtin_nontemporal_load((u32_gptr)src + 3);
#endif
}

// Poll one 16B LL line until both flags match, or comm abortFlag is set.
// Returns false if the wait was aborted so the kernel can exit.
__device__ __forceinline__ bool ddaLLWaitLineFlags(const uint32_t* src, uint32_t flag, const uint32_t* abortFlag,
                                                   uint32_t& d0, uint32_t& f0, uint32_t& d1, uint32_t& f1) {
  int spins = 0;
  do {
    ddaLLLoadLineB128(src, d0, f0, d1, f1);
    if (f0 == flag && f1 == flag) return true;
    if (ddaAbortSpinTick(abortFlag, spins)) return false;
  } while (true);
}

__device__ __forceinline__ uint32_t ddaGetLLEpochInc(const uint32_t* __restrict__ epochDev, int flatBlockId, uint32_t inc) {
  uint32_t flag = epochDev[flatBlockId] + inc;
  if (flag == 0u) flag = 2u; // skip 0 sentinel; keep bank parity
  __syncthreads();
  return flag;
}

__device__ __forceinline__ void ddaSetLLEpoch(uint32_t* __restrict__ epochDev, int epochLen,
                                              int flatBlockId, int total, uint32_t flag) {
  // Bump every cell this block owns (cell e belongs to block e % total), split
  // across the block's threads. The barrier is what makes the flag read above
  // safe: thread 0 rewrites cell flatBlockId here, so every thread must have
  // read it first. Cells at or above `total` are never read this launch.
  for (int e = flatBlockId + (int)threadIdx.x * total;
       e < epochLen;
       e += total * (int)blockDim.x) {
    epochDev[e] = flag;
  }
}

} // namespace dda::common
