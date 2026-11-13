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
#include "rccl_float8.h"
using bf16 = __hip_bfloat16;
#else
#include <cuda.h>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
using bf16 = __nv_bfloat16;
using bf162 = __nv_bfloat162;
#endif

#include "algorithms/dda/device/dda_kernel_params.h"
#include "nccl_device/rccl_ptr.h"

namespace dda::common {

// fp8 element tags. rccl_float8 / rccl_bfloat8 are different types in the host
// pass and the gfx942 device pass (FNUZ vs OCP), so a kernel templated on them
// would be registered under a name the gfx942 code object does not contain. The
// kernels are instantiated on these instead, and vecElementAdd resolves the
// device's encoding.
struct fp8e4m3 {
  uint8_t bits;
};
struct fp8e5m2 {
  uint8_t bits;
};
static_assert(sizeof(fp8e4m3) == 1 && sizeof(fp8e5m2) == 1, "fp8 tags must be one byte");

template <typename T>
inline constexpr bool is_fp8_v = std::is_same<T, fp8e4m3>::value || std::is_same<T, fp8e5m2>::value;

template <typename T>
inline constexpr bool is_supported_type_v = (std::is_same<T, float>::value || std::is_same<T, half>::value ||
                                             std::is_same<T, int8_t>::value || std::is_same<T, bf16>::value
#if defined(__HIP_PLATFORM_AMD__) || defined(__HIP_PLATFORM_HCC__)
                                             || std::is_same<T, __hip_bfloat16>::value ||
                                             std::is_same<T, fp8e4m3>::value || std::is_same<T, fp8e5m2>::value
#endif
);

// Adds the eight fp8 lanes of an 8-byte word with the pk8 add the core reduce
// kernels use (hadd8 / hadd8_b), so DDA rounds as they do. Only gfx1250 admits
// fp8 to DDA; other arches compile these kernels but never launch them.
template <typename T>
static inline __device__ uint64_t vecElementAddFp8x8(uint64_t a, uint64_t b) {
  static_assert(is_fp8_v<T>, "dda: vecElementAddFp8x8 takes an fp8 tag");
#if defined(RCCL_FP8_HAS_PK8)
  const auto x = __builtin_bit_cast(fp8x8_storage_t, a);
  const auto y = __builtin_bit_cast(fp8x8_storage_t, b);
  if constexpr (std::is_same<T, fp8e4m3>::value) {
    return __builtin_bit_cast(uint64_t, hadd8(x, y));
  } else {
    return __builtin_bit_cast(uint64_t, hadd8_b(x, y));
  }
#else
  __builtin_trap();
  return 0;
#endif
}

template <typename T>
static inline __device__ uint32_t vecElementAdd(const uint32_t& a, const uint32_t& b) {
  static_assert(is_supported_type_v<T>, "dda: unsupported element type");
  static_assert(!is_fp8_v<T>, "dda: fp8 adds eight lanes at a time; use vecElementAddFp8x8");
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
  if constexpr (is_fp8_v<T>) {
    const uint64_t lo = vecElementAddFp8x8<T>(((uint64_t)a.y << 32) | a.x, ((uint64_t)b.y << 32) | b.x);
    const uint64_t hi = vecElementAddFp8x8<T>(((uint64_t)a.w << 32) | a.z, ((uint64_t)b.w << 32) | b.z);
    res.x = (uint32_t)lo;
    res.y = (uint32_t)(lo >> 32);
    res.z = (uint32_t)hi;
    res.w = (uint32_t)(hi >> 32);
  } else {
    res.x = vecElementAdd<T>(a.x, b.x);
    res.y = vecElementAdd<T>(a.y, b.y);
    res.z = vecElementAdd<T>(a.z, b.z);
    res.w = vecElementAdd<T>(a.w, b.w);
  }
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
    size_t srcIdx = (pattern == 2) ? idx : (idx + static_cast<size_t>(selfRank) * idxEnd);
    size_t destIdx = (pattern == 1) ? (idx + static_cast<size_t>(selfRank) * idxEnd) : idx;

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
      size_t destIdx = idx + static_cast<size_t>(srcRank) * idxEnd;
      size_t srcIdx;
      if (enable_offset) {
        srcIdx = destIdx;
      } else {
        srcIdx = idx;
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

// copyFromSrcToDest loads user buffers as uint4, so both pointers have to be
// 16-byte aligned. Shared by the IPC eligibility checks.
inline bool ddaUserBuffers16ByteAligned(const void* sendbuff, const void* recvbuff) {
  return (reinterpret_cast<uintptr_t>(sendbuff) % 16) == 0 &&
         (reinterpret_cast<uintptr_t>(recvbuff) % 16) == 0;
}

// #12928 copied ReduceScatter IPC shards inside the kernel up to this size.
// That path is reverted: the host always pre-copies into scratch. The constant
// and InKernel enumerator stay so the kernel template still compiles.
constexpr size_t kDdaRsIpcKernelCopyMaxBytes = 1ULL * 1024 * 1024;

enum class DdaRsIpcStaging {
  InKernel,    // kernel copies each shard from sendbuff
  HostPreCopy, // host cudaMemcpyAsync before launch
};

inline DdaRsIpcStaging ddaReduceScatterIpcStaging(size_t recvcount, int typeSize) {
  (void)recvcount;
  (void)typeSize;
  return DdaRsIpcStaging::HostPreCopy;
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

// DDA block widths come in 64-thread units: one wave64 or two wave32 waves, so no block ends in a partial wave.
constexpr uint32_t kDdaThreadGranule = 64;

// Blocks under a grid cap: `capBlocks` when set and lower, else `blocks`.
inline size_t ddaCapBlocks(size_t blocks, uint32_t capBlocks) {
  return (capBlocks > 0 && capBlocks < blocks) ? capBlocks : blocks;
}

// Block width under a grid cap: `capThreads` rounded down to whole granules, floored at one; kernels stride the grid.
inline uint32_t ddaCapThreads(uint32_t threads, uint32_t capThreads) {
  if (capThreads == 0 || threads <= capThreads) {
    return threads;
  }
  return std::max(capThreads / kDdaThreadGranule * kDdaThreadGranule, kDdaThreadGranule);
}

inline std::pair<dim3, dim3> getGridAndBlockDims(size_t count, int typeSize, size_t maxBlocks) {
  constexpr uint32_t kThreadsPerBlock = 512;

  const uint32_t elementsPerThread = 16 / typeSize; // we do 16 Byte load in kernel

  const uint32_t elementsPerWarp = elementsPerThread * kDdaThreadGranule;

  dim3 threads(0, 1, 1);
  dim3 blocks(0, 1, 1);
  if (ddaAlltoAllSingleBlockGrid(count, typeSize)) {
    threads.x = divRoundUp(count, elementsPerWarp) * kDdaThreadGranule;
    blocks.x = 1;
  } else {
    auto warpsRequired = divRoundUp(count, elementsPerWarp);
    blocks.x = calcBlockCount(divRoundUp(count, elementsPerThread), kThreadsPerBlock, maxBlocks);
    auto warpsPerBlock = divRoundUp(warpsRequired, blocks.x);
    auto threadsPerBlock = std::min<uint32_t>(kThreadsPerBlock, warpsPerBlock * kDdaThreadGranule);
    threads.x = threadsPerBlock;
  }

  return std::make_pair(blocks, threads);
}

// Per-rank staging slot size in scratch bytes, fixed at compile time so the
// double-buffered layout is identical on every rank and call. The eligibility
// checks derive their per-message caps from it.
// Footprint = 2 banks * nRanks * (kDdaLLMaxBytes)
// example: 128 MiB at 16 MiB for 4 ranks
constexpr size_t kDdaLLMaxBytes = (size_t)(32) * 1024 * 1024;

// Scratch is double buffered under a single epoch counter, so a bank is half of
// it, floored to 128B so LL128 lines in bank 1 stay 128B-aligned. Every LL/LL128
// tier that derives its slot stride from the bank rather than a fixed constant
// puts bank 1 at the same byte offset, which is what stops one tier landing on
// flag words another is still polling.
constexpr size_t ddaBankSize(size_t bytes) {
  return (bytes / 2) / 128 * 128;
}

// Per-rank slot carved out of a bank, counted in whatever unit the protocol
// addresses the wire with -- 16B packets for LL, 8B words for LL128. `div` folds
// the unit size, the rank count, and the number of staging areas the tier puts
// in one bank (two for the two-shot tiers, which publish and then write back).
// `align` is the granule the slot has to be a whole number of, so a slot always
// begins on a boundary the protocol's accesses require.
//
// Kernel and eligibility both call this, so the size the host admits is exactly
// the geometry the kernel addresses.
constexpr size_t ddaLLSlotPkts(size_t bankSize, size_t div, size_t align) {
  const size_t slotUnits = bankSize / div;
  return (slotUnits / align) * align;
}

// Grid of peer columns that stays within totalBlockCap: past the cap a column serves
// several peers (DdaPeerColumns). Each column gets blocksWanted blocks, at most its
// share of the cap and at least one.
struct DdaPeerGrid {
  unsigned cols;
  unsigned blocksPerCol;
};
constexpr DdaPeerGrid ddaPeerGrid(int nPeers, size_t blocksWanted, size_t totalBlockCap) {
  const size_t cap = totalBlockCap < 1 ? 1 : totalBlockCap;
  const size_t peers = nPeers < 1 ? 1 : (size_t)nPeers;
  const size_t cols = peers < cap ? peers : cap;
  const size_t share = cap / cols;
  const size_t perCol = blocksWanted < share ? blocksWanted : share;
  return {(unsigned)cols, perCol < 1 ? 1u : (unsigned)perCol};
}

// Peers this block's column serves: blockIdx.x, blockIdx.x + gridDim.x, ... below nPeers.
// Issue every send before the first poll, or two columns can each wait on the other's send.
struct DdaPeerColumns {
  struct Iter {
    int x;
    int step;
    __device__ int operator*() const { return x; }
    __device__ Iter& operator++() {
      x += step;
      return *this;
    }
    __device__ bool operator!=(const Iter& end) const { return x < end.x; }
  };
  int nPeers;
  __device__ Iter begin() const { return {(int)blockIdx.x, (int)gridDim.x}; }
  __device__ Iter end() const { return {nPeers, 0}; }
};

// Peers staged in one pass. A runtime-sized array, or a runtime index into
// one, is placed in scratch. 8 covers rank counts 4 and 8 in a single pass.
constexpr int kDdaLLPeerBatch = 8;

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

// Folds one packet's 8B payload (d0, d1) into the accumulator (acc0, acc1).
template <typename T>
__device__ __forceinline__ void ddaLLAddPayload(uint32_t& acc0, uint32_t& acc1, uint32_t d0, uint32_t d1) {
  if constexpr (is_fp8_v<T>) {
    const uint64_t sum = vecElementAddFp8x8<T>(((uint64_t)acc1 << 32) | acc0, ((uint64_t)d1 << 32) | d0);
    acc0 = (uint32_t)sum;
    acc1 = (uint32_t)(sum >> 32);
  } else {
    acc0 = vecElementAdd<T>(acc0, d0);
    acc1 = vecElementAdd<T>(acc1, d1);
  }
}

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

// Poll one 16B LL line until both flags match. #11245's abort exit is reverted.
__device__ __forceinline__ bool ddaLLWaitLineFlags(const volatile LLPacket16* src, uint32_t flag,
                                                   DdaKernelParams params, uint32_t& d0, uint32_t& f0, uint32_t& d1,
                                                   uint32_t& f1) {
  (void)params;
  const uint32_t* line = reinterpret_cast<const uint32_t*>(const_cast<const LLPacket16*>(src));
  do {
    ddaLLLoadLineB128(line, d0, f0, d1, f1);
  } while (!(f0 == flag && f1 == flag));
  return true;
}

// Lines one sweep of ddaLLWaitLineFlagsBatched polls at most: a 4- or 8-rank comm's peers fit one sweep.
constexpr int kDdaLLRecvBatch = 8;

// Sweep size for n lines known at compile time (0 < n <= kDdaLLRecvBatch), else kDdaLLRecvBatch.
constexpr int ddaLLRecvBatch(int n) {
  return n > 0 && n <= kDdaLLRecvBatch ? n : kDdaLLRecvBatch;
}

// Polled lines from which a waiting lane spins on one stale line between sweeps: with many loads per sweep the re-sweep
// crowds the CU memory path its remote stores also use; with few the extra sweep costs more than it saves.
constexpr int kDdaLLSpinOneMinLines = 5;

// Payload bytes one LL packet carries; the send buffer is read one packet per thread.
constexpr int kDdaLLPayloadBytes = 8;
static_assert(sizeof(LLPacket16) == 2 * kDdaLLPayloadBytes, "an LL line is payload plus one flag per 4B word");

// Bytes one global_prefetch pulls into L2: a 128B cache line.
constexpr int kDdaLLPrefetchLineBytes = 128;

// Consecutive threads whose packets share a prefetch line; one per group issues it, for any wave size.
constexpr int kDdaLLPrefetchThreadsPerLine = kDdaLLPrefetchLineBytes / kDdaLLPayloadBytes;
static_assert((kDdaLLPrefetchThreadsPerLine & (kDdaLLPrefetchThreadsPerLine - 1)) == 0, "thread group is a power of 2");

// L2-prefetches send packets first + k * step (k < count, below limit) before the LL epoch barrier; gfx1250 only.
__device__ __forceinline__ void ddaLLPrefetchSend(const void* sendbuff, size_t first, size_t step, int count,
                                                  size_t limit) {
#if (defined(__gfx1250__) || defined(__gfx1250_strict__)) && __has_builtin(__builtin_amdgcn_global_prefetch)
  if (threadIdx.x % kDdaLLPrefetchThreadsPerLine != 0) {
    return;
  }
  const char* base = reinterpret_cast<const char*>(sendbuff);
#pragma unroll 1
  for (int k = 0; k < count && first + (size_t)k * step < limit; ++k) {
    const char* pkt = base + (first + (size_t)k * step) * kDdaLLPayloadBytes;
    // th 1 also walks the page tables, so the first post-barrier read of the line does not.
    __builtin_amdgcn_global_prefetch((const __attribute__((address_space(1))) void*)pkt, 0x11);
  }
#else
  (void)sendbuff;
  (void)first;
  (void)step;
  (void)count;
  (void)limit;
#endif
}

// Spins on the first stale line alone until both its flags land, so a waiting lane keeps one load in flight, not kN.
template <int kN>
__device__ __forceinline__ bool ddaLLSpinOneStale(const uint32_t* const (&line)[kN], const bool (&stale)[kN],
                                                  uint32_t flag, DdaKernelParams params, int& spins) {
  (void)params;
  (void)spins;
  const uint32_t* wait = line[0];
  bool found = false;
#pragma unroll
  for (int i = 0; i < kN; ++i) {
    wait = stale[i] && !found ? line[i] : wait;
    found |= stale[i];
  }
  if (!found) {
    return true;
  }
  while (true) {
    uint32_t d0, f0, d1, f1;
    ddaLLLoadLineB128(wait, d0, f0, d1, f1);
    if (f0 == flag && f1 == flag) {
      return true;
    }
  }
}

// Waits until both flags of every line selected by mask (per lane) match. Each sweep issues kN loads before testing
// any flag, so they share one wait (a load under a per-line branch would wait on its own); a slot outside mask reloads
// the first selected line, so only selected lines are read. Sweeps repeat until every selected line is fresh; with
// kDdaLLSpinOneMinLines or more selected lines a waiting lane spins on one stale line between
// sweeps. A single selected line is polled alone: there is no other load to share its wait, and a sweep would issue kN
// loads of it. d0/d1 of slots outside mask are overwritten.
template <int kN>
__device__ __forceinline__ bool ddaLLWaitLineFlagsBatched(const uint32_t* const (&line)[kN], uint32_t mask,
                                                          uint32_t flag, DdaKernelParams params, uint32_t (&d0)[kN],
                                                          uint32_t (&d1)[kN]) {
  static_assert(kN > 0 && kN <= 32, "one mask bit per line");
  if (mask == 0) {
    return true;
  }
  const uint32_t* first = line[0];
#pragma unroll
  for (int i = kN - 1; i >= 0; --i) {
    first = ((mask >> i) & 1u) != 0 ? line[i] : first;
  }
  if (__builtin_popcount(mask) == 1) {
    uint32_t w0, f0, w1, f1;
    const auto* only = reinterpret_cast<const volatile LLPacket16*>(first);
    const bool ok = ddaLLWaitLineFlags(only, flag, params, w0, f0, w1, f1);
#pragma unroll
    for (int i = 0; i < kN; ++i) {
      d0[i] = w0;
      d1[i] = w1;
    }
    return ok;
  }
  const uint32_t* src[kN];
#pragma unroll
  for (int i = 0; i < kN; ++i) {
    src[i] = ((mask >> i) & 1u) != 0 ? line[i] : first;
  }
  const bool spinOne = __builtin_popcount(mask) >= kDdaLLSpinOneMinLines;
  int spins = 0;
  while (true) {
    uint32_t f0[kN], f1[kN];
#if RCCL_HAVE_GLOBAL_DWORDX4_BUILTINS
    asm volatile("" ::: "memory");
    v4u got[kN];
#pragma unroll
    for (int i = 0; i < kN; ++i) {
      got[i] = __builtin_amdgcn_global_load_b128((v4u_gptr)src[i], RCCL_SYSTEM_SYNCSCOPE);
    }
    // Keep every load above this point so the flag tests cannot sink one between them.
    asm volatile("" ::: "memory");
#pragma unroll
    for (int i = 0; i < kN; ++i) {
      d0[i] = got[i][0];
      f0[i] = got[i][1];
      d1[i] = got[i][2];
      f1[i] = got[i][3];
    }
#else
#pragma unroll
    for (int i = 0; i < kN; ++i) {
      ddaLLLoadLineB128(src[i], d0[i], f0[i], d1[i], f1[i]);
    }
    // Keep every load above this point so none sinks past the flag tests or out of the loop.
    asm volatile("" ::: "memory");
#endif
    bool stale = false;
    bool staleLine[kN];
#pragma unroll
    for (int i = 0; i < kN; ++i) {
      staleLine[i] = ((mask >> i) & 1u) != 0 && ((f0[i] != flag) | (f1[i] != flag));
      stale |= staleLine[i];
    }
    if (!stale) {
      return true;
    }
    if (spinOne) {
      ddaLLSpinOneStale(src, staleLine, flag, params, spins);
    }
  }
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

// Copies `bytes` (a multiple of 16) from src to dst in 16-byte vectors, thread `gtid` of
// `gstride` threads taking every gstride-th vector, so the whole grid shares the copy.
__device__ __forceinline__ void ddaGridCopy16B(const void* src, void* dst, size_t bytes, size_t gtid, size_t gstride) {
  v4u_gptr s4 = (v4u_gptr)src;
  v4u_gptr d4 = (v4u_gptr)dst;
  const size_t nVec = bytes >> 4;
  for (size_t i = gtid; i < nVec; i += gstride) d4[i] = s4[i];
}

} // namespace dda::common
