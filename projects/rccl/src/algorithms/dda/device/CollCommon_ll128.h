/*************************************************************************
 * Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
 *
 * Shared device core for the LL128-protocol DDA collectives (gfx1250 fabric).
 *
 * Line format (matches prims_ll128 NCCL_LL128_LINEELEMS/DATAELEMS):
 *   128B "line" = 16 x uint64. words 0..14 carry 120B of payload, word 15 is
 *   the epoch flag. Efficiency 120/128 = 93.75% (vs 50% for the 16B LL line).
 *
 * Ordering (warp-cooperative, UNFENCED): 16 lanes own one line. Lanes 0..14
 * store their payload word, lane 15 stores the flag LAST. In SIMT lockstep the
 * payload-store instruction is issued (program order) strictly before the
 * flag-store instruction for the whole wave, and gfx1250 preserves remote-write
 * ordering per lane's program order (validated by the ll128 ordering probe and
 * the warp AllGather/AllReduce microbenchmarks, both zero-corruption). So a
 * reader that observes the flag also observes the payload -- no release/acquire
 * fence required. Double buffering (bank = flag & 1) + a monotonic epoch avoids
 * a reader accepting a stale line from a prior epoch.
 *
 * HIP-graph safety: the flag/bank are derived on-device from the per-block
 * epoch array (comm->ddaLLEpochDev), never passed from the host, so nothing is
 * baked into a captured graph. See ddaLLEpochBegin / ddaLLEpochEnd below and
 * commit 6820b6f4a9 ("Make DDA LL kernels HIP graph-capture safe").
 *
 * See LICENSE.txt for license information.
 ************************************************************************/

#pragma once

#include <cstddef>
#include <cstdint>

#if defined(__HIP_PLATFORM_AMD__) || defined(__HIP_PLATFORM_HCC__)
#include <hip/hip_runtime.h>
#else
#include <cuda_runtime.h>
#endif

#include "algorithms/dda/device/CollCommon.h"
#include "nccl_device/rccl_ptr.h"

namespace dda::common {

// ---- LL128 line geometry ----
constexpr int kDdaLL128LineElems = 16;                    // 16 x uint64 = 128B
constexpr int kDdaLL128DataElems = 15;                    // 15 payload words
constexpr int kDdaLL128FlagElem = 15;                    // word 15 == flag
constexpr int kDdaLL128Lanes = kDdaLL128LineElems;    // 16 lanes/line
constexpr size_t kDdaLL128LineBytes = 128;
constexpr size_t kDdaLL128DataBytes = (size_t)kDdaLL128DataElems * 8; // 120

// 128B line: 15 payload words + 1 trailing flag word.
struct LLLine128 {
  uint64_t w[kDdaLL128LineElems];
};
static_assert(sizeof(LLLine128) == 128, "LLLine128 must be exactly 128 bytes");

// Number of 128B lines needed to carry nWords 8B words (ceil).
__host__ __device__ __forceinline__ size_t ddaLL128NumLines(size_t nWords) {
  return (nWords + (size_t)kDdaLL128DataElems - 1) / (size_t)kDdaLL128DataElems;
}

// ---- 8B system-scope, non-tearing store/load (one word == one lane) ----
// RELAXED order: the flag-last ordering that makes the unfenced protocol correct
// comes from gfx1250 preserving per-thread program order of system-scope stores,
// not from an atomic fence. Matches the validated microbenchmark primitives.
__device__ __forceinline__ void ddaLL128StoreWord(uint64_t* p, uint64_t v) {
  __hip_atomic_store((u64_gptr)p, v, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_SYSTEM);
}
__device__ __forceinline__ uint64_t ddaLL128LoadWord(const uint64_t* p) {
  return __hip_atomic_load((u64_gptr) const_cast<uint64_t*>(p), __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_SYSTEM);
}

// Poll the LL128 flag word until it matches, or comm abortFlag is set.
// Returns false if the wait was aborted so the kernel can exit.
__device__ __forceinline__ bool ddaLL128WaitFlag(const uint64_t* flagAddr, uint64_t flag, const uint32_t* abortFlag) {
  int spins = 0;
  while (ddaLL128LoadWord(flagAddr) != flag) {
    if (ddaAbortSpinTick(abortFlag, spins)) return false;
  }
  return true;
}

// Element-wise add of the T-elements packed into two 8B payload words. An 8B
// word holds 2 x fp32 or 4 x fp16/bf16; each 4B half is folded with the shared
// vecElementAdd<T> (which handles the per-type packing), then recombined.
template <typename T>
__device__ __forceinline__ uint64_t ddaLL128AddWord(uint64_t a, uint64_t b) {
  const uint32_t lo = vecElementAdd<T>((uint32_t)a, (uint32_t)b);
  const uint32_t hi = vecElementAdd<T>((uint32_t)(a >> 32), (uint32_t)(b >> 32));
  return ((uint64_t)hi << 32) | (uint64_t)lo;
}

// ---- HIP-graph-safe per-block epoch (shared by all LL/LL128 kernels) ----
// Entry: tid 0 reads this block's epoch cell (all cells hold the same value),
// derives the next flag on-device, and broadcasts it via the caller-provided
// shared slot. flag 0 is the "cleared scratch" sentinel and is skipped so bank
// parity (flag & 1) is preserved. Returns the flag for this launch.
__device__ __forceinline__ uint32_t ddaLLEpochBegin(const uint32_t* __restrict__ epochDev, int flatBlockId,
                                                    uint32_t& s_flag) {
  if (threadIdx.x == 0) {
    uint32_t f = epochDev[flatBlockId] + 1u;
    if (f == 0u) f = 2u; // skip 0 sentinel; keep bank parity
    s_flag = f;
  }
  __syncthreads();
  return s_flag;
}

// Exit: tid 0 advances every epoch cell this block strides over to `flag`, so
// all cells stay in lock-step even when a later launch uses a different block
// count. `total` == gridDim.x * gridDim.y.
__device__ __forceinline__ void ddaLLEpochEnd(uint32_t* __restrict__ epochDev, int flatBlockId, int total, int epochLen,
                                              uint32_t flag) {
  if (threadIdx.x == 0) {
    for (int e = flatBlockId; e < epochLen; e += total) {
      epochDev[e] = flag;
    }
  }
}

} // namespace dda::common
