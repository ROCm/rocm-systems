/*************************************************************************
 * Copyright (c) Advanced Micro Devices, Inc. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

// Pure, host-testable policy helpers for the GIN Anvil-SDMA hybrid AllGather
// (all_gather_perf -D 3, GinHybridAllGatherKernel). These capture the message
// sizing, tier-selection, threshold-resolution and bandwidth math that the
// device kernel and rccl-tests host scaffolding both rely on, in one place that
// can be unit-tested on the host with no GPU (see gin_sdma_allgather_policy_test.cpp).
//
// The device kernel calls the same helpers so the LSA<->SDMA crossover predicate
// tested on the host is byte-for-byte the one that runs on the GPU. Under
// GIN_SDMA_HOST_ONLY the __host__/__device__ attributes drop to no-ops so the
// header compiles as plain C++ for the host unit test.

#ifndef GIN_SDMA_ALLGATHER_POLICY_H_
#define GIN_SDMA_ALLGATHER_POLICY_H_

#include <cstddef>
#include <cstdint>

#if defined(GIN_SDMA_HOST_ONLY)
#ifndef GIN_SDMA_AG_HD
#define GIN_SDMA_AG_HD /* host-only: no HIP attributes */
#endif
#else
#include <hip/hip_runtime.h>
#ifndef GIN_SDMA_AG_HD
#define GIN_SDMA_AG_HD __host__ __device__
#endif
#endif

namespace gin_sdma_allgather {

// Per-rank send element count used by all_gather_perf: floor(count/nranks)
// rounded down to a 16-byte-aligned element count (16/eltSize elements). eltSize
// must be a power-of-two divisor of 16 (1,2,4,8,16), matching wordSize() of the
// rccl-tests element types. Returns 0 if inputs are degenerate.
GIN_SDMA_AG_HD inline size_t chunkBaseCount(size_t count, size_t eltSize, int nranks) {
  if (nranks <= 0 || eltSize == 0 || eltSize > 16) return 0;
  const size_t perRank = count / (size_t)nranks;
  const size_t eltsPer16 = 16 / eltSize;   // 16,8,4,2,1
  const size_t mask = ~(eltsPer16 - 1);    // align down to a multiple of eltsPer16
  return perRank & mask;
}

// Per-rank send bytes for the AllGather (== the GinHybridAllGatherKernel chunk).
GIN_SDMA_AG_HD inline size_t chunkBytes(size_t perRankCount, size_t eltSize) {
  return perRankCount * eltSize;
}

// Tier predicate driving GinHybridAllGatherKernel: a per-rank chunk at or below
// the threshold takes the direct-LSA all-peers store tier; above it takes the
// all-peers GIN-put (Anvil-SDMA) tier. This is THE crossover the device kernel
// evaluates (chunkBytes <= sdmaThreshold).
GIN_SDMA_AG_HD inline bool chunkUsesLsaTier(size_t chunkBytes_, size_t sdmaThreshold) {
  return chunkBytes_ <= sdmaThreshold;
}

// Resolve the effective SDMA crossover threshold the kernel reads from the GIN
// Anvil-SDMA device context: use the context value only when the context is
// present and its layout magic validated; otherwise fall back to the compiled
// default. Mirrors AllGatherGetSdmaThreshold() so the fallback path is testable
// without a live device context.
GIN_SDMA_AG_HD inline size_t resolveSdmaThreshold(bool ctxPresent, bool magicValid,
                                                  size_t ctxThreshold, size_t defaultThreshold) {
  if (ctxPresent && magicValid) return ctxThreshold;
  return defaultThreshold;
}

// AllGather algorithm / bus bandwidth (GB/s) given the per-rank element count,
// element size, elapsed seconds and rank count. algBw counts every rank's
// contribution; busBw applies the AllGather (nranks-1)/nranks correction.
GIN_SDMA_AG_HD inline void bandwidthGBps(size_t perRankCount, int typeSize, double sec,
                                         int nranks, double* algBw, double* busBw) {
  const double baseBw = (double)(perRankCount * (size_t)typeSize * (size_t)nranks) / 1.0e9 / sec;
  if (algBw) *algBw = baseBw;
  if (busBw) {
    const double factor = ((double)(nranks - 1)) / ((double)nranks);
    *busBw = baseBw * factor;
  }
}

}  // namespace gin_sdma_allgather

#endif  // GIN_SDMA_ALLGATHER_POLICY_H_
