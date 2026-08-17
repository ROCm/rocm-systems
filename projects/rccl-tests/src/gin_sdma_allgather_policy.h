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

// Compiled default AllGather LSA<->SDMA crossover (bytes per rank). 32 KiB is the
// measured crossover on 8x MI355X (gfx950): LSA-direct xGMI stores win at/below
// it (plateau ~6.3 GB/s), Anvil-SDMA copy engines win above it (scale to
// ~390 GB/s). Used when neither the per-collective nor the global env override
// is set. Decoupled from the backend gin.put inline threshold
// (NCCL_GIN_ANVIL_SDMA_THRESHOLD / ctx->sdmaThreshold), which stays global.
static constexpr size_t kAllGatherSdmaThresholdDefault = 32768;

// Resolve the per-collective AllGather crossover threshold with precedence:
// per-collective env (NCCL_GIN_ANVIL_SDMA_THRESHOLD_ALLGATHER) > global env
// (NCCL_GIN_ANVIL_SDMA_THRESHOLD) > compiled default. "Set" means the env var was
// present and parsed (an explicit 0 is honored, forcing the all-SDMA tier). Pure
// so the precedence is unit-testable without the environment; the getenv/parse
// wrapper lives host-side in AllGatherResolveSdmaThreshold() (all_gather.cu).
GIN_SDMA_AG_HD inline size_t pickSdmaThreshold(bool perCollSet, unsigned long long perCollVal,
                                               bool globalSet, unsigned long long globalVal,
                                               size_t compiledDefault) {
  if (perCollSet) return (size_t)perCollVal;
  if (globalSet) return (size_t)globalVal;
  return compiledDefault;
}

// Sentinel meaning "CTA-count env var unset/empty/unparseable" (mirrors the
// ReduceScatter kThresholdUnset); allGatherCtas() falls back to the size-adaptive
// ladder when the env value is this sentinel.
static constexpr size_t kAllGatherCtasUnset = (size_t)-1;

// AllGather -D 3 size-adaptive CTA count (decoupled from -V, mirrors ReduceScatter
// and the broadcast/reduce rings). Keyed off the SAME tier predicate the kernel
// evaluates (chunkUsesLsaTier), so the CTA choice tracks the actual tier even when
// NCCL_GIN_ANVIL_SDMA_THRESHOLD[_ALLGATHER] moves the crossover:
//   * LSA-direct tier (chunk <= threshold): a grid-stride all-peers store that
//     scales with threads; ~16 CTAs peaks on 8x MI355X (128 KiB 448% of 1-CTA;
//     tiny sizes are latency-bound and CTA-indifferent).
//   * GIN-put / Anvil-SDMA tier (chunk > threshold): only nRanks threads issue the
//     puts (the copy engines move the bytes), so extra CTAs are pure barrier/signal
//     overhead -- FEW CTAs win (V=32 costs ~13% at 8 MiB, ~18% at 2 MiB vs V=4),
//     and 4 is a stable near-peak across 1 MiB-128 MiB.
// NCCL_GIN_ANVIL_AG_CTAS pins a fixed count for all sizes (diagnostic; must be <=
// the launched -V/deviceCtaCount-sized barrier pool -- see allGatherMaxCtas).
static constexpr int kAllGatherCtasLsa  = 16;   // chunk <= threshold (direct store)
static constexpr int kAllGatherCtasSdma = 4;    // chunk >  threshold (GIN puts)
GIN_SDMA_AG_HD inline int allGatherCtas(size_t chunkBytes_, size_t sdmaThreshold, size_t envCtas) {
  if (envCtas != kAllGatherCtasUnset && envCtas > 0)
    return (envCtas > 128) ? 128 : (int)envCtas;
  return chunkUsesLsaTier(chunkBytes_, sdmaThreshold) ? kAllGatherCtasLsa : kAllGatherCtasSdma;
}
GIN_SDMA_AG_HD inline int allGatherMaxCtas() {
  return (kAllGatherCtasLsa > kAllGatherCtasSdma) ? kAllGatherCtasLsa : kAllGatherCtasSdma;
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
