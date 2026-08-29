/*************************************************************************
 * Copyright (c) Advanced Micro Devices, Inc. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

// Pure, host-testable policy helpers for the GIN Anvil-SDMA ReduceScatter
// (reduce_scatter_perf -D 3, GinReduceScatterKernel). These capture the per-rank
// output-slice sizing, the size-adaptive CTA selection, the devComm barrier/signal
// requirement shape and the bandwidth math that reduce_scatter.cu relies on, in one
// place that can be unit-tested on the host with no GPU (see
// gin_sdma_reducescatter_policy_test.cpp). Companion to gin_sdma_allgather_policy.h.
//
// The device kernel calls the same helpers so the CTA schedule and requirement
// shape tested on the host are byte-for-byte the ones used on the GPU. Under
// GIN_SDMA_HOST_ONLY the __host__/__device__ attributes drop to no-ops so the
// header compiles as plain C++ for the host unit test.
//
// NOTE on the tier: the shipped GinReduceScatterKernel is SINGLE-TIER (a direct
// LSA read-reduce for all sizes; the load SCHEDULE adapts by total bytes, but the
// algorithm never changes). The RSTier / threshold helpers below are retained for
// documentation and forward-compatibility (a future put-partials large tier, see
// reducescatter-gin-sdma-phase2.md) and to give NCCL_GIN_ANVIL_SDMA_THRESHOLD_-
// REDUCESCATTER a defined meaning; the current kernel does not branch on them.

#ifndef GIN_SDMA_REDUCESCATTER_POLICY_H_
#define GIN_SDMA_REDUCESCATTER_POLICY_H_

#include <cstddef>
#include <cstdint>

#if defined(GIN_SDMA_HOST_ONLY)
#ifndef GIN_SDMA_RS_HD
#define GIN_SDMA_RS_HD /* host-only: no HIP attributes */
#endif
#else
#include <hip/hip_runtime.h>
#ifndef GIN_SDMA_RS_HD
#define GIN_SDMA_RS_HD __host__ __device__
#endif
#endif

namespace gin_sdma_reducescatter {

// Sentinel meaning "env var unset/empty/unparseable" (mirrors AllGather's
// pickSdmaThreshold "not set" and the backend's TEST_SDMA_THRESHOLD_UNSET).
static constexpr size_t kThresholdUnset = (size_t)-1;

// Per-rank output-slice element count used by reduce_scatter_perf:
// floor(count/nranks) rounded down to a 16-byte-aligned element count
// (16/eltSize elements). eltSize must be a power-of-two divisor of 16
// (1,2,4,8,16), matching wordSize() of the rccl-tests element types. Mirrors
// ReduceScatterGetCollByteCount. Returns 0 if inputs are degenerate.
GIN_SDMA_RS_HD inline size_t sliceBaseCount(size_t count, size_t eltSize, int nranks) {
  if (nranks <= 0 || eltSize == 0 || eltSize > 16) return 0;
  const size_t perRank = count / (size_t)nranks;
  const size_t eltsPer16 = 16 / eltSize;   // 16,8,4,2,1
  const size_t mask = ~(eltsPer16 - 1);    // align down to a multiple of eltsPer16
  return perRank & mask;
}

// Per-rank output-slice bytes.
GIN_SDMA_RS_HD inline size_t sliceBytes(size_t perRankCount, size_t eltSize) {
  return perRankCount * eltSize;
}

// Compiled default ReduceScatter LSA<->GIN crossover (bytes per rank slice).
// 256 KiB/rank is the provisional cutover from the Phase-2 design plan. RETAINED
// FOR DOCUMENTATION ONLY: the shipped kernel is single-tier LSA read-reduce and
// does not branch on this value (see the file-top NOTE and
// reducescatter-gin-sdma-phase2.md). Used by pickSdmaThreshold as the fallback.
static constexpr size_t kReduceScatterSdmaThresholdDefault = 262144;  // 256 KiB/rank slice

enum class RSTier { LSA, Gin };

// Reserved tier predicate (parity with the AllGather / movement collectives): a
// per-rank slice at/below the threshold would take the LSA read-reduce, above it
// the (future) put-partials GIN tier. The current kernel is single-tier LSA, so
// this is used only by the host unit test / forward-compat callers.
GIN_SDMA_RS_HD inline RSTier reduceScatterKernelTier(size_t sliceBytes_, size_t sdmaThreshold) {
  return (sliceBytes_ <= sdmaThreshold) ? RSTier::LSA : RSTier::Gin;
}

// Resolve the per-collective ReduceScatter threshold with precedence:
// per-collective env (NCCL_GIN_ANVIL_SDMA_THRESHOLD_REDUCESCATTER) > global env
// (NCCL_GIN_ANVIL_SDMA_THRESHOLD) > compiled default. "Set" means the env var was
// present and parsed (an explicit 0 is honored). Pure so the precedence is
// unit-testable without the environment; the getenv/parse wrapper lives host-side
// in reduce_scatter.cu. Mirrors gin_sdma_allgather::pickSdmaThreshold.
GIN_SDMA_RS_HD inline size_t pickSdmaThreshold(bool perCollSet, unsigned long long perCollVal,
                                               bool globalSet, unsigned long long globalVal,
                                               size_t compiledDefault) {
  if (perCollSet) return (size_t)perCollVal;
  if (globalSet) return (size_t)globalVal;
  return compiledDefault;
}

// devComm resource requirements for the -D 3 ReduceScatter kernel: one barrier +
// one lsaBarrier + one signal per CTA, GIN required. The single-tier LSA read-
// reduce actually uses only the lsaBarrier (entry), but the pools are sized
// uniformly per CTA. Mirrors the movement-collective shape.
struct DevReqs {
  int barrierCount;
  int lsaBarrierCount;
  int ginSignalCount;
  bool needsGin;
  bool supported;
};
GIN_SDMA_RS_HD inline DevReqs reduceScatterDevReqs(int deviceCtaCount) {
  DevReqs r{deviceCtaCount, deviceCtaCount, deviceCtaCount, true, true};
  return r;
}

// ReduceScatter -D 3 size-adaptive CTA count (decoupled from -V, mirrors the
// broadcast/reduce rings). The LSA read-reduce is occupancy-bound in the
// grid-stride mid-band [kReduceScatterCtaMidLo, kReduceScatterCtaMidHi): ~48 CTAs
// peaks there (33 MiB ~100% of host, 16 MiB ->98%), while the small tier and the
// large sizes (>= kReduceScatterCtaMidHi) peak at 32 -- more CTAs add xGMI incast
// (4 MiB 194->168 busbw at 48 CTAs; large sizes prefer 32 even now that the grid-
// stride 8-way path covers them). The bare -V default (16) badly under-launches
// the mid-band (16 MiB ~46%, 33 MiB ~43% of host); self-selecting repairs that for
// callers that don't pass -V. NCCL_GIN_ANVIL_RS_CTAS pins a fixed count (diagnostic).
// NOTE: the shipped kernel now runs the grid-stride path for ALL sizes (the warp-
// unroll large tier is disabled by default, see reduce_scatter.cu ReduceScatter-
// UnrollMinBytes); this ladder's >=kReduceScatterCtaMidHi arm applies to that path.
static constexpr int    kReduceScatterCtasMid   = 48;                    // grid-stride mid-band
static constexpr int    kReduceScatterCtasOther = 32;                    // small + large sizes
static constexpr size_t kReduceScatterCtaMidLo  = 8ull  * 1024 * 1024;   // >= -> mid band
static constexpr size_t kReduceScatterCtaMidHi  = 48ull * 1024 * 1024;   // <  -> mid band
GIN_SDMA_RS_HD inline int reduceScatterCtas(size_t totalBytes, size_t envCtas) {
  if (envCtas != kThresholdUnset && envCtas > 0)
    return (envCtas > 128) ? 128 : (int)envCtas;
  if (totalBytes >= kReduceScatterCtaMidLo && totalBytes < kReduceScatterCtaMidHi)
    return kReduceScatterCtasMid;
  return kReduceScatterCtasOther;
}
GIN_SDMA_RS_HD inline int reduceScatterMaxCtas() {
  return (kReduceScatterCtasMid > kReduceScatterCtasOther) ? kReduceScatterCtasMid
                                                           : kReduceScatterCtasOther;
}

// Bytes of scratch-window a future put-partials large tier would need per rank: it
// stages N incoming per-source partials, each up to the largest per-rank slice, so
// it needs the full per-rank send-buffer worth (N * maxChunkBytes ==
// maxSendBytesPerRank), rounded up to the 128 B resource-buffer granularity. Zero
// -> zero (the shipped single-tier kernel registers NO scratch). Retained for
// forward-compat / documentation.
GIN_SDMA_RS_HD inline size_t reduceScatterScratchBytes(size_t maxSendBytesPerRank) {
  if (maxSendBytesPerRank == 0) return 0;
  return (maxSendBytesPerRank + 127) & ~(size_t)127;
}

// ReduceScatter algorithm / bus bandwidth (GB/s) given the per-rank output-slice
// element count, element size, elapsed seconds and rank count. algBw counts every
// rank's contribution (count*typesize*nranks); busBw applies the ReduceScatter
// (nranks-1)/nranks correction. Mirrors ReduceScatterGetBw.
GIN_SDMA_RS_HD inline void bandwidthGBps(size_t perRankCount, int typeSize, double sec,
                                         int nranks, double* algBw, double* busBw) {
  const double baseBw = (double)(perRankCount * (size_t)typeSize * (size_t)nranks) / 1.0e9 / sec;
  if (algBw) *algBw = baseBw;
  if (busBw) {
    const double factor = ((double)(nranks - 1)) / ((double)nranks);
    *busBw = baseBw * factor;
  }
}

}  // namespace gin_sdma_reducescatter

#endif  // GIN_SDMA_REDUCESCATTER_POLICY_H_
