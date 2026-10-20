/*************************************************************************
 * Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
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
// NOTE on the tiers: GinReduceScatterKernel is CTA-budget hybrid.
//   * gridDim.x >= kReduceScatterSdmaCtaCeil (16): single-tier LSA read-reduce.
//     That is the default. The size-adaptive ladder launches 32 or 48 CTAs.
//   * gridDim.x < 16: GIN/SDMA scatter of each per-dest slice into the owner's
//     resource-window slots, then a local SM reduce. Entered only when
//     NCCL_GIN_ANVIL_RS_CTAS pins a grid below 16. The size-based RSTier helpers
//     are not what selects this path.

#ifndef GIN_SDMA_REDUCESCATTER_POLICY_H_
#define GIN_SDMA_REDUCESCATTER_POLICY_H_

#include <cstddef>
#include <cstdint>
#include <cstdlib>

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

// Element offset of rank's owned output slice in every peer's send buffer.
// Shared by the production LSA read-reduce and its GPU addressing test.
GIN_SDMA_RS_HD inline size_t reduceScatterSliceOffset(int rank, size_t perRankCount) {
  return (size_t)rank * perRankCount;
}

// Compiled default ReduceScatter LSA<->GIN crossover (bytes per rank slice).
// RETAINED FOR DOCUMENTATION: the launched kernel keys the SDMA tier off CTA
// count (usesSdmaTier), not this slice threshold. Used by pickSdmaThreshold.
static constexpr size_t kReduceScatterSdmaThresholdDefault = 262144;  // 256 KiB/rank slice

enum class RSTier { LSA, Gin };

// Size-based predicate (parity with AllGather). The launched kernel uses
// usesSdmaTier(gridDim.x) instead.
GIN_SDMA_RS_HD inline RSTier reduceScatterKernelTier(size_t sliceBytes_, size_t sdmaThreshold) {
  return (sliceBytes_ <= sdmaThreshold) ? RSTier::LSA : RSTier::Gin;
}

// Grids smaller than this take the SDMA-scatter + local-reduce path. Default
// self-select (32/48) stays on LSA unless NCCL_GIN_ANVIL_RS_CTAS pins below this.
static constexpr int kReduceScatterSdmaCtaCeil = 16;
// Put-side grid used by the host policy test. The kernel launches whatever pin
// usesSdmaTier accepts; this is the suggested small grid, capped by nRanks.
static constexpr int kReduceScatterCtasSdma = 4;
// Max bytes per source slot in the SDMA scratch window. A larger slice falls
// back to LSA at the same small grid. 64 MiB/slot is 512 MiB at 8 ranks.
static constexpr size_t kReduceScatterSdmaSlotMaxDefault = 64ull * 1024 * 1024;

GIN_SDMA_RS_HD inline bool usesSdmaTier(int gridCtas) {
  return gridCtas > 0 && gridCtas < kReduceScatterSdmaCtaCeil;
}

GIN_SDMA_RS_HD inline int reduceScatterSdmaCtas(int nRanks) {
  const int n = (nRanks > 0) ? nRanks : 1;
  return (kReduceScatterCtasSdma < n) ? kReduceScatterCtasSdma : n;
}

// True when the registered scratch window can hold one slot per rank. Rejects a
// zero slice and a multiply that would wrap.
GIN_SDMA_RS_HD inline bool sdmaScratchFits(size_t sliceBytes_, int nRanks, size_t scratchBytes) {
  if (nRanks <= 0 || sliceBytes_ == 0) return false;
  if (sliceBytes_ > (size_t)-1 / (size_t)nRanks) return false;
  return scratchBytes >= sliceBytes_ * (size_t)nRanks;
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
// The kernel uses one grid-stride load schedule for all sizes; this ladder's
// >=kReduceScatterCtaMidHi arm still applies to that path.
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

// Size of the barrier/lsaBarrier/signal pools the ReduceScatter kernel launches
// with: max(-V/deviceCtaCount, the size-adaptive ladder's peak). This is BOTH the
// pool allocated in ReduceScatterGetDevCommRequirements AND the ceiling that
// reduceScatterGridCtas() clamps any launched grid to, so the "grid <= pool"
// invariant lives in one place. Mirrors gin_sdma_allgather::allGatherPoolCtas.
GIN_SDMA_RS_HD inline int reduceScatterPoolCtas(int deviceCtaCount) {
  const int maxTier = reduceScatterMaxCtas();
  return (deviceCtaCount > maxTier) ? deviceCtaCount : maxTier;
}

// Launched grid CTA count: the size-adaptive ladder (or an NCCL_GIN_ANVIL_RS_CTAS
// pin) clamped to poolCtas. The kernel indexes devComm.lsaBarrier by blockIdx.x,
// so a grid above the pool would index past the array -- reduceScatterCtas() alone
// caps a pin at 128, which exceeds the default pool of max(-V, 48).
GIN_SDMA_RS_HD inline int reduceScatterGridCtas(size_t totalBytes, size_t envCtas,
                                                int poolCtas) {
  if (poolCtas < 1) poolCtas = 1;
  const int c = reduceScatterCtas(totalBytes, envCtas);
  return (c > poolCtas) ? poolCtas : c;
}

// Bytes of scratch-window the SDMA-scatter tier needs: N incoming per-source
// slots, rounded up to the 128 B resource-buffer granularity. Zero if the
// input is zero.
GIN_SDMA_RS_HD inline size_t reduceScatterScratchBytes(size_t maxSendBytesPerRank) {
  if (maxSendBytesPerRank == 0) return 0;
  return (maxSendBytesPerRank + 127) & ~(size_t)127;
}

GIN_SDMA_RS_HD inline size_t reduceScatterSdmaScratchBytes(int nRanks, size_t slotMax) {
  if (nRanks <= 0 || slotMax == 0) return 0;
  if (slotMax > (size_t)-1 / (size_t)nRanks) return 0;
  return reduceScatterScratchBytes((size_t)nRanks * slotMax);
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

// Whether ReduceScatterRunTest skips one (type, op) pair of its sweep.
//   * fp8 never runs prod or mulsum, on either path.
//   * The host path (deviceImpl == 0) skips floating avg: ring ReduceScatter
//     reaches it through PreMulSum on non-direct primitives, which is only
//     validated on AllReduce's direct path. Integer avg (SumPostDiv) runs. The
//     -D 3 kernel implements avg itself, so the device path runs it.
//   * The device path has no PreMulSum kernel, so it skips mulsum for every
//     type; the host path keeps it.
inline bool reduceScatterSkipsOp(int deviceImpl, bool floatingType, bool fp8Type,
                                 bool isProd, bool isAvg, bool isMulsum) {
  if (fp8Type && (isProd || isMulsum)) return true;
  if (deviceImpl == 0 && isAvg && floatingType) return true;
  if (deviceImpl != 0 && isMulsum) return true;
  return false;
}

// Host-side env resolution (unit-testable without pulling in reduce_scatter.cu).
//
// Deliberately NOT wrapped in `#if !defined(__HIP_DEVICE_COMPILE__)`. HIP parses
// the whole translation unit in the device pass, including the bodies of host
// functions it will not codegen, so hiding these there breaks name lookup in
// ReduceScatterParseCtasEnv and fails the reduce_scatter_perf build on every
// arch. Plain `inline` is already host-only. Note gin_sdma_allgather_policy.h
// does the opposite: it guards parseAllGatherCtasEnv, and all_gather.cu works
// around that guard with a __HIP_DEVICE_COMPILE__ branch.

// Parse NCCL_GIN_ANVIL_RS_CTAS. Returns kThresholdUnset for null/empty/negative/
// trailing-garbage so "8foo" and "-2" do not pin a CTA count. strtoull wraps a
// leading '-' into a huge unsigned, which the CTA clamp would then honor.
inline size_t parseReduceScatterCtasEnvString(const char* e) {
  if (e == nullptr) return kThresholdUnset;
  // strtoull skips leading whitespace before a sign; trim it first so " -2"
  // cannot wrap to a large unsigned CTA pin.
  while (*e == ' ' || *e == '\t') e++;
  if (e[0] == '\0' || e[0] == '-') return kThresholdUnset;
  char* end = nullptr;
  unsigned long long v = strtoull(e, &end, 10);
  if (end == e || *end != '\0') return kThresholdUnset;
  if ((size_t)v == kThresholdUnset) return kThresholdUnset;
  return (size_t)v;
}

inline size_t parseReduceScatterCtasEnv(const char* name) {
  return parseReduceScatterCtasEnvString(getenv(name));
}

}  // namespace gin_sdma_reducescatter

#endif  // GIN_SDMA_REDUCESCATTER_POLICY_H_
