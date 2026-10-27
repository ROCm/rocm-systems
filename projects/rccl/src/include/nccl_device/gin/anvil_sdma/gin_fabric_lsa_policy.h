/*************************************************************************
 * Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
 *
 * Direct LSA-flat-from-fabric AllToAll small-message policy (host + device).
 * Opt-in via RCCL_GIN_FABRIC_LSA_A2A / NCCL_GIN_FABRIC_LSA_A2A.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

#ifndef _NCCL_DEVICE_GIN_ANVIL_SDMA_GIN_FABRIC_LSA_POLICY_H_
#define _NCCL_DEVICE_GIN_ANVIL_SDMA_GIN_FABRIC_LSA_POLICY_H_

#include <cstddef>
#include <cstdint>
#include "gin_fabric_ll_policy.h"

#if defined(__CUDACC__) || defined(__HIPCC__)
#define GIN_FABRIC_LSA_HD __host__ __device__
#else
#define GIN_FABRIC_LSA_HD
#endif

namespace gin {
namespace fabric {

constexpr int kGinFabricLsaA2AMaxCtas = 64;
constexpr int kGinFabricLsaA2AMaxChunksPerPeer = 8;
constexpr size_t kGinFabricLsaA2AVectorBytes = 16;

struct GinFabricLsaA2ABand {
  size_t maxBytesPerPeer;
  int chunksPerPeer;
  int threadsPerCta;
};

// Geometry bands aligned with gin_alltoall_sdma.cu LSA path.
constexpr GinFabricLsaA2ABand kGinFabricLsaA2ABands[] = {
    {16 * 1024, 2, 256},
    {64 * 1024, 4, 256},
    {512 * 1024, 8, 256},
    {~(size_t)0, 8, 512},
};

GIN_FABRIC_LSA_HD inline size_t ginFabricLsaA2ADivUp(size_t a, size_t b) { return (a + b - 1) / b; }
GIN_FABRIC_LSA_HD inline size_t ginFabricLsaA2AAlignUp(size_t a, size_t b) {
  return ginFabricLsaA2ADivUp(a, b) * b;
}

GIN_FABRIC_LSA_HD inline void ginFabricLsaA2ALaunchConfig(int nRanks, size_t bytesPerPeer, int* chunksPerPeer,
                                                            int* threads, size_t* chunkBytes) {
  const GinFabricLsaA2ABand* band = kGinFabricLsaA2ABands;
  while (bytesPerPeer >= band->maxBytesPerPeer) band++;

  int maxChunks = kGinFabricLsaA2AMaxCtas / nRanks;
  if (maxChunks < 1) maxChunks = 1;
  int chunks = band->chunksPerPeer > maxChunks ? maxChunks : band->chunksPerPeer;

  *chunksPerPeer = chunks;
  *threads = band->threadsPerCta;
  *chunkBytes = ginFabricLsaA2AAlignUp(ginFabricLsaA2ADivUp(bytesPerPeer, (size_t)chunks), kGinFabricLsaA2AVectorBytes);
}

GIN_FABRIC_LSA_HD inline int ginFabricLsaA2ALsaBarrierPool(int nRanks) {
  int pool = nRanks * kGinFabricLsaA2AMaxChunksPerPeer;
  if (pool > kGinFabricLsaA2AMaxCtas) pool = kGinFabricLsaA2AMaxCtas;
  return pool;
}

// Total-size gate (nRanks * per-peer bytes) for the direct LSA path.
GIN_FABRIC_LSA_HD inline bool ginFabricLsaAlltoAllSizeOk(int nRanks, size_t perChunkBytes, size_t lsaThreshold) {
  if (lsaThreshold == 0) return false;
  if (nRanks < 2 || nRanks > kGinFabricLsaA2AMaxCtas) return false;
  if (perChunkBytes == 0) return false;
  if ((size_t)nRanks * perChunkBytes > lsaThreshold) return false;
  return true;
}

// GPU pick for gin-sdma device-API A2A (measured 1p4g MI455):
//   LL  : total size <= fabric LL threshold (default 256 KiB)
//   LSA : only if opted in and LL not taken, total size <= LSA threshold
//   put : otherwise (Anvil SDMA / gin.put). Default leaves LSA off because the
//         256 KiB–32 MiB LSA band is slower than gin.put on this platform.
enum class GinFabricA2AAlgo : int { Put = 0, Ll = 1, Lsa = 2 };

GIN_FABRIC_LSA_HD inline GinFabricA2AAlgo ginFabricA2ASelectAlgo(bool ctxOk, uint32_t llOn, uint32_t lsaOn, int nRanks,
                                                                size_t perChunkBytes, size_t llThreshold,
                                                                size_t llScratchBytes, size_t lsaThreshold) {
  if (ctxOk && llOn != 0 && ginFabricLlAlltoAllSizeOk(nRanks, perChunkBytes, llThreshold, llScratchBytes)) {
    return GinFabricA2AAlgo::Ll;
  }
  if (ctxOk && lsaOn != 0 && ginFabricLsaAlltoAllSizeOk(nRanks, perChunkBytes, lsaThreshold)) {
    return GinFabricA2AAlgo::Lsa;
  }
  return GinFabricA2AAlgo::Put;
}

GIN_FABRIC_LSA_HD inline bool ginFabricLsaA2ACapablePredicate(bool fabricMemPath, bool mnnvl, bool fabricHandles,
                                                              int ginNranks, int commNranks, int lsaTeamNranks) {
  if (!fabricMemPath || !mnnvl || !fabricHandles) return false;
  if (ginNranks != commNranks) return false;
  if (lsaTeamNranks != commNranks) return false;
  return true;
}

#if !defined(__CUDA_ARCH__) && !defined(__HIP_DEVICE_COMPILE__)

#include <cstdlib>

// Matches host GIN A2A GIN_A2A_SDMA_MIN_BYTES: LSA below this per-peer size, SDMA at/above.
constexpr size_t kGinFabricLsaA2ADefaultMaxPerPeer = (size_t)8 * 1024 * 1024;

inline bool ginFabricLsaA2AOptInFromEnv() {
  const char* e = getenv("RCCL_GIN_FABRIC_LSA_A2A");
  if (!e || !e[0]) e = getenv("NCCL_GIN_FABRIC_LSA_A2A");
  return e && atoi(e) != 0;
}

// -1 = unset (off), 0 = force off, 1 = force on.
inline int ginFabricLsaA2AEnvMode() {
  const char* e = getenv("RCCL_GIN_FABRIC_LSA_A2A");
  if (!e || !e[0]) e = getenv("NCCL_GIN_FABRIC_LSA_A2A");
  if (!e || !e[0]) return -1;
  return atoi(e) != 0 ? 1 : 0;
}

inline bool ginFabricLsaA2AWantEnabled(bool capable) {
  if (!capable) return false;
  // 1p4g MI455: LSA between the LL cap and 8 MiB/peer is slower than gin.put.
  // Unset stays off; RCCL_GIN_FABRIC_LSA_A2A=1 opts in.
  return ginFabricLsaA2AEnvMode() == 1;
}

inline size_t resolveGinFabricLsaThresholdAlltoAll(int nRanks) {
  unsigned long long val = 0;
  if (parseGinFabricLLThresholdEnv("RCCL_GIN_FABRIC_LSA_THRESHOLD_ALLTOALL", &val) ||
      parseGinFabricLLThresholdEnv("NCCL_GIN_FABRIC_LSA_THRESHOLD_ALLTOALL", &val)) {
    return (size_t)val;
  }
  if (nRanks < 2) return 0;
  if (ginFabricLsaA2AEnvMode() != 1) return 0;
  return (size_t)nRanks * kGinFabricLsaA2ADefaultMaxPerPeer;
}

#endif  // host

}  // namespace fabric
}  // namespace gin

#undef GIN_FABRIC_LSA_HD

#endif  // _NCCL_DEVICE_GIN_ANVIL_SDMA_GIN_FABRIC_LSA_POLICY_H_
