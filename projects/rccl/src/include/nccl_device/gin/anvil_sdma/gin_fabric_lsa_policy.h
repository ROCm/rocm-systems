/*************************************************************************
 * Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
 *
 * Direct LSA-flat-from-fabric AllToAll small-message policy (host + device).
 * Opt-in via RCCL_GIN_ANVIL_FABRIC_LSA_A2A / NCCL_GIN_ANVIL_FABRIC_LSA_A2A, read
 * by gin_anvil_reach.cc; this header holds no environment access.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

#ifndef _NCCL_DEVICE_GIN_ANVIL_SDMA_GIN_FABRIC_LSA_POLICY_H_
#define _NCCL_DEVICE_GIN_ANVIL_SDMA_GIN_FABRIC_LSA_POLICY_H_

#include <cstddef>
#include <cstdint>

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

// Geometry bands, after the gin_alltoall_sdma.cu LSA path. Every band uses 256
// threads, the block size kGinFabricLsaA2ADefaultThreshold was measured with.
constexpr GinFabricLsaA2ABand kGinFabricLsaA2ABands[] = {
    {16 * 1024, 2, 256},
    {64 * 1024, 4, 256},
    {512 * 1024, 8, 256},
    {~(size_t)0, 8, 256},
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

// Total-size gate (nRanks * per-peer bytes) for the direct LSA path. Per-peer
// sizes that are not a multiple of 16 bytes go to gin.put: the lane was tuned
// on vector copies only.
GIN_FABRIC_LSA_HD inline bool ginFabricLsaAlltoAllSizeOk(int nRanks, size_t perChunkBytes, size_t lsaThreshold) {
  if (lsaThreshold == 0) return false;
  if (nRanks < 2 || nRanks > kGinFabricLsaA2AMaxCtas) return false;
  if (perChunkBytes == 0 || perChunkBytes % kGinFabricLsaA2AVectorBytes != 0) return false;
  if ((size_t)nRanks * perChunkBytes > lsaThreshold) return false;
  return true;
}

// GPU pick for gin-sdma device-API A2A:
//   LSA : only if every rank opted in, total size <= LSA threshold
//   put : otherwise (Anvil SDMA / gin.put)
enum class GinFabricA2AAlgo : int { Put = 0, Lsa = 1 };

GIN_FABRIC_LSA_HD inline GinFabricA2AAlgo ginFabricA2ASelectAlgo(bool ctxOk, uint32_t lsaOn, int nRanks,
                                                                size_t perChunkBytes, size_t lsaThreshold) {
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

// Default total-size threshold of the lane. On MI450 the lane beats gin.put up to 2 MiB total and
// loses from 4-8 MiB at 4, 8 and 16 ranks alike (1P4G, 2P8G and 4P16G pods), so the crossover is a
// total size, not a per-peer one. Visible on the device pass: createContext calls this from a .cc
// that HIP still parses for device when ENABLE_DEVICE_LINKER=OFF.
constexpr size_t kGinFabricLsaA2ADefaultThreshold = (size_t)2 * 1024 * 1024;

// This rank's lane threshold: envThreshold (RCCL_GIN_ANVIL_FABRIC_LSA_THRESHOLD_ALLTOALL,
// negative when unset) or the default, and 0 when the rank did not opt in.
GIN_FABRIC_LSA_HD inline size_t ginFabricLsaA2AThreshold(bool optIn, int64_t envThreshold, int nRanks) {
  if (!optIn || nRanks < 2) return 0;
  return envThreshold >= 0 ? (size_t)envThreshold : kGinFabricLsaA2ADefaultThreshold;
}

}  // namespace fabric
}  // namespace gin

#undef GIN_FABRIC_LSA_HD

#endif  // _NCCL_DEVICE_GIN_ANVIL_SDMA_GIN_FABRIC_LSA_POLICY_H_
