/*************************************************************************
 * Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

// AllToAll fabric LL vs gin.put/SDMA total-size gate (bytes, nRanks * per-peer).
// If RCCL_GIN_FABRIC_LL_THRESHOLD_ALLTOALL is set (explicit 0 is honored),
// use it; otherwise fall back to RCCL_DDA_LL_THRESHOLD.

#ifndef _NCCL_DEVICE_GIN_ANVIL_SDMA_GIN_FABRIC_LL_POLICY_H_
#define _NCCL_DEVICE_GIN_ANVIL_SDMA_GIN_FABRIC_LL_POLICY_H_

#include <cstddef>

#if !defined(__CUDA_ARCH__) && !defined(__HIP_DEVICE_COMPILE__)
#include "gin/gin_fabric_a2a_host.h"
#include <cerrno>
#include <cstdlib>
#endif

namespace gin {
namespace fabric {

// Keep in sync with dda::common::kDdaMaxNranks / kDdaLLMaxBytes / LLPacket16.
constexpr int kGinFabricLlMaxNranks = 72;
constexpr size_t kGinFabricLlMaxBytes = (size_t)16 * 1024 * 1024;
constexpr size_t kGinFabricLlPacketBytes = 16;
constexpr size_t kGinFabricLlA2ASlotStridePkts = kGinFabricLlMaxBytes / kGinFabricLlPacketBytes;
constexpr size_t kGinFabricLlA2APktsPerBlock = 256;
constexpr int kGinFabricLlAgMaxBlocksPerPeer = 8;

inline size_t ginFabricLlA2AScratchBytes(int nRanks) {
  return (size_t)2 * (size_t)nRanks * kGinFabricLlA2ASlotStridePkts * kGinFabricLlPacketBytes;
}

inline int ginFabricLlAlltoAllBlocksPerPeer(size_t perChunkBytes) {
  const size_t nPk = perChunkBytes >> 3;
  if (nPk <= kGinFabricLlA2APktsPerBlock) return 1;
  size_t bpp = (nPk + kGinFabricLlA2APktsPerBlock - 1) / kGinFabricLlA2APktsPerBlock;
  if (bpp > (size_t)kGinFabricLlAgMaxBlocksPerPeer) bpp = (size_t)kGinFabricLlAgMaxBlocksPerPeer;
  return (int)bpp;
}

inline bool ginFabricLlLaneResourcesOk(int nRanks, size_t scratchBytes, size_t llThreshold) {
  if (llThreshold == 0) return false;
  if (nRanks < 2 || nRanks > kGinFabricLlMaxNranks) return false;
  if (ginFabricLlA2AScratchBytes(nRanks) > scratchBytes) return false;
  return true;
}

inline size_t pickGinFabricLLThresholdAlltoAll(bool alltoallSet, unsigned long long alltoallVal,
                                               size_t ddaLLThreshold) {
  if (alltoallSet) return (size_t)alltoallVal;
  return ddaLLThreshold;
}

#if !defined(__CUDA_ARCH__) && !defined(__HIP_DEVICE_COMPILE__)

inline bool ginFabricLlAlltoAllEligible(ncclGinFabricA2ALane const& lane, int nRanks, size_t count, size_t typeBytes,
                                        bool dtypeOk) {
  if (!lane.enabled) return false;
  if (lane.peerScratch == nullptr || lane.llEpoch == nullptr) return false;
  if (!dtypeOk || count == 0) return false;
  if (!ginFabricLlLaneResourcesOk(nRanks, lane.scratchBytes, lane.llThreshold)) return false;
  const size_t perChunkBytes = count * typeBytes;
  if (perChunkBytes % 16 != 0) return false;
  if (perChunkBytes * 2 > kGinFabricLlMaxBytes) return false;
  if ((size_t)nRanks * perChunkBytes > lane.llThreshold) return false;
  return true;
}

inline bool parseGinFabricLLThresholdEnv(const char* name, unsigned long long* val) {
  const char* e = getenv(name);
  if (!e || !e[0]) return false;
  if (*e < '0' || *e > '9') return false;
  errno = 0;
  char* end = nullptr;
  unsigned long long v = strtoull(e, &end, 10);
  if (end == e || *end != '\0' || errno == ERANGE) return false;
  *val = v;
  return true;
}

inline size_t resolveGinFabricLLThresholdAlltoAll(size_t ddaLLThreshold) {
  unsigned long long alltoallVal = 0;
  const bool alltoallSet =
      parseGinFabricLLThresholdEnv("RCCL_GIN_FABRIC_LL_THRESHOLD_ALLTOALL", &alltoallVal) ||
      parseGinFabricLLThresholdEnv("NCCL_GIN_FABRIC_LL_THRESHOLD_ALLTOALL", &alltoallVal);
  return pickGinFabricLLThresholdAlltoAll(alltoallSet, alltoallVal, ddaLLThreshold);
}

#endif // host

} // namespace fabric
} // namespace gin

#endif // _NCCL_DEVICE_GIN_ANVIL_SDMA_GIN_FABRIC_LL_POLICY_H_
