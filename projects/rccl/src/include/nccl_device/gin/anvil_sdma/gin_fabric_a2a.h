/*************************************************************************
 * Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
 *
 * Device-side eligibility for the GIN AllToAll fabric LL small-message lane.
 * Mirrors ncclAllToAllDdaFabricLLEligible plus the host-side total-size gate.
 ************************************************************************/

#pragma once

#include <type_traits>

#include "nccl_device.h"
#include "algorithms/dda/device/CollCommon.h"

namespace gin::fabric {

using dda::common::kDdaLLMaxBytes;
using dda::common::kDdaLLA2ASlotStridePkts;
using dda::common::LLPacket16;
using dda::common::bf16;

constexpr int kDdaMaxNranks = 72;
constexpr size_t kDdaLLA2APktsPerBlock = 256;
constexpr int kDdaLLAgMaxBlocksPerPeer = 8;

__device__ __forceinline__ size_t ddaLLA2AScratchSize(int nRanks) {
  return (size_t)2 * (size_t)nRanks * kDdaLLA2ASlotStridePkts * sizeof(LLPacket16);
}

__device__ __forceinline__ int ddaLLA2ABlocksPerPeer(size_t perChunkBytes) {
  const size_t nPk = perChunkBytes >> 3;
  if (nPk <= kDdaLLA2APktsPerBlock) return 1;
  size_t bpp = (nPk + kDdaLLA2APktsPerBlock - 1) / kDdaLLA2APktsPerBlock;
  if (bpp > (size_t)kDdaLLAgMaxBlocksPerPeer) bpp = (size_t)kDdaLLAgMaxBlocksPerPeer;
  return (int)bpp;
}

template <typename T>
__device__ __forceinline__ bool ginAlltoAllFabricLLEligible(struct ncclDevComm const& devComm, size_t count) {
  if (!devComm.ginFabricSmallMsgEnabled) return false;
  if (devComm.ginFabricPeerScratch == nullptr || devComm.ginFabricLLEpoch == nullptr) return false;
  if (count == 0) return false;
  if (devComm.nRanks < 2 || devComm.nRanks > kDdaMaxNranks) return false;
  if (!std::is_same<T, float>::value && !std::is_same<T, half>::value &&
      !std::is_same<T, bf16>::value) {
    return false;
  }

  const size_t perChunkBytes = count * sizeof(T);
  if (perChunkBytes % 16 != 0) return false;
  if (perChunkBytes * 2 > kDdaLLMaxBytes) return false;
  if (ddaLLA2AScratchSize(devComm.nRanks) > devComm.ginFabricScratchBytes) return false;
  if (devComm.ginFabricLLThreshold > 0 &&
      (size_t)devComm.nRanks * perChunkBytes > devComm.ginFabricLLThreshold) {
    return false;
  }
  return true;
}

} // namespace gin::fabric
