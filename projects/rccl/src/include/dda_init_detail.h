/*************************************************************************
 * Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
 *
 * See LICENSE.txt for license information.
 ************************************************************************/

#pragma once

#include "algorithms/CollCommon.h"
#include "fabric_gpu_barrier.h"
#include "ipc_gpu_barrier.h"

#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <memory>
#include <new>

#define DDA_IPC_MAXBLOCKS 24
#define DDA_IPC_BUFFER_SIZE 268435456

#define DDA_FABRIC_MAXBLOCKS 256
#define DDA_FABRIC_BUFFER_SIZE 10737418240ULL

namespace nccl_dda_detail {

constexpr int kDdaNranks = meta::comms::NRANKS;

// Per-comm IPC barrier state stored in ncclComm::ddaIpcBarrierState.
struct DdaIpcBarrierState {
  std::unique_ptr<meta::comms::IpcGpuBarrierResources> resources;
  meta::comms::IpcGpuBarrier barrierHost;
};

// Per-comm fabric barrier state stored in ncclComm::ddaFabricBarrierState.
struct DdaFabricBarrierState {
  std::unique_ptr<meta::comms::FabricGpuBarrierResources> resources;
  meta::comms::FabricGpuBarrier barrierHost;
};

inline int ddaMaxNBlocksForScratch() {
  unsigned maxBlocks = DDA_IPC_MAXBLOCKS;
  return static_cast<int>(maxBlocks);
}

inline int ddaFabricMaxNBlocksForScratch() {
  static int maxBlocks = -1;
  if (maxBlocks < 0) {
    int n = DDA_FABRIC_MAXBLOCKS;
    const char* s = getenv("RCCL_DDA_FABRIC_MAXBLOCKS");
    if (s != nullptr) {
      n = atoi(s);
    }
    if (n < 1) {
      n = 1;
    }
    if (n > 256) {
      n = 256;
    }
    maxBlocks = n;
  }
  return maxBlocks;
}

constexpr int kDdaLLAgMaxBlocksPerPeer = 8;

// The LL AllReduce tier is intentionally narrow (tiny messages, latency-bound),
// so it caps its grid at kDdaFabricLLArMaxBlocks instead of the 256-wide limit
// used by LL128/Simple. AR uses the shared ddaLLEpochDev counter (same as AG/RS)
// so that bank = flag & 1 is consistent across all LL operation types.
constexpr int kDdaFabricLLArMaxBlocks = 24;

// Number of device epoch cells for the LL collectives. it is sized for the larger of the two
// max(AG total blocks, AR total blocks).
inline size_t ddaLLEpochCount(int nRanks, int arMaxBlocks) {
  const size_t ag = (size_t)nRanks * (size_t)kDdaLLAgMaxBlocksPerPeer;
  const size_t ar = (size_t)arMaxBlocks;
  return ag > ar ? ag : ar;
}

} // namespace nccl_dda_detail
