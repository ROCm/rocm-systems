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
#include <cstdlib>
#include <memory>
#include <new>

#define DDA_IPC_MAXBLOCKS 24
#define DDA_IPC_BUFFER_SIZE 268435456

#define DDA_FABRIC_MAXBLOCKS 24
// 288 MiB: size of LL128 all-gather scratch, 2 * total_size * 16/15, fits a 128 MiB msg.
// Raising DDA_ALLGATHER_LL128_THRESHOLD beyond 128 MiB need this to be adjusted
#define DDA_FABRIC_BUFFER_SIZE 301989888

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

constexpr int kDdaLLMaxBlocks = 256;

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
    if (n > kDdaLLMaxBlocks) {
      n = kDdaLLMaxBlocks;
    }
    maxBlocks = n;
  }
  return maxBlocks;
}

} // namespace nccl_dda_detail
