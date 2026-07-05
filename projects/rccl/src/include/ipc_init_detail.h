/*************************************************************************
 * Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
 *
 * Shared DDA IPC helpers for ipc_init.cu and dda_all_reduce_ipc.cu only.
 * Do not include from host .cc files (pulls GPU barrier / device types).
 * See LICENSE.txt for license information.
 ************************************************************************/

#pragma once

#include "algorithms/CollCommon.h"
#include "ipc_gpu_barrier.h"

#include <cstddef>
#include <cstdlib>
#include <memory>
#include <new>
#define DDA_IPC_MAXBLOCKS 24
#define DDA_IPC_BUFFER_SIZE 67108864

namespace nccl_dda_ipc_detail {

constexpr int kDdaNranks = meta::comms::NRANKS;

struct DdaIpcBarrierState {
  std::unique_ptr<meta::comms::IpcGpuBarrierResources> resources;
  meta::comms::IpcGpuBarrier barrierHost;
};

inline int ddaMaxNBlocksForScratch() {
  unsigned maxBlocks = DDA_IPC_MAXBLOCKS;
  return static_cast<int>(maxBlocks);
}

} // namespace nccl_dda_ipc_detail
