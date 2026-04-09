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
//RCCL_PARAM(DdaIpcMaxBlocks, "DDA_IPC_MAXBLOCKS", 24);

namespace nccl_dda_ipc_detail {

constexpr int kDdaNranks = 8;

struct DdaIpcBarrierState {
  std::unique_ptr<meta::comms::IpcGpuBarrierResources> resources;
  meta::comms::IpcGpuBarrier barrierHost;
};

/** Max grid.x for any element count that fits in scratch (covers float / half / bf16). */
inline int ddaMaxNBlocksForScratch(size_t scratchBytes) {
  constexpr unsigned threads = 512;
  unsigned maxBlocks = DDA_IPC_MAXBLOCKS;
  return static_cast<int>(maxBlocks);
}

inline size_t ddaIpcScratchBytesFromEnv() {
  const char* e = getenv("RCCL_DDA_IPC_BYTES");
  if (e == nullptr || e[0] == '\0') {
    return 64ULL * 1024 * 1024;
  }
  char* end = nullptr;
  unsigned long long v = strtoull(e, &end, 0);
  if (end == e) {
    return 64ULL * 1024 * 1024;
  }
  return static_cast<size_t>(v);
}

} // namespace nccl_dda_ipc_detail
