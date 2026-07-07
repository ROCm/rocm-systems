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
#define DDA_IPC_BUFFER_SIZE 67108864

#define DDA_FABRIC_MAXBLOCKS 24
#define DDA_FABRIC_BUFFER_SIZE 67108864

// DDA fabric LL (low-latency) all-reduce: messages at or below this many bytes
// take the remote-write LL path; larger sizes use the Simple remote-read
// flat/tree kernels. This also bounds the per-slot size of the dedicated LL
// recv buffer. Overridable via RCCL_DDA_FABRIC_LL_MAX_BYTES.
#define DDA_FABRIC_LL_MAX_BYTES 32768

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

// Byte threshold (and per-slot cap) for the DDA fabric LL all-reduce path.
inline size_t ddaFabricLLMaxBytes() {
  static size_t maxBytes = 0;
  if (maxBytes == 0) {
    size_t n = DDA_FABRIC_LL_MAX_BYTES;
    const char* s = getenv("RCCL_DDA_FABRIC_LL_MAX_BYTES");
    if (s != nullptr) {
      long v = atol(s);
      if (v > 0) {
        n = static_cast<size_t>(v);
      }
    }
    // Round down to a 16-byte (one LL packet) multiple; keep a sane floor.
    n &= ~size_t(15);
    if (n < 16) {
      n = 16;
    }
    maxBytes = n;
  }
  return maxBytes;
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

} // namespace nccl_dda_detail
