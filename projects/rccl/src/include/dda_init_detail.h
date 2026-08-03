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

// Maximum message accepted by the LL128 AllReduce path. Its per-call slot
// stride is derived from the actual message, while eligibility is bounded by
// both this cap and the communicator's runtime scratch capacity.
constexpr size_t kDdaLL128ArMaxBytes = 1073741824ULL; // 1 GiB

struct DdaFabricScratchSizing {
  size_t bytes;
  size_t effectiveLL128Threshold;
};

// Compute the fabric scratch allocation and the effective LL128 AllReduce
// threshold from one configuration snapshot. An explicit buffer-size override
// takes precedence over derived sizing, while protocol thresholds still report
// their effective values.
inline DdaFabricScratchSizing ddaFabricScratchSizing(int nRanks, int64_t overrideBytes, int64_t ddaEnabled,
                                                     int64_t ddaThreshold, int64_t ll128Enabled,
                                                     int64_t ll128Threshold) {
  const size_t simpleCap = ddaEnabled && ddaThreshold > 0 ? (size_t)ddaThreshold : 0;

  size_t ll128Cap = simpleCap && ll128Enabled && ll128Threshold > 0 ? (size_t)ll128Threshold : 0;
  if (ll128Cap > simpleCap) ll128Cap = simpleCap;
  if (ll128Cap > kDdaLL128ArMaxBytes) ll128Cap = kDdaLL128ArMaxBytes;

  if (overrideBytes >= 0) {
    return {overrideBytes > 0 ? (size_t)overrideBytes : 0, ll128Cap};
  }
  if (simpleCap == 0) {
    return {0, ll128Cap};
  }

  if (nRanks < 1) nRanks = 1;

  // LL128 line geometry (see CollCommon_ll128.h): 128B lines, 15 payload words.
  const size_t words = (ll128Cap + 7) / 8;
  const size_t lines = (words + 14) / 15;
  const size_t ll128Ar = (size_t)2 * (size_t)nRanks * lines * 128;

  size_t bytes = simpleCap > ll128Ar ? simpleCap : ll128Ar;
  bytes += bytes / 8; // ~12% margin for the fixed LL/AG/A2A/RS slot arrays
  return {bytes, ll128Cap};
}

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
