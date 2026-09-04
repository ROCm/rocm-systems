/*************************************************************************
 * Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
 *
 * See LICENSE.txt for license information.
 ************************************************************************/

#pragma once

#include "algorithms/dda/device/CollCommon.h"
#include "algorithms/dda/device/CollCommon_ll128.h"
#include "algorithms/dda/fabric/fabric_gpu_barrier.h"
#include "algorithms/dda/ipc/ipc_gpu_barrier.h"

#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <memory>
#include <new>

#define DDA_IPC_MAXBLOCKS 24
#define DDA_IPC_BUFFER_SIZE 268435456

#define DDA_FABRIC_MAXBLOCKS 256

namespace nccl_dda_detail {

using dda::common::ddaLL128Slices;
using dda::common::kDdaLL128WireBytesPerSlice;
using dda::common::kDdaLLMaxBytes;
constexpr int kDdaNranks = dda::common::NRANKS;

// Compute the fabric scratch allocation from the runtime configuration.
// An explicit buffer-size override takes precedence over derived sizing.
//
// The derived size is: max(simpleCap, llFloor, ll128Floor) where:
// - simpleCap: DDA_THRESHOLD (default 128 MiB)
// - llFloor:   2 banks * nRanks * kDdaLLMaxBytes (when LL enabled)
// - ll128Floor: whole slices per rank to carry DDA_LL128_THRESHOLD, 2 banks
//
// Collectives that need more scratch (e.g., LL128 AR with large messages) are
// bounded by the eligibility check (scratchNeeded > ddaScratchBytes), which
// causes them to fall through to Simple path.
inline size_t ddaFabricScratchSizing(int nRanks, int64_t overrideBytes, int64_t ddaEnabled, int64_t ddaThreshold,
                                     int64_t llEnabled, int64_t ll128Enabled, int64_t ll128Threshold = 0) {
  if (overrideBytes >= 0) {
    return overrideBytes > 0 ? (size_t)overrideBytes : 0;
  }

  const size_t simpleCap = ddaEnabled && ddaThreshold > 0 ? (size_t)ddaThreshold : 0;
  if (simpleCap == 0) {
    return 0;
  }

  if (nRanks < 1) nRanks = 1;

  // LL fixed slot arrays: 2 banks * nRanks * slotMaxBytes.
  const size_t llFloor = llEnabled ? (size_t)2 * nRanks * kDdaLLMaxBytes : 0;

  // LL128 slot arrays sized to carry the LL128 threshold: whole slices per rank,
  // nRanks slots, 2 banks.
  size_t ll128Floor = 0;
  if ((llEnabled || ll128Enabled) && ll128Threshold > 0) {
    const size_t perRank = ((size_t)ll128Threshold + (size_t)nRanks - 1) / (size_t)nRanks;
    size_t slotSlices = ddaLL128Slices(perRank);
    slotSlices += slotSlices & 1; // even: the two-shot tier halves this slot
    ll128Floor = (size_t)2 * nRanks * slotSlices * (size_t)kDdaLL128WireBytesPerSlice;
  }

  size_t bytes = simpleCap;
  if (llFloor > bytes) bytes = llFloor;
  if (ll128Floor > bytes) bytes = ll128Floor;

  return bytes;
}

// Per-comm IPC barrier state stored in ncclComm::ddaIpcBarrierState.
struct DdaIpcBarrierState {
  std::unique_ptr<dda::common::IpcGpuBarrierResources> resources;
  dda::common::IpcGpuBarrier barrierHost;
};

// Per-comm fabric barrier state stored in ncclComm::ddaFabricBarrierState.
struct DdaFabricBarrierState {
  std::unique_ptr<dda::common::FabricGpuBarrierResources> resources;
  dda::common::FabricGpuBarrier barrierHost;
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

// Number of device epoch cells for the LL collectives. it is sized for the larger of the two
// max(AG total blocks, AR total blocks).
inline size_t ddaLLEpochCount(int nRanks, int arMaxBlocks) {
  const size_t ag = (size_t)nRanks * (size_t)kDdaLLAgMaxBlocksPerPeer;
  const size_t ar = (size_t)arMaxBlocks;
  return ag > ar ? ag : ar;
}

} // namespace nccl_dda_detail
