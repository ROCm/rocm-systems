/*************************************************************************
 * Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
 *
 * See LICENSE.txt for license information.
 ************************************************************************/

#pragma once

#include "algorithms/dda/device/CollCommon.h"
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

using dda::common::kDdaLLMaxBytes;
constexpr int kDdaNranks = dda::common::NRANKS;

// LL/LL128 fixed slot layout constants. These define the per-rank slot stride
// used by the kernels, which must be known at compile time so all ranks use
// identical offsets. The values match the algorithm headers:
//   LL128: 512 KiB per rank (all_gather_dda_fabric_ll128.h, etc.)
constexpr size_t kDdaLL128MaxPerRankBytes = 524288;   // 512 KiB
constexpr int kDdaLL128DataElems = 15;                // payload words per 128B line

// Derive slot stride from max per-rank bytes.
constexpr size_t kDdaLL128SlotStrideLines =
  (kDdaLL128MaxPerRankBytes / 8 + kDdaLL128DataElems - 1) / kDdaLL128DataElems;  // 4370 lines

// Compute the fabric scratch allocation from the runtime configuration.
// An explicit buffer-size override takes precedence over derived sizing.
//
// The derived size is: max(simpleCap, llFloor, ll128Floor) where:
// - simpleCap: DDA_THRESHOLD (default 128 MiB)
// - llFloor:   2 banks * nRanks * kDdaLLMaxBytes (when LL enabled)
// - ll128Floor: 2 banks * nRanks * kDdaLL128SlotStrideLines * 128B (when LL128 enabled)
//
// Collectives that need more scratch (e.g., LL128 AR with large messages) are
// bounded by the eligibility check (scratchNeeded > ddaScratchBytes), which
// causes them to fall through to Simple path.
inline size_t ddaFabricScratchSizing(int nRanks, int64_t overrideBytes, int64_t ddaEnabled, int64_t ddaThreshold,
                                     int64_t llEnabled, int64_t ll128Enabled) {
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

  // LL128 fixed slot arrays: 2 banks * nRanks * slotStrideLines * 128B.
  const size_t ll128Floor = ll128Enabled ? (size_t)2 * nRanks * kDdaLL128SlotStrideLines * 128 : 0;

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

inline int ddaFabricMaxNBlocksForScratch(int cuCount, const char* overrideValue) {
  int maxBlocks = cuCount;
  if (maxBlocks < 1) {
    maxBlocks = 1;
  }
  if (maxBlocks > DDA_FABRIC_MAXBLOCKS) {
    maxBlocks = DDA_FABRIC_MAXBLOCKS;
  }

  if (overrideValue != nullptr) {
    int requested = atoi(overrideValue);
    if (requested < 1) {
      requested = 1;
    }
    if (requested < maxBlocks) {
      maxBlocks = requested;
    }
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
