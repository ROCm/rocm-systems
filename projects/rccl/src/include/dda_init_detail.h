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

namespace nccl_dda_detail {

constexpr int kDdaNranks = meta::comms::NRANKS;

// LL/LL128 fixed slot layout constants. These define the per-rank slot stride
// used by the kernels, which must be known at compile time so all ranks use
// identical offsets. The values match the algorithm headers:
//   LL:    128 KiB per rank (all_gather_dda_fabric_ll.h, etc.)
//   LL128: 512 KiB per rank (all_gather_dda_fabric_ll128.h, etc.)
constexpr size_t kDdaLLMaxPerRankBytes = 131072;      // 128 KiB
constexpr size_t kDdaLL128MaxPerRankBytes = 524288;   // 512 KiB
constexpr int kDdaLL128DataElems = 15;                // payload words per 128B line

// Derive slot stride from max per-rank bytes.
constexpr size_t kDdaLLSlotStridePkts = kDdaLLMaxPerRankBytes / 8;  // 16384 pkts (16B lines)
constexpr size_t kDdaLL128SlotStrideLines =
  (kDdaLL128MaxPerRankBytes / 8 + kDdaLL128DataElems - 1) / kDdaLL128DataElems;  // 4370 lines

// Compute the fabric scratch allocation from the runtime configuration.
// An explicit buffer-size override takes precedence over derived sizing.
//
// The derived size is: max(simpleCap, llFloor, ll128Floor) where:
// - simpleCap: DDA_THRESHOLD (default 128 MiB)
// - llFloor:   2 banks * nRanks * kDdaLLSlotStridePkts * 16B (when LL enabled)
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

  // LL fixed slot arrays: 2 banks * nRanks * slotStridePkts * 16B.
  const size_t llFloor = llEnabled ? (size_t)2 * nRanks * kDdaLLSlotStridePkts * 16 : 0;

  // LL128 fixed slot arrays: 2 banks * nRanks * slotStrideLines * 128B.
  const size_t ll128Floor = ll128Enabled ? (size_t)2 * nRanks * kDdaLL128SlotStrideLines * 128 : 0;

  size_t bytes = simpleCap;
  if (llFloor > bytes) bytes = llFloor;
  if (ll128Floor > bytes) bytes = ll128Floor;

  return bytes;
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
