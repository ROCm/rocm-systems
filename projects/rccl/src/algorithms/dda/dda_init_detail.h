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
constexpr int kDdaNranks = dda::common::NRANKS;

// The fabric DDA scratch is one allocation split into one region per protocol,
// at the same offsets on every rank. A flag word of one protocol must never hold
// another protocol's data: the LL/LL128 receivers accept a line as soon as its
// flag matches the epoch, so stale Simple, CE or LL128 bytes in an LL flag slot
// could otherwise pass for this epoch's data. Each size comes from its own CVAR
// (RCCL_DDA_FABRIC_{SIMPLE,CE,LL,LL128}_BUFFER_SIZE); a tier whose message does
// not fit its region fails eligibility and falls back to the next tier.
enum DdaFabricRegion { kDdaRegionSimple = 0, kDdaRegionCe, kDdaRegionLL, kDdaRegionLL128, kDdaRegionCount };

// Region granule: keeps every region base 128B-aligned for LL128 lines.
constexpr size_t kDdaFabricRegionAlign = 4096;

struct DdaFabricRegionLayout {
  size_t offset[kDdaRegionCount];
  size_t bytes[kDdaRegionCount]; // rounded up to kDdaFabricRegionAlign; 0 = region (and its tiers) disabled
  size_t total;
};

// Lays the regions out in enum order. A negative request is treated as 0.
inline DdaFabricRegionLayout ddaFabricRegionLayout(const int64_t (&requested)[kDdaRegionCount]) {
  DdaFabricRegionLayout layout{};
  size_t offset = 0;
  for (int i = 0; i < kDdaRegionCount; ++i) {
    const size_t want = requested[i] > 0 ? (size_t)requested[i] : 0;
    const size_t bytes = (want + kDdaFabricRegionAlign - 1) / kDdaFabricRegionAlign * kDdaFabricRegionAlign;
    layout.offset[i] = offset;
    layout.bytes[i] = bytes;
    offset += bytes;
  }
  layout.total = offset;
  return layout;
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

struct DdaFabricMaxBlocksOverride {
  bool specified = false;
  bool valid = false;
  long requested = 0;
};

inline int ddaFabricMaxNBlocksForScratch(int cuCount, const char* overrideValue,
                                         DdaFabricMaxBlocksOverride* parsedOverride = nullptr) {
  int maxBlocks = cuCount;
  if (maxBlocks < 1) {
    maxBlocks = 1;
  }
  if (maxBlocks > DDA_FABRIC_MAXBLOCKS) {
    maxBlocks = DDA_FABRIC_MAXBLOCKS;
  }

  DdaFabricMaxBlocksOverride parsed;
  if (overrideValue != nullptr && overrideValue[0] != '\0') {
    parsed.specified = true;
    char* endptr = nullptr;
    parsed.requested = strtol(overrideValue, &endptr, 10);
    // Only apply if the entire string was a valid integer
    if (endptr != overrideValue && *endptr == '\0') {
      parsed.valid = true;
      // Clamp to [1, maxBlocks] while still a long to avoid int overflow
      if (parsed.requested < 1) {
        maxBlocks = 1;
      } else if (parsed.requested < maxBlocks) {
        maxBlocks = static_cast<int>(parsed.requested);
      }
      // requested >= maxBlocks: keep maxBlocks unchanged (override cannot raise)
    }
  }

  if (parsedOverride != nullptr) {
    *parsedOverride = parsed;
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
