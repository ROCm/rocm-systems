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

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <memory>
#include <new>
#include <utility>

#define DDA_IPC_MAXBLOCKS 24
#define DDA_IPC_BUFFER_SIZE 268435456
static_assert(DDA_IPC_BUFFER_SIZE > dda::common::kDdaIpcLLRegionBytes, "the IPC LL region must leave Simple scratch");
static_assert(DDA_IPC_BUFFER_SIZE % 16 == 0 && dda::common::kDdaIpcLLRegionBytes % 16 == 0,
              "the IPC LL region offset must be a whole number of 16B LL lines");

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
// - simpleCap: rcclDdaScratchPayloadCap() (max DDA/CE-scratch table/env cap,
//   including graph VMM even when the comm never captures, or the pre-table
//   DDA defaults when the arch table is ignored)
// - llFloor:   2 banks * nRanks * kDdaLLMaxBytes (when LL enabled)
// - ll128Floor: whole slices per rank to carry DDA_LL128_THRESHOLD, 2 banks
//
// simpleCap is the 1:1 payload footprint (VMM Simple, AG CE-Scratch). LL/LL128
// slot arrays can still exceed that at high rank counts, which is why the
// floors remain. Kernel-internal slot caps (kDdaLLArMaxBytes, etc.) may still
// refuse a message even when scratch is large enough.
inline size_t ddaFabricScratchSizing(int nRanks, int64_t overrideBytes, int64_t ddaEnabled, size_t ddaThreshold,
                                     int64_t llEnabled, int64_t ll128Enabled, size_t ll128Threshold = 0) {
  if (overrideBytes >= 0) {
    return overrideBytes > 0 ? (size_t)overrideBytes : 0;
  }

  const size_t simpleCap = ddaEnabled && ddaThreshold > 0 ? ddaThreshold : 0;
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
    const size_t perRank = (ll128Threshold + (size_t)nRanks - 1) / (size_t)nRanks;
    size_t slotSlices = ddaLL128Slices(perRank);
    slotSlices += slotSlices & 1; // even: the two-shot tier halves this slot
    ll128Floor = (size_t)2 * nRanks * slotSlices * (size_t)kDdaLL128WireBytesPerSlice;
  }

  size_t bytes = simpleCap;
  if (llFloor > bytes) bytes = llFloor;
  if (ll128Floor > bytes) bytes = ll128Floor;

  return bytes;
}

// IPC scratch top down: LL region, staging banks 1 and 0 when banked, then the Simple / CE payload (ddaScratchBytes).
constexpr size_t kDdaIpcStagingBankBytes = size_t{8} << 20;  // larger in-kernel staging runs unbanked
static_assert(DDA_IPC_BUFFER_SIZE > dda::common::kDdaIpcLLRegionBytes + 2 * kDdaIpcStagingBankBytes,
              "the staging banks must leave IPC payload scratch");

// Device peer table entries: scratch bases, LL region bases (ddaLLRegion), staging bank 0 bases, then bank 1 bases.
constexpr int kDdaIpcStagingBankPeerIndex = 2 * kDdaNranks;
constexpr int kDdaIpcPeerTableEntries = 4 * kDdaNranks;

// Per-comm IPC barrier state stored in ncclComm::ddaIpcBarrierState.
struct DdaIpcBarrierState {
  std::unique_ptr<dda::common::IpcGpuBarrierResources> resources;
  dda::common::IpcGpuBarrier barrierHost;
  size_t scratchAllocBytes = 0;            // the whole IPC scratch, which fini frees
  size_t stagingBankBytes = 0;             // set with stagingBankPeerPtrsDev, which gates banking
  void* stagingBankPeerPtrsDev = nullptr;  // bank 0 then bank 1 peer bases inside ncclComm::ddaPeerPtrsDev; not owned
};

// RCCL_DDA_IPC_BANKED_SCRATCH: unset (-1) follows the arch table, 0 off, any other non-zero value on.
inline bool ddaIpcBankedScratchWanted(int64_t param, bool archDefault) {
  return param == -1 ? archDefault : param != 0;
}

// Simple / CE payload bytes of an allocBytes IPC scratch: below the LL region, and below the staging banks when banked.
inline size_t ddaIpcPayloadBytes(size_t allocBytes, bool banked) {
  return allocBytes - dda::common::kDdaIpcLLRegionBytes - (banked ? 2 * kDdaIpcStagingBankBytes : 0);
}

// Where staging bank 0 or 1 starts in an allocBytes IPC scratch, right below the LL region.
inline size_t ddaIpcStagingBankOffset(size_t allocBytes, int bank) {
  return ddaIpcPayloadBytes(allocBytes, /*banked=*/true) + static_cast<size_t>(bank) * kDdaIpcStagingBankBytes;
}

// For in-kernel staging only: a copy before the launch cannot read the device-side call count that picks the bank.
inline bool ddaIpcCallBanked(const DdaIpcBarrierState* state, size_t footprintBytes) {
  return state != nullptr && state->stagingBankPeerPtrsDev != nullptr && footprintBytes <= state->stagingBankBytes;
}

// The peer table an in-kernel staged call launches with: the staging bank table when it is banked, else ipcbuffs.
template <typename T>
inline T** ddaIpcStagingTable(const DdaIpcBarrierState* state, T** ipcbuffs, size_t footprintBytes, bool* banked) {
  *banked = ddaIpcCallBanked(state, footprintBytes);
  return *banked ? static_cast<T**>(state->stagingBankPeerPtrsDev) : ipcbuffs;
}

// The init ready gather also carries whether a rank wants banked scratch, so every rank banks or none does.
inline int ddaIpcReadyWord(int ready, bool banked) {
  return ready | (banked ? 1 << 8 : 0);
}

// The lowest ready level, and banked only when every rank wants it.
inline void ddaIpcAgreeReadyWords(const int* words, int nRanks, int* ready, bool* banked) {
  *ready = words[0] & 0xff;
  *banked = true;
  for (int r = 0; r < nRanks; ++r) {
    *ready = std::min(*ready, words[r] & 0xff);
    *banked = *banked && (words[r] >> 8) != 0;
  }
}

// Per-comm fabric barrier state stored in ncclComm::ddaFabricBarrierState.
struct DdaFabricBarrierState {
  std::unique_ptr<dda::common::FabricGpuBarrierResources> resources;
  dda::common::FabricGpuBarrier barrierHost;
};

inline int ddaMaxNBlocksForScratch() {
  unsigned maxBlocks = DDA_IPC_MAXBLOCKS;
  return static_cast<int>(maxBlocks);
}

// Grid and block of the IPC Simple kernels for `count` elements of `typeSize` bytes.
inline std::pair<dim3, dim3> ddaIpcSimpleGeom(size_t count, int typeSize) {
  return dda::common::getGridAndBlockDims(count, typeSize, ddaMaxNBlocksForScratch());
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
