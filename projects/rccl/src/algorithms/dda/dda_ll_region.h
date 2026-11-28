/*************************************************************************
 * Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
 *
 * Where the DDA LL tiers stage their lines, on the fabric handler and on the single-node IPC handler alike.
 *
 * See LICENSE.txt for license information.
 ************************************************************************/

#ifndef DDA_LL_REGION_H_
#define DDA_LL_REGION_H_

#include <algorithm>
#include <cstddef>

#include "algorithms/dda/dda_init_detail.h"
#include "comm.h"

namespace nccl_dda_detail {

// The comm runs DDA on the single-node IPC handler (gfx942, gfx950), not on the fabric one.
inline bool ddaLLOnIpc(const ncclComm* comm) {
  return comm->ddaFabricMemHandler == nullptr && comm->ddaIpcMemHandler != nullptr;
}

// What an LL tier needs: scratch, the peer table; on IPC also the LL epoch cells and the 8-rank single-node comm.
inline bool ddaLLRegionReady(const ncclComm* comm) {
  if (comm->ddaScratch == nullptr || comm->ddaPeerPtrsDev == nullptr) {
    return false;
  }
  if (comm->ddaFabricMemHandler != nullptr) {
    return true;
  }
  return comm->ddaIpcMemHandler != nullptr && comm->ddaLLEpochDev != nullptr && comm->ddaLLEpochLen > 0 &&
         comm->nNodes == 1 && comm->nRanks == kDdaNranks;
}

// Rank r's LL lines: two banks of `bank` bytes from peers[r], bank = flag & 1. Fabric: the whole scratch; IPC: the
// region past ddaScratchBytes.
struct DdaLLRegion {
  void** peers;
  size_t bank;
};

inline DdaLLRegion ddaLLRegion(const ncclComm* comm) {
  void** peers = static_cast<void**>(comm->ddaPeerPtrsDev);
  if (ddaLLOnIpc(comm)) {
    return {peers + kDdaNranks, dda::common::ddaBankSize(dda::common::kDdaIpcLLRegionBytes)};
  }
  return {peers, dda::common::ddaBankSize(comm->ddaScratchBytes)};
}

// Grid of an IPC LL launch over unitBytes, a thread per packet: within the IPC Simple grid for those bytes, or
// wideBlocks when set (a table wide window, rcclDdaIpcLLWideBlocks), and the epoch cells, and capBlocks when set.
inline unsigned ddaIpcLLBlocks(const ncclComm* comm, size_t unitBytes, unsigned threads, uint32_t capBlocks = 0,
                               uint32_t wideBlocks = 0) {
  const size_t grid = wideBlocks > 0 ? wideBlocks : ddaIpcSimpleGeom(unitBytes, 1).first.x;
  const size_t cells = (size_t)std::max(comm->ddaLLEpochLen, 1);
  const size_t budget = dda::common::ddaCapBlocks(std::min(grid, cells), capBlocks);
  const size_t need = ((unitBytes >> 3) + threads - 1) / threads;
  return (unsigned)std::max<size_t>(std::min(need, budget), 1);
}

// Rows per peer column of a block-per-peer IPC LL launch (AllGather, AlltoAll) over unitBytes per peer, a thread per
// packet: within wideBlocks shared by the nRanks columns, kDdaLLAgMaxBlocksPerPeer and the epoch cells (on IPC at
// least kDdaNranks x kDdaLLAgMaxBlocksPerPeer, ddaLLEpochCount). At least one: every peer needs a column.
inline unsigned ddaIpcLLPeerRows(const ncclComm* comm, size_t unitBytes, unsigned threads, uint32_t wideBlocks) {
  const size_t cols = (size_t)std::max(comm->nRanks, 1);
  const size_t cells = (size_t)std::max(comm->ddaLLEpochLen, 1);
  const size_t wanted = ((unitBytes >> 3) + threads - 1) / threads;
  const size_t rows = std::min({wanted, (size_t)wideBlocks / cols, (size_t)kDdaLLAgMaxBlocksPerPeer, cells / cols});
  return (unsigned)std::max<size_t>(rows, 1);
}

} // namespace nccl_dda_detail

#endif // DDA_LL_REGION_H_
