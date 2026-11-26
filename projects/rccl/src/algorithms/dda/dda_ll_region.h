/*************************************************************************
 * Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
 *
 * Where the DDA LL tiers stage their lines, on the fabric handler and on the single-node IPC handler alike.
 *
 * See LICENSE.txt for license information.
 ************************************************************************/

#ifndef DDA_LL_REGION_H_
#define DDA_LL_REGION_H_

#include <cstddef>

#include "algorithms/dda/dda_init_detail.h"
#include "comm.h"

namespace nccl_dda_detail {

// The comm runs DDA on the single-node IPC handler (gfx942, gfx950), not on the fabric one.
inline bool ddaLLOnIpc(const ncclComm* comm) {
  return comm->ddaFabricMemHandler == nullptr && comm->ddaIpcMemHandler != nullptr;
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

} // namespace nccl_dda_detail

#endif // DDA_LL_REGION_H_
