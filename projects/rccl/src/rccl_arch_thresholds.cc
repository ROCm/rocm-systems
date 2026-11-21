/*************************************************************************
 * Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

// Per-architecture DDA/CE dispatch threshold tables and tuning index lookup.
// Moved from graph/tuning.cc to isolate RCCL-specific dispatch data from
// NVIDIA-originated graph infrastructure.

#include "rccl_arch_thresholds.h"
#include "archinfo.h"  // IsArchMatch
#include "net.h"       // rcclPrimaryNic, rcclIBNicTypeAINIC

#include <string>
#include <vector>

// Per-arch DDA/CE dispatch threshold tables.
// DDA arrays: [Broadcast=0, Reduce=1, AllGather=2, ReduceScatter=3, AllReduce=4,
//              SendRecv=5, Send=6, Recv=7, AlltoAll=8]. 0 disables that tier.
// All collectives compare total message bytes against the table values.

// The tables live next to each arch's rules in rccl_scheduler/arch/.
#include "rccl_scheduler/arch/gfx1250_thresholds.h"
#include "rccl_scheduler/arch/gfx950_thresholds.h"
#include "rccl_scheduler/arch/gfx942_thresholds.h"

bool rcclWindowFits(size_t totalBytes, size_t minBytes, size_t maxBytes) {
  if (maxBytes != 0 && minBytes > maxBytes) {
    static bool warned = false;
    if (!warned) {
      warned = true;
      WARN("rcclWindowFits: min (%zu) > max (%zu); path silently disabled", minBytes, maxBytes);
    }
  }
  return maxBytes > 0 && totalBytes >= minBytes && totalBytes <= maxBytes;
}

const rcclArchThresholds* rcclGetArchThresholds(const char* gcn, rcclTuningPartition partition) {
  if (gcn == nullptr) return nullptr;
  if (IsArchMatch(gcn, "gfx1250")) {
    // CPX is not tuned yet and uses the SPX table.
    return partition == RCCL_TUNING_PARTITION_DPX ? &rcclArchThresholds_gfx1250_dpx : &rcclArchThresholds_gfx1250;
  }
  if (IsArchMatch(gcn, "gfx950")) return &rcclArchThresholds_gfx950;
  if (IsArchMatch(gcn, "gfx942")) return &rcclArchThresholds_gfx942;
  return nullptr;
}
