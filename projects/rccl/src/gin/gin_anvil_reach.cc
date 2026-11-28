/*************************************************************************
 * Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

#include "gin/gin_anvil_reach.h"

#include "archinfo.h"
#include "comm.h"
#include "param.h"

// Off by default until fabric registration is measured on MI455 silicon.
RCCL_PARAM(GinAnvilFabricMem, "GIN_ANVIL_FABRIC_MEM", 0);

// comm->cuMemSupport, not ncclCuMemEnable(): cuMem is a per-process setting,
// and a caller may run a collective only where this is true, so every rank must
// agree on it.
static bool ginAnvilFabricMemPredicate(struct ncclComm* comm, bool optIn) {
  if (comm == nullptr) return false;
  const bool mnnvlMi450 = comm->MNNVL == 1 && IsArchMatch(comm->archName, "gfx1250");
  return optIn && mnnvlMi450 && comm->clique.size == comm->nRanks && comm->cuMemSupport;
}

bool ginAnvilFabricMemCapable(struct ncclComm* comm) { return ginAnvilFabricMemPredicate(comm, true); }

bool ginAnvilUseFabricMem(struct ncclComm* comm) {
  return ginAnvilFabricMemPredicate(comm, rcclParamGinAnvilFabricMem() != 0);
}
