/*************************************************************************
 * Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

#include "gin/gin_anvil_reach.h"

#include "archinfo.h"
#include "comm.h"
#include "cudawrap.h"
#include "param.h"

// Off by default until fabric registration is measured on MI455 silicon.
RCCL_PARAM(GinAnvilFabricMem, "GIN_ANVIL_FABRIC_MEM", 0);

static bool ginAnvilFabricMemPredicate(struct ncclComm* comm, bool optIn) {
  if (comm == nullptr) return false;
  const bool mnnvlMi450 = comm->MNNVL == 1 && IsArchMatch(comm->archName, "gfx1250");
  return optIn && mnnvlMi450 && comm->clique.size == comm->nRanks && ncclCuMemEnable();
}

bool ginAnvilFabricMemCapable(struct ncclComm* comm) { return ginAnvilFabricMemPredicate(comm, true); }

bool ginAnvilUseFabricMem(struct ncclComm* comm) {
  return ginAnvilFabricMemPredicate(comm, rcclParamGinAnvilFabricMem() != 0);
}
