/*************************************************************************
 * Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

#include "gin/gin_anvil_reach.h"

#include "archinfo.h"
#include "comm.h"
#include "cudawrap.h"
#include "nccl_device/gin/anvil_sdma/gin_fabric_lsa_policy.h"
#include "p2p.h"
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

bool ginFabricLsaA2ACapable(struct ncclComm* comm, int ginNranks) {
  if (comm == nullptr) return false;
  return gin::fabric::ginFabricLsaA2ACapablePredicate(ginAnvilUseFabricMem(comm), comm->MNNVL != 0,
                                                      ncclCuMemHandleType == CU_MEM_HANDLE_TYPE_FABRIC, ginNranks,
                                                      comm->nRanks, comm->devrState.lsaSize);
}
