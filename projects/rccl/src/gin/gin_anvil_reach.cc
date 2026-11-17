/*************************************************************************
 * Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

#include "gin/gin_anvil_reach.h"

#include "archinfo.h"
#include "comm.h"
#include "graph/topo.h"
#include "param.h"

// Off by default until fabric registration is measured on MI455 silicon.
RCCL_PARAM(GinAnvilFabricMem, "GIN_ANVIL_FABRIC_MEM", 0);
// Off by default until cross-host SDMA is measured in an MI455 pod.
RCCL_PARAM(GinAnvilFabricPod, "GIN_ANVIL_FABRIC_POD", 0);

GinAnvilArch ginAnvilArch(struct ncclComm* comm) {
  if (comm == nullptr) return GinAnvilArch::Other;
  if (IsArchMatch(comm->archName, "gfx942")) return GinAnvilArch::Mi300;
  if (IsArchMatch(comm->archName, "gfx950")) return GinAnvilArch::Mi350;
  if (IsArchMatch(comm->archName, "gfx1250")) return GinAnvilArch::Mi450;
  return GinAnvilArch::Other;
}

// comm->cuMemSupport, not ncclCuMemEnable(): cuMem is a per-process setting,
// and a caller may run a collective only where this is true, so every rank must
// agree on it.
static bool ginAnvilFabricMemPredicate(struct ncclComm* comm, bool optIn) {
  if (comm == nullptr) return false;
  const bool mnnvlMi450 = comm->MNNVL == 1 && ginAnvilArch(comm) == GinAnvilArch::Mi450;
  return optIn && mnnvlMi450 && comm->clique.size == comm->nRanks && comm->cuMemSupport;
}

bool ginAnvilFabricMemCapable(struct ncclComm* comm) { return ginAnvilFabricMemPredicate(comm, true); }

bool ginAnvilUseFabricMem(struct ncclComm* comm) {
  return ginAnvilFabricMemPredicate(comm, rcclParamGinAnvilFabricMem() != 0);
}

GinAnvilSdmaReach ginAnvilSdmaReachCapable(struct ncclComm* comm) {
  if (comm == nullptr || comm->topo == nullptr) return GinAnvilSdmaReach::None;
  return ginAnvilSdmaReachPredicate(ginAnvilArch(comm), comm->topo->nHosts, comm->nRanks,
                                    ginAnvilFabricMemCapable(comm));
}

bool ginAnvilUseFabricPod(struct ncclComm* comm) {
  return ginAnvilSdmaReachCapable(comm) == GinAnvilSdmaReach::FabricPod && rcclParamGinAnvilFabricPod() != 0 &&
         ginAnvilUseFabricMem(comm);
}
