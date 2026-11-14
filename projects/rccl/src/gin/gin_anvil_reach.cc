/*************************************************************************
 * Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

#include "gin/gin_anvil_reach.h"

#include "archinfo.h"
#include "comm.h"
#include "cudawrap.h"
#include "graph/topo.h"
#include "nccl_device/gin/anvil_sdma/gin_fabric_lsa_policy.h"
#include "p2p.h"
#include "param.h"

// Off by default until fabric registration is measured on MI455 silicon.
RCCL_PARAM(GinAnvilFabricMem, "GIN_ANVIL_FABRIC_MEM", 0);
RCCL_PARAM_NCCL_ALIAS(GinAnvilFabricLsaA2A, "GIN_ANVIL_FABRIC_LSA_A2A", 0);
RCCL_PARAM_NCCL_ALIAS(GinAnvilFabricLsaThresholdAlltoAll, "GIN_ANVIL_FABRIC_LSA_THRESHOLD_ALLTOALL", -1);
// Off by default until cross-host SDMA is measured in an MI455 pod.
RCCL_PARAM(GinAnvilFabricPod, "GIN_ANVIL_FABRIC_POD", 0);

GinAnvilArch ginAnvilArch(struct ncclComm* comm) {
  if (comm == nullptr) return GinAnvilArch::Other;
  if (IsArchMatch(comm->archName, "gfx942")) return GinAnvilArch::Mi300;
  if (IsArchMatch(comm->archName, "gfx950")) return GinAnvilArch::Mi350;
  if (IsArchMatch(comm->archName, "gfx1250")) return GinAnvilArch::Mi450;
  return GinAnvilArch::Other;
}

static bool ginAnvilFabricMemPredicate(struct ncclComm* comm, bool optIn) {
  if (comm == nullptr) return false;
  const bool mnnvlMi450 = comm->MNNVL == 1 && ginAnvilArch(comm) == GinAnvilArch::Mi450;
  return optIn && mnnvlMi450 && comm->clique.size == comm->nRanks && ncclCuMemEnable();
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

bool ginFabricLsaA2ACapable(struct ncclComm* comm, int ginNranks) {
  if (comm == nullptr) return false;
  return gin::fabric::ginFabricLsaA2ACapablePredicate(ginAnvilUseFabricMem(comm), comm->MNNVL != 0,
                                                      ncclCuMemHandleType == CU_MEM_HANDLE_TYPE_FABRIC, ginNranks,
                                                      comm->nRanks, comm->devrState.lsaSize);
}

bool ginAnvilFabricLsaA2AOptIn() { return rcclParamGinAnvilFabricLsaA2A() != 0; }

int64_t ginAnvilFabricLsaThresholdEnv() {
  const int64_t v = rcclParamGinAnvilFabricLsaThresholdAlltoAll();
  return v < 0 ? -1 : v;
}
