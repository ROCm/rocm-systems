/*************************************************************************
 * Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

#ifndef _GIN_ANVIL_REACH_H_
#define _GIN_ANVIL_REACH_H_

#include <cstdint>
#include "nccl_device/gin/anvil_sdma/gin_anvil_ipc_table.h"

// Which peers the Anvil SDMA GIN backend can serve, per GPU generation: see src/gin/README.md.
enum class GinAnvilArch : int { Other = 0, Mi300 = 1, Mi350 = 2, Mi450 = 3 };

enum class GinAnvilSdmaReach : int {
  None = 0,       // some peer is beyond this GPU's SDMA engines
  Host = 1,       // every peer is on this host
  FabricPod = 2,  // peers span hosts, all inside one MI450 fabric pod
};

// fabricPod: comm's peers are fabric-mapped (ginAnvilFabricMemCapable), and for
// the agreed reach, every rank opted in to the pod.
inline GinAnvilSdmaReach ginAnvilSdmaReachPredicate(GinAnvilArch arch, int nHosts, int nRanks, bool fabricPod) {
  if (nHosts <= 1) return GinAnvilSdmaReach::Host;
  switch (arch) {
  case GinAnvilArch::Mi450:
    return fabricPod && nRanks <= NCCL_GIN_ANVIL_IPC_MAX_RANKS ? GinAnvilSdmaReach::FabricPod
                                                               : GinAnvilSdmaReach::None;
  case GinAnvilArch::Mi300:
  case GinAnvilArch::Mi350:
  case GinAnvilArch::Other:
  default:
    return GinAnvilSdmaReach::None;
  }
}

inline const char* ginAnvilArchName(GinAnvilArch arch) {
  switch (arch) {
  case GinAnvilArch::Mi300: return "MI300 (gfx942)";
  case GinAnvilArch::Mi350: return "MI350 (gfx950)";
  case GinAnvilArch::Mi450: return "MI450 (gfx1250)";
  default: return "this GPU";
  }
}

// Why the agreed reach is None, for the log line. podCapable is whether the
// topology alone would have allowed a pod (ginAnvilSdmaReachCapable).
inline const char* ginAnvilSdmaNoReachReason(GinAnvilArch arch, int nRanks, bool podCapable) {
  if (arch != GinAnvilArch::Mi450) return "SDMA reaches only GPUs on the same host";
  if (nRanks > NCCL_GIN_ANVIL_IPC_MAX_RANKS) return "the communicator exceeds Anvil's peer table size";
  if (podCapable) return "crossing hosts in a fabric pod needs RCCL_GIN_ANVIL_FABRIC_POD=1 and "
                         "RCCL_GIN_ANVIL_FABRIC_MEM=1 on every rank";
  return "SDMA crosses hosts only inside one fabric pod (MNNVL, a single clique, cuMem)";
}

struct ncclComm;

// Readers of comm state, defined in src/gin/gin_anvil_reach.cc.
GinAnvilArch ginAnvilArch(struct ncclComm* comm);

// Reach of comm by topology alone, as if every rank opted in to the pod. Every
// rank of comm gets the same answer.
GinAnvilSdmaReach ginAnvilSdmaReachCapable(struct ncclComm* comm);

// This rank's pod opt-in: ginAnvilSdmaReachCapable is FabricPod, and both
// RCCL_GIN_ANVIL_FABRIC_POD=1 and RCCL_GIN_ANVIL_FABRIC_MEM=1 are set. The
// opt-ins can differ between ranks, so ncclGinSetDefaultBackend agrees on it.
bool ginAnvilUseFabricPod(struct ncclComm* comm);

// True when comm could register Anvil SDMA windows through exported fabric VMM
// handles: MI450 with MNNVL, one clique covering comm, and cuMem. Depends on
// topology only, so every rank of comm gets the same answer.
bool ginAnvilFabricMemCapable(struct ncclComm* comm);

// ginAnvilFabricMemCapable plus this rank's RCCL_GIN_ANVIL_FABRIC_MEM=1 opt-in.
// The opt-in can differ between ranks, so the plugin agrees on it at connect.
bool ginAnvilUseFabricMem(struct ncclComm* comm);

// True when peer user windows are reachable via LSA-flat fabric mapping for this comm.
bool ginFabricLsaA2ACapable(struct ncclComm* comm, int ginNranks);

// This rank's RCCL_GIN_ANVIL_FABRIC_LSA_A2A=1 opt-in into the LSA AllToAll lane.
bool ginAnvilFabricLsaA2AOptIn();

// RCCL_GIN_ANVIL_FABRIC_LSA_THRESHOLD_ALLTOALL in bytes, or -1 when unset.
int64_t ginAnvilFabricLsaThresholdEnv();

#endif  // _GIN_ANVIL_REACH_H_
