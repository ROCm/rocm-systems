/*************************************************************************
 * Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

#ifndef _GIN_ANVIL_REACH_H_
#define _GIN_ANVIL_REACH_H_

// Ranks in Anvil's peer table (NCCL_GIN_ANVIL_IPC_MAX_RANKS), which bounds a
// communicator that crosses hosts, not the pod. Restated because this header is
// in every build and gin_anvil_ipc_table.h only in an ENABLE_ROCSHMEM_GIN one;
// gin_plugin_anvil_sdma.cc checks the two agree.
constexpr int kGinAnvilIpcMaxRanks = 16;

// Which peers the Anvil SDMA GIN backend can serve: see src/gin/README.md.
enum class GinAnvilSdmaReach : int {
  None = 0,       // some peer is beyond this GPU's SDMA engines
  Host = 1,       // every peer is on this host
  FabricPod = 2,  // peers span hosts, all fabric-mapped in one pod
};

// fabricPod: comm's peers are fabric-mapped (ginAnvilFabricMemCapable), and for
// the agreed reach, every rank opted in to the pod.
inline GinAnvilSdmaReach ginAnvilSdmaReachPredicate(int nHosts, int nRanks, bool fabricPod) {
  if (nHosts <= 1) return GinAnvilSdmaReach::Host;
  return fabricPod && nRanks <= kGinAnvilIpcMaxRanks ? GinAnvilSdmaReach::FabricPod : GinAnvilSdmaReach::None;
}

// Why the agreed reach is None on a multi-host communicator, for the log line.
// fabricCapable is ginAnvilFabricMemCapable.
inline const char* ginAnvilSdmaNoReachReason(bool fabricCapable, int nRanks) {
  if (!fabricCapable) return "SDMA crosses hosts only inside one fabric pod (MNNVL, a single clique, cuMem)";
  if (nRanks > kGinAnvilIpcMaxRanks) return "the communicator has more ranks than Anvil's peer table";
  return "crossing hosts in a fabric pod needs RCCL_GIN_ANVIL_FABRIC_POD=1 and RCCL_GIN_ANVIL_FABRIC_MEM=1 on "
         "every rank";
}

struct ncclComm;

// Readers of comm state, defined in src/gin/gin_anvil_reach.cc.

// Reach of comm by topology alone, as if every rank opted in to the pod. Every
// rank of comm gets the same answer.
GinAnvilSdmaReach ginAnvilSdmaReachCapable(struct ncclComm* comm);

// This rank's pod opt-in: ginAnvilSdmaReachCapable is FabricPod, and both
// RCCL_GIN_ANVIL_FABRIC_POD=1 and RCCL_GIN_ANVIL_FABRIC_MEM=1 are set. The
// opt-ins can differ between ranks, so ncclGinSetDefaultBackend agrees on it.
bool ginAnvilUseFabricPod(struct ncclComm* comm);

// True when comm could register Anvil SDMA windows through exported fabric VMM
// handles: MI450 with MNNVL, one clique covering comm, and cuMem on every rank.
// Reads only state that the ranks agreed on at init, so every rank of comm gets
// the same answer.
bool ginAnvilFabricMemCapable(struct ncclComm* comm);

// ginAnvilFabricMemCapable plus this rank's RCCL_GIN_ANVIL_FABRIC_MEM=1 opt-in.
// The opt-in can differ between ranks, so the plugin agrees on it at connect.
bool ginAnvilUseFabricMem(struct ncclComm* comm);

#endif  // _GIN_ANVIL_REACH_H_
