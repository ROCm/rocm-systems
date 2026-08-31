/*************************************************************************
 * Copyright (c) 2016-2022, NVIDIA CORPORATION. All rights reserved.
 * Modifications Copyright (c) 2019-2023 Advanced Micro Devices, Inc. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

#ifndef NCCL_GRAPH_H_
#define NCCL_GRAPH_H_

#include "nccl.h"
#include "device.h"
#include "os.h"
#include <limits.h>
#include <stdlib.h>
#include <ctype.h>
#include <stdio.h>
#include "gdrwrap.h"

ncclResult_t ncclTopoCudaPath(int cudaDev, char** path);

// Reduce a per-rank channel count to the global minimum across comm. Used on the
// grow path, which keeps a fresh sharedRes (owner==comm) and so skips the
// tpNChannels/tpP2pNChannels clamp; the arch-specific caps can otherwise leave
// ranks with different counts and break the ncclConnect exchange.
ncclResult_t ncclTopoReconcileGrowChannels(struct ncclComm* comm, int* value);

struct ncclTopoSystem;
// Build the topology
ncclResult_t ncclTopoGetSystem(struct ncclComm* comm, struct ncclTopoSystem** system, const char* dumpXmlFile = NULL);
ncclResult_t ncclTopoSortSystem(struct ncclTopoSystem* system);
ncclResult_t ncclTopoPrint(struct ncclTopoSystem* system);

ncclResult_t ncclTopoComputePaths(struct ncclTopoSystem* system, struct ncclComm* comm);
ncclResult_t ncclTopoCheckCrossNicSupport(bool* supported);
ncclResult_t ncclTopoCheckNicFused(struct ncclComm* comm, bool* fused);
void ncclTopoFree(struct ncclTopoSystem* system);
ncclResult_t ncclTopoTrimSystem(struct ncclTopoSystem* system, struct ncclComm* comm);
ncclResult_t ncclTopoComputeP2pChannels(struct ncclComm* comm);
ncclResult_t ncclTopoComputeP2pChannelsPerPeer(struct ncclComm* comm);
// Resolved NCCL_MAX_P2P_NCHANNELS (MAXCHANNELS when unset, else clamped user value).
int ncclMaxP2pNchannels();
// Upper bound for p2pnChannels: 64, MAXCHANNELS on single-node gfx1250, or the
// user's NCCL_MAX_P2P_NCHANNELS above that. Always a power of two.
int ncclP2pChannelsUpperBound(struct ncclComm* comm, bool* userOptedHigherOut);
ncclResult_t ncclTopoGetNvbGpus(struct ncclTopoSystem* system, int rank, int* nranks, int** ranks);
ncclResult_t ncclTopoPathAllNVLink(struct ncclTopoSystem* system, int* allNvLink);
ncclResult_t ncclTopoPathAllDirectNVLink(struct ncclTopoSystem* system, bool* allNvlinkConnected);
ncclResult_t ncclTopoComputeCommCPU(struct ncclComm* comm);

// Query topology
ncclResult_t ncclTopoGetNetDev(struct ncclComm* comm, int rank, struct ncclTopoGraph* graph, int channelId,
                               int peerRank, int64_t* id, int* dev, int* proxyRank);
ncclResult_t ncclTopoCheckP2p(struct ncclComm* comm, struct ncclTopoSystem* system, int rank1, int rank2, int* p2p,
                              int* read, int* intermediateRank, int* cudaP2p);
ncclResult_t ncclTopoCheckMNNVL(struct ncclComm* comm, struct ncclPeerInfo* info1, struct ncclPeerInfo* info2,
                                int* ret);
enum ncclTopoGdrMode {
  ncclTopoGdrModeDisable = 0,
  ncclTopoGdrModeDefault = 1,
  ncclTopoGdrModePci = 2,
  ncclTopoGdrModeNum = 3
};
ncclResult_t ncclTopoCheckGdr(struct ncclTopoSystem* topo, int rank, int64_t netId, int read,
                              enum ncclTopoGdrMode* gdrMode);
enum ncclTopoFlushType {
  ncclTopoFlushNone = 0,   // no flush needed
  ncclTopoFlushAlways = 1, // flush always needed
  ncclTopoFlushC2c = 2     // PCIe NIC and C2C sync path are unordered, flush is needed.
};
static inline uint32_t ncclGdcPinFlag(enum ncclTopoFlushType flush) {
  if (flush == ncclTopoFlushC2c && ncclGdrPinV2Available()) return GDR_PIN_FLAG_FORCE_PCIE;
  return GDR_PIN_FLAG_DEFAULT;
}
ncclResult_t ncclTopoNeedFlush(struct ncclComm* comm, int64_t netId, int netDev, int rank, bool netManaged,
                               enum ncclTopoFlushType* flush);
ncclResult_t ncclTopoGetMinNetBw(struct ncclTopoSystem* system, int rank, float* bw);
ncclResult_t ncclTopoIsGdrAvail(struct ncclTopoSystem* system, int rank, bool* avail);
ncclResult_t ncclTopoCheckNet(struct ncclTopoSystem* system, int rank1, int rank2, int* net);
int ncclPxnDisable(struct ncclComm* comm);
ncclResult_t ncclTopoGetPxnRanks(struct ncclComm* comm, int** intermediateRanks, int* nranks);
ncclResult_t ncclGetLocalCpu(struct ncclTopoSystem* system, int gpu, int* retCpu);

ncclResult_t ncclGetUserP2pLevel(int* level);

#define MAX_XGMI_INTER_GPUS 4
ncclResult_t ncclTopoGetIntraNetDev(struct ncclTopoSystem* system, int rank, struct ncclTopoGraph* graph, int channelId,
                                    int type, int64_t* id, int* dev);
ncclResult_t ncclTopoGetLinkType(struct ncclTopoSystem* system, int cudaDev1, int cudaDev2, bool* isXGMI,
                                 int maxInter = MAX_XGMI_INTER_GPUS, int nInter = 0, int* inter = nullptr);

// Find CPU affinity
ncclResult_t ncclTopoGetCpuAffinity(struct ncclTopoSystem* system, int rank, ncclAffinity* affinity);

#define NCCL_TOPO_CPU_ARCH_X86 1
#define NCCL_TOPO_CPU_ARCH_POWER 2
#define NCCL_TOPO_CPU_ARCH_ARM 3
#define NCCL_TOPO_CPU_ARCH_MIXED 4
#define NCCL_TOPO_CPU_VENDOR_INTEL 1
#define NCCL_TOPO_CPU_VENDOR_AMD 2
#define NCCL_TOPO_CPU_VENDOR_ZHAOXIN 3
#define NCCL_TOPO_CPU_VENDOR_MIXED 4
#define NCCL_TOPO_CPU_MODEL_INTEL_BDW 1
#define NCCL_TOPO_CPU_MODEL_INTEL_SKL 2
#define NCCL_TOPO_CPU_MODEL_INTEL_SRP 3
#define NCCL_TOPO_CPU_MODEL_INTEL_ERP 4
#define NCCL_TOPO_CPU_MODEL_AMD_ZEN 5
#define NCCL_TOPO_CPU_MODEL_AMD_ROME 6
#define NCCL_TOPO_CPU_MODEL_YONGFENG 1
ncclResult_t ncclTopoCpuType(struct ncclTopoSystem* system, int* arch, int* vendor, int* model);
ncclResult_t ncclTopoGetGpuCount(struct ncclTopoSystem* system, int* count);
ncclResult_t ncclTopoGetNetCount(struct ncclTopoSystem* system, int* count);
ncclResult_t ncclTopoGetNvsCount(struct ncclTopoSystem* system, int* count);
ncclResult_t ncclTopoGetLocalNet(struct ncclTopoSystem* system, int rank, int channelId, int64_t* id, int* dev);
ncclResult_t ncclTopoGetLocalGinDevs(struct ncclComm* comm, int* localGinDevs, int* localGinCount);
ncclResult_t ncclTopoGetLocalRmaDevs(struct ncclComm* comm, int* localRmaDevs, int* localRmaCount);
ncclResult_t ncclTopoGetLocalGpu(struct ncclTopoSystem* system, int64_t netId, int* gpuIndex);
ncclResult_t ncclTopoGetLocalNetCountByBw(struct ncclTopoSystem* system, int gpu, int* count, float* bw);

enum netDevsPolicy {
  NETDEVS_POLICY_AUTO = 0x0,
  NETDEVS_POLICY_ALL = 0x1,
  NETDEVS_POLICY_MAX = 0x2,
  NETDEVS_POLICY_UNDEF = 0xffffffff
};
ncclResult_t ncclTopoGetNetDevsPolicy(enum netDevsPolicy* policy, int* policyNum);

// Allows for up to 144 GPUs in scale-up domain (72 GPUs in DPX mode).
// [RCCL] Not raised to upstream's 640: ncclTopoGraph's RCCL-specific treeBase array is
// O(NCCL_TOPO_MAX_NODES^2), so 640 would blow sizeof(ncclComm) up to ~17.6 MiB. TODO: decouple.
#define NCCL_TOPO_MAX_NODES 144
ncclResult_t ncclTopoGetLocal(struct ncclTopoSystem* system, int type, int index, int resultType,
                              int locals[NCCL_TOPO_MAX_NODES], int* localCount, int* pathType);
ncclResult_t ncclTopoGetDevNodes(struct ncclTopoSystem* system, int64_t baseId, struct ncclTopoNode** nodes,
                                 int* nNodes);

// Local (myself)
#define PATH_LOC 0

// Connection traversing NVLink
#define PATH_NVL 1

// Connection through NVLink using an intermediate GPU
#define PATH_NVB 2

// Connection through C2C
#define PATH_C2C 3

// Connection traversing at most a single PCIe bridge
#define PATH_PIX 4

// Connection traversing multiple PCIe bridges (without traversing the PCIe Host Bridge)
#define PATH_PXB 5

// Connection between a GPU and a NIC using the C2C connection to the CPU and the PCIe connection to the NIC
#define PATH_P2C 6

// Connection between a GPU and a NIC using an intermediate GPU. Used to enable rail-local, aggregated network
// send/recv operations.
#define PATH_PXN 7

// Connection traversing PCIe as well as a PCIe Host Bridge (typically the CPU)
#define PATH_PHB 8

// Connection traversing PCIe as well as the SMP interconnect between NUMA nodes (e.g., QPI/UPI)
#define PATH_SYS 9

// Connection through the network
#define PATH_NET 10

// New type of path which should precede PATH_PIX
#define PATH_PORT PATH_NVL

// Disconnected
#define PATH_DIS 11
extern const char* topoPathTypeStr[];

// Init search. Needs to be done before calling ncclTopoCompute
ncclResult_t ncclTopoSearchInit(struct ncclTopoSystem* system);

#define NCCL_TOPO_PATTERN_BALANCED_TREE \
  1 // Spread NIC traffic between two GPUs (Tree parent + one child on first GPU, second child on second GPU)
#define NCCL_TOPO_PATTERN_SPLIT_TREE \
  2 // Spread NIC traffic between two GPUs (Tree parent on first GPU, tree children on the second GPU)
#define NCCL_TOPO_PATTERN_TREE 3 // All NIC traffic going to/from the same GPU
#define NCCL_TOPO_PATTERN_RING 4 // Ring
#define NCCL_TOPO_PATTERN_NVLS 5 // NVLS+SHARP and NVLS+Tree
#define NCCL_TOPO_PATTERN_COLLNET_DIRECT 6 // Collnet Direct
struct ncclTopoGraph {
  // Input / output
  int id; // ring : 0, tree : 1, collnet : 2, nvls : 3, collnetDirect : 4
  int pattern;
  int crossNic;
  int collNet;
  int minChannels;
  int maxChannels;
  // Output
  int nChannels;
  float bwIntra;
  float bwInter;
  float latencyInter;
  int typeIntra;
  int typeInter;
  int sameChannels;
  int nHops;
  int intra[MAXCHANNELS * NCCL_TOPO_MAX_NODES];
  int64_t inter[MAXCHANNELS * 2];
  int nIntraChannels;
  int intraNets[MAXCHANNELS * NCCL_TOPO_MAX_NODES * 2];
  char treeBase[NCCL_TOPO_MAX_NODES][NCCL_TOPO_MAX_NODES * 4];
};
ncclResult_t ncclTopoCompute(struct ncclTopoSystem* system, struct ncclTopoGraph* graph);

ncclResult_t ncclTopoPrintGraph(struct ncclTopoSystem* system, struct ncclTopoGraph* graph);
ncclResult_t ncclTopoDumpGraphs(struct ncclTopoSystem* system, int ngraphs, struct ncclTopoGraph** graphs);

struct ncclTopoRanks {
  int crossNicRing;
  int ringRecv[MAXCHANNELS];
  int ringSend[MAXCHANNELS];
  int ringPrev[MAXCHANNELS];
  int ringNext[MAXCHANNELS];
  int treeToParent[MAXCHANNELS];
  int treeToChild0[MAXCHANNELS];
  int treeToChild1[MAXCHANNELS];
  int nvlsHeads[MAXCHANNELS];
  int nvlsHeadNum;
};

ncclResult_t ncclTopoPreset(struct ncclComm* comm, struct ncclTopoGraph* (&graphs)[NCCL_NUM_ALGORITHMS],
                            struct ncclTopoRanks* topoRanks);

ncclResult_t ncclTopoPostset(struct ncclComm* comm, int* firstRanks, int* treePatterns,
                             struct ncclTopoRanks** allTopoRanks, int* rings, struct ncclTopoGraph** graphs,
                             struct ncclComm* parent, int nc);
ncclResult_t ncclTreeBasePostset(struct ncclComm* comm, struct ncclTopoGraph* treeGraph);

ncclResult_t ncclTopoInitTunerConstants(struct ncclComm* comm);
ncclResult_t ncclTopoTuneModel(struct ncclComm* comm, int minCompCap, int maxCompCap, struct ncclTopoGraph** graphs);
ncclResult_t ncclTopoGetAlgoTime(struct ncclComm* comm, int coll, int algorithm, int protocol, size_t nBytes,
                                 int numPipeOps, float* time);
// Per-architecture DDA/CE/Ring dispatch threshold table.
// DDA arrays are indexed by ncclFunc_t through AlltoAll (ncclFuncAlltoAll == 8).
// AR/AG/A2A compare total message bytes; RS compares rsShardBytes (per-rank).
// 0 disables that tier for that collective. No parallel built-in defaults:
// an unset env var uses this table, and an arch with no table gets 0.
enum { RCCL_DDA_FUNC_COUNT = ncclFuncAlltoAll + 1 };
struct rcclArchThresholds {
  // DDA tier upper bounds, per collective.  gfx1250 uses fabric LL/LL128/VMM;
  // gfx942/gfx950 use ddaVmmMax as the DDA-IPC cap (LL/LL128 unused, stay 0).
  // All arrays are indexed by ncclFunc_t; 0 disables that tier for that coll.
  // AR/AG/AlltoAll compare total message bytes; RS compares per-rank shard bytes.
  size_t ddaLLMax[RCCL_DDA_FUNC_COUNT];     // DDA LL tier:    0 .. ddaLLMax[func]
  size_t ddaLL128Max[RCCL_DDA_FUNC_COUNT];  // DDA LL128 tier: ddaLLMax[func]+1 .. ddaLL128Max[func]
  size_t ddaVmmMax[RCCL_DDA_FUNC_COUNT];    // DDA VMM/IPC cap: ddaLL128Max[func]+1 .. ddaVmmMax[func]

  // R2 variant: DDA VMM cap when recv buffer is registered (winRegType !=
  // ncclSymSendNonregRecvNonreg).  On gfx1250, RS fires DDA even when
  // symEligible=true; lowering this cap for R2 lets CE win at smaller sizes.
  // 0 means "use ddaVmmMax" (same as R0 behavior) -- only RS needs an override.
  size_t ddaVmmMaxR2[RCCL_DDA_FUNC_COUNT];

  // Graph-mode variant: DDA VMM cap to use when the comm is inside a graph
  // capture (graphCapturingHint=true).  CE AllReduce is blocked during graph
  // captures, so DDA can fill the full window that CE would normally absorb.
  // 0 means "use ddaVmmMax" (no graph-specific override).
  size_t ddaVmmMaxGraph[RCCL_DDA_FUNC_COUNT];

  // CE non-registered (scratch) window per collective, total bytes.
  // CE-Scratch fires when ceNonRegMin[func] <= totalBytes <= ceNonRegMax[func].
  // ceNonRegMin[func] = 0 means no lower bound (fires immediately above DDA exit).
  // ceNonRegMax[func] = 0 means CE-Scratch / CE 2-shot disabled for that collective.
  // AllReduce 2-shot: this is a selector cap only. ceARTmpBuf is allocated at
  // NCCL_CE_AR_TMPBUF_DEFAULT_BYTES and grown only when this entry (or
  // RCCL_CE_AR_MAX_MSG_BYTES) is larger.
  size_t ceNonRegMin[RCCL_DDA_FUNC_COUNT];
  size_t ceNonRegMax[RCCL_DDA_FUNC_COUNT];

  // CE registered-window upper bound per collective, total bytes.
  // For AllReduce: registered CE copies through user symmetric windows (tuning cap only).
  // For AllGather: upper bound for CE-registered (R=2). 0 means no upper bound.
  size_t ceRegMax[RCCL_DDA_FUNC_COUNT];

  // Symmetric kernel upper-bound per collective when recv buffer is registered (R2).
  // Above this size CE is faster than symk; setting this withdraws symk as the final
  // choice in rcclSelectAllReduce so CE-registered can win. It does not unblock the
  // CE 2-shot or DDA branches, which stay gated on whether symk was requested at all.
  // 0 means no suppression (symk may win at any size for that collective).
  // Only AllReduce is relevant today; other collectives default to 0.
  size_t symMaxR2[RCCL_DDA_FUNC_COUNT];
  // Graph-mode variant of symMaxR2 (graphCapturingHint / stream capture). CE is
  // graph-unsafe, so graph mode typically wants no suppression (keep symk).
  // 0 means no suppression, same convention as symMaxR2 -- not "inherit eager".
  size_t symMaxR2Graph[RCCL_DDA_FUNC_COUNT];

  // Per-size unroll factor breakpoints for gfx1250.  Each entry is a
  // (maxBytes, unrollIdx) pair: the first entry whose maxBytes >= msgBytes
  // wins.  A terminal entry with maxBytes == SIZE_MAX covers everything larger.
  // unrollIdx values: NCCL_UNROLL_1=0, NCCL_UNROLL_2=1, NCCL_UNROLL_4=2,
  //                   NCCL_UNROLL_8=3, NCCL_UNROLL_16=4, NCCL_UNROLL_32=5.
  // Null pointer means "keep the comm-level default (gfx1250 default = UNROLL_32)".
  struct rcclUnrollEntry { size_t maxBytes; int unrollIdx; };
  const rcclUnrollEntry* unrollMapAR;   // AllReduce per-size unroll breakpoints
  const rcclUnrollEntry* unrollMapAG;   // AllGather per-size unroll breakpoints
  const rcclUnrollEntry* unrollMapRS;   // ReduceScatter per-size unroll breakpoints
  const rcclUnrollEntry* unrollMapA2A;  // AlltoAll per-size unroll breakpoints

};
const rcclArchThresholds* rcclGetArchThresholds(const char* gcn);
int rcclGetTuningIndexForArch(const char* gfxarch);
#endif
