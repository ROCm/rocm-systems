/*************************************************************************
 * Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

// Fail-loud stub floor for the topology subsystem, shared by host-only
// microtests. These satisfy a unit-under-test's link-time symbol closure; the
// shallower tests never call them (abort-on-call). A test that needs to drive
// one of these replaces that individual entry with a real fake.

#include <cstdlib>
#include <functional>
#include <string>
#include <vector>
#include <sched.h>

#include "nccl.h"
#include "os.h"   // ncclAffinity

struct ncclComm;
struct ncclTopoGraph;
struct ncclTopoRanks;
struct ncclTopoSystem;

// Controllable (was fail-loud). :1982; the value reaches the AllGather3 payload, so the default is deterministic.
extern std::function<ncclResult_t(struct ncclComm*, bool*)> g_ncclTopoCheckNicFused;
ncclResult_t ncclTopoCheckNicFused(struct ncclComm* comm, bool* fused) { return g_ncclTopoCheckNicFused(comm, fused); }
// Controllable (was fail-loud). Defaults to FAILURE: this is the rung-2 ladder terminator at :1648. Its
// four later call sites (:1673, :1690, :1692, :1702 -- the tree/CollNet/NVLS graphs) are driven by the
// rung-3 tests. Records the graph pointer: without it, :1648 being handed treeGraph instead of
// ringGraph is invisible, the same "a fake that drops an argument untests it" rule as nccl_stubs.cc:90.
// A std::function rather than a result knob: rung 3 needs it to SUCCEED and write graph->nChannels,
// which :1671-1672 then read to size the tree graph. Default still fails, so rung 2 is unchanged.
extern std::function<ncclResult_t(struct ncclTopoSystem*, struct ncclTopoGraph*)> g_ncclTopoCompute;
extern int g_ncclTopoComputeCalls;
extern std::vector<struct ncclTopoGraph*> g_ncclTopoComputeGraphs;
ncclResult_t ncclTopoCompute(struct ncclTopoSystem* system, struct ncclTopoGraph* graph) {
  g_ncclTopoComputeCalls++;
  g_ncclTopoComputeGraphs.push_back(graph);
  return g_ncclTopoCompute(system, graph);
}
extern ncclResult_t g_ncclTopoComputeCommCPUResult;
ncclResult_t ncclTopoComputeCommCPU(struct ncclComm* comm) { return g_ncclTopoComputeCommCPUResult; }
ncclResult_t ncclTopoComputeP2pChannels(struct ncclComm* comm) { ::abort(); }
// Controllable (was fail-loud). Rung-3 terminator at :1774. Its sentinel is ncclTimeout rather than the
// ncclRemoteError the earlier rungs share, so a rung-3 test proves it cleared the whole graph block
// instead of having stopped at :1648 with ncclTopoCompute left un-armed. Nothing reachable from
// initTransportsRank produces ncclTimeout (init.cc:4388 is a ncclGetErrorString case).
extern ncclResult_t g_ncclTopoComputeP2pChannelsPerPeerResult;
ncclResult_t ncclTopoComputeP2pChannelsPerPeer(struct ncclComm* comm) {
  return g_ncclTopoComputeP2pChannelsPerPeerResult;
}
// Called TWICE (:1591 pre-trim, :1596 post-trim), so a plain result knob cannot tell them apart --
// FailAt selects which call fails, the way g_callocFailAt selects which allocation does.
extern int g_ncclTopoComputePathsCalls;
extern int g_ncclTopoComputePathsFailAt;  // -1 = never fail
ncclResult_t ncclTopoComputePaths(struct ncclTopoSystem* system, struct ncclComm* comm) {
  return g_ncclTopoComputePathsCalls++ == g_ncclTopoComputePathsFailAt ? ncclSystemError : ncclSuccess;
}
// Controllable (was fail-loud). :1764, gated on comm->rank == NCCL_GRAPH_DUMP_FILE_RANK. Records ngraphs
// because the call site hardcodes 5 -- a count that must track the five graphs actually computed above.
extern ncclResult_t g_ncclTopoDumpGraphsResult;
extern int g_ncclTopoDumpGraphsCalls;
extern int g_ncclTopoDumpGraphsNgraphs;
extern std::vector<struct ncclTopoGraph*> g_ncclTopoDumpGraphsArray;
ncclResult_t ncclTopoDumpGraphs(struct ncclTopoSystem* system, int ngraphs, struct ncclTopoGraph** graphs) {
  g_ncclTopoDumpGraphsCalls++;
  g_ncclTopoDumpGraphsNgraphs = ngraphs;  // assert this, not the vector length, if :1764's 5 ever moves
  // Clamp the read: the caller's array is dumpGraphs[kDumpGraphsCapacity] at :1757, so trusting a
  // larger ngraphs would read off its stack rather than failing an assertion.
  const int kDumpGraphsCapacity = 5;
  const int n = ngraphs < kDumpGraphsCapacity ? ngraphs : kDumpGraphsCapacity;
  g_ncclTopoDumpGraphsArray.assign(graphs, graphs + n);  // :1763 orders direct BEFORE chain
  return g_ncclTopoDumpGraphsResult;
}
void ncclTopoFree(struct ncclTopoSystem* system) { ::abort(); }
// Controllable (was fail-loud). A std::function: :1608-1610 branch on the WRITTEN mask, and exit::2404
// forwards it, so a result-only seam could drive neither. Records rank so :1607 passing comm->rank is visible.
extern std::function<ncclResult_t(struct ncclTopoSystem*, int, ncclAffinity*)> g_ncclTopoGetCpuAffinity;
extern int g_ncclTopoGetCpuAffinityLastRank;
ncclResult_t ncclTopoGetCpuAffinity(struct ncclTopoSystem* system, int rank, ncclAffinity* affinity) {
  g_ncclTopoGetCpuAffinityLastRank = rank;
  return g_ncclTopoGetCpuAffinity(system, rank, affinity);
}
// Controllable (was fail-loud). :1983 and :1953; both feed the AllGather3 payload.
extern std::function<ncclResult_t(struct ncclTopoSystem*, int, float*)> g_ncclTopoGetMinNetBw;
ncclResult_t ncclTopoGetMinNetBw(struct ncclTopoSystem* system, int rank, float* bw) {
  return g_ncclTopoGetMinNetBw(system, rank, bw);
}
extern std::function<ncclResult_t(struct ncclTopoSystem*, int, int*, float*)> g_ncclTopoGetLocalNetCountByBw;
ncclResult_t ncclTopoGetLocalNetCountByBw(struct ncclTopoSystem* system, int gpu, int* count, float* bw) {
  return g_ncclTopoGetLocalNetCountByBw(system, gpu, count, bw);
}
ncclResult_t ncclTopoGetNvbGpus(struct ncclTopoSystem* system, int rank, int* nranks, int** ranks) { ::abort(); }
ncclResult_t ncclTopoGetPxnRanks(struct ncclComm* comm, int** intermediateRanks, int* nranks) { ::abort(); }
// Controllable (was fail-loud). This is the FIRST call after initTransportsRank's MNNVL/intra-proc block, so arming it
// to fail terminates the error-injection ladder and makes :1462-1565 coverable. Default stays failure because both call
// sites (:1573, :1576) are on paths no test drives to success yet -- MICROTEST_README.md, "Adding more controllable
// seams". Records dumpXmlFile so :1573 vs :1576 is visible.
extern std::function<ncclResult_t(struct ncclComm*, struct ncclTopoSystem**, const char*)> g_ncclTopoGetSystem;
ncclResult_t ncclTopoGetSystem(struct ncclComm* comm, struct ncclTopoSystem** system, const char* dumpXmlFile) {
  return g_ncclTopoGetSystem(comm, system, dumpXmlFile);
}
ncclResult_t ncclTopoInitTunerConstants(struct ncclComm* comm) { ::abort(); }
ncclResult_t ncclTopoPathAllDirectNVLink(struct ncclTopoSystem* system, bool* allNvlinkConnected) { ::abort(); }
// Controllable (was fail-loud). :1985 writes comm->isAllNvlink, which :2037 then folds across ranks.
extern std::function<ncclResult_t(struct ncclTopoSystem*, int*)> g_ncclTopoPathAllNVLink;
ncclResult_t ncclTopoPathAllNVLink(struct ncclTopoSystem* system, int* allNvLink) {
  return g_ncclTopoPathAllNVLink(system, allNvLink);
}
extern ncclResult_t g_ncclTopoPrintResult;
ncclResult_t ncclTopoPrint(struct ncclTopoSystem* system) { return g_ncclTopoPrintResult; }
// Controllable (was fail-loud). Five call sites (:1649, :1674, :1691, :1693, :1703), each paired with an
// ncclTopoCompute. Records the graph so a swapped pair -- printing the tree graph after computing the
// ring one -- is visible; a result-only seam would not see it.
extern ncclResult_t g_ncclTopoPrintGraphResult;
extern std::vector<struct ncclTopoGraph*> g_ncclTopoPrintGraphGraphs;
ncclResult_t ncclTopoPrintGraph(struct ncclTopoSystem* system, struct ncclTopoGraph* graph) {
  g_ncclTopoPrintGraphGraphs.push_back(graph);
  return g_ncclTopoPrintGraphResult;
}
extern ncclResult_t g_ncclTopoSearchInitResult;
ncclResult_t ncclTopoSearchInit(struct ncclTopoSystem* system) { return g_ncclTopoSearchInitResult; }
extern ncclResult_t g_ncclTopoTrimSystemResult;
ncclResult_t ncclTopoTrimSystem(struct ncclTopoSystem* system, struct ncclComm* comm) {
  return g_ncclTopoTrimSystemResult;
}
ncclResult_t ncclTopoTuneModel(struct ncclComm* comm, int minCompCap, int maxCompCap, struct ncclTopoGraph** graphs) { ::abort(); }
// Controllable (was fail-loud). Rung-4 terminator at :2213; records nc, the :2163 min that nothing on comm exposes.
extern ncclResult_t g_ncclTopoPostsetResult;
extern int g_ncclTopoPostsetCalls;
extern int g_ncclTopoPostsetNc;
ncclResult_t ncclTopoPostset(struct ncclComm*, int*, int*, struct ncclTopoRanks**, int*, struct ncclTopoGraph**,
                             struct ncclComm*, int nc) {
  g_ncclTopoPostsetCalls++;
  g_ncclTopoPostsetNc = nc;
  return g_ncclTopoPostsetResult;
}
// Controllable (was fail-loud). :1994 fills this rank's topoRanks slot in the AllGather3 payload.
extern std::function<ncclResult_t(struct ncclComm*, struct ncclTopoRanks*)> g_ncclTopoPreset;
extern int g_ncclTopoPresetCalls;
ncclResult_t ncclTopoPreset(struct ncclComm* comm, struct ncclTopoGraph* (&)[7], struct ncclTopoRanks* topoRanks) {
  g_ncclTopoPresetCalls++;
  return g_ncclTopoPreset(comm, topoRanks);
}
// Controllable (was fail-loud). :1999, gated on uniformRanksPerHost.
extern ncclResult_t g_rcclCheckRomeTopoModelIdxConsensusResult;
extern int g_rcclCheckRomeTopoModelIdxConsensusCalls;
extern int g_rcclRomeConsensusNranks;
extern int g_rcclRomeConsensusIdx0;
extern std::string g_rcclRomeConsensusHost0;
ncclResult_t rcclCheckRomeTopoModelIdxConsensus(int nranks, std::function<int(int)> modelIdx,
                                                std::function<const char*(int)> hostname,
                                                std::function<unsigned long(int)>) {
  g_rcclCheckRomeTopoModelIdxConsensusCalls++;
  g_rcclRomeConsensusNranks = nranks;
  if (nranks > 0) {
    g_rcclRomeConsensusIdx0 = modelIdx(0);
    g_rcclRomeConsensusHost0 = hostname(0);
  }
  return g_rcclCheckRomeTopoModelIdxConsensusResult;
}
