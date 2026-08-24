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
#include <sched.h>

#include "nccl.h"
#include "os.h"   // ncclAffinity

struct ncclComm;
struct ncclTopoGraph;
struct ncclTopoRanks;
struct ncclTopoSystem;

ncclResult_t ncclTopoCheckNicFused(struct ncclComm* comm, bool* fused) { ::abort(); }
// Controllable (was fail-loud). Defaults to FAILURE: this is the phase-3/4/5 ladder terminator at :1648,
// and its four later call sites (:1649 onward, the tree/NVLS graphs) are on paths no test drives yet.
extern ncclResult_t g_ncclTopoComputeResult;
extern int g_ncclTopoComputeCalls;
ncclResult_t ncclTopoCompute(struct ncclTopoSystem* system, struct ncclTopoGraph* graph) {
  g_ncclTopoComputeCalls++;
  return g_ncclTopoComputeResult;
}
extern ncclResult_t g_ncclTopoComputeCommCPUResult;
ncclResult_t ncclTopoComputeCommCPU(struct ncclComm* comm) { return g_ncclTopoComputeCommCPUResult; }
ncclResult_t ncclTopoComputeP2pChannels(struct ncclComm* comm) { ::abort(); }
ncclResult_t ncclTopoComputeP2pChannelsPerPeer(struct ncclComm* comm) { ::abort(); }
// Called TWICE (:1591 pre-trim, :1596 post-trim), so a plain result knob cannot tell them apart --
// FailAt selects which call fails, the way g_callocFailAt selects which allocation does.
extern int g_ncclTopoComputePathsCalls;
extern int g_ncclTopoComputePathsFailAt;  // -1 = never fail
ncclResult_t ncclTopoComputePaths(struct ncclTopoSystem* system, struct ncclComm* comm) {
  return g_ncclTopoComputePathsCalls++ == g_ncclTopoComputePathsFailAt ? ncclSystemError : ncclSuccess;
}
ncclResult_t ncclTopoDumpGraphs(struct ncclTopoSystem* system, int ngraphs, struct ncclTopoGraph** graphs) { ::abort(); }
void ncclTopoFree(struct ncclTopoSystem* system) { ::abort(); }
// Controllable (was fail-loud). A std::function: :1608-1610 branch on the WRITTEN mask, and exit::2404
// forwards it, so a result-only seam could drive neither. Records rank so :1607 passing comm->rank is visible.
extern std::function<ncclResult_t(struct ncclTopoSystem*, int, ncclAffinity*)> g_ncclTopoGetCpuAffinity;
extern int g_ncclTopoGetCpuAffinityLastRank;
ncclResult_t ncclTopoGetCpuAffinity(struct ncclTopoSystem* system, int rank, ncclAffinity* affinity) {
  g_ncclTopoGetCpuAffinityLastRank = rank;
  return g_ncclTopoGetCpuAffinity(system, rank, affinity);
}
ncclResult_t ncclTopoGetMinNetBw(struct ncclTopoSystem* system, int rank, float* bw) { ::abort(); }
ncclResult_t ncclTopoGetLocalNetCountByBw(struct ncclTopoSystem* system, int gpu, int* count, float* bw) { ::abort(); }
ncclResult_t ncclTopoGetNvbGpus(struct ncclTopoSystem* system, int rank, int* nranks, int** ranks) { ::abort(); }
ncclResult_t ncclTopoGetPxnRanks(struct ncclComm* comm, int** intermediateRanks, int* nranks) { ::abort(); }
// Controllable (was fail-loud). This is the FIRST call after initTransportsRank's MNNVL/intra-proc block, so arming it
// to fail terminates the error-injection ladder and makes :1462-1565 coverable. Default stays failure: both call sites
// (:1573, :1576) are on paths no test drives to success yet (seams.md 2). Records dumpXmlFile so :1573 vs :1576 is visible.
extern std::function<ncclResult_t(struct ncclComm*, struct ncclTopoSystem**, const char*)> g_ncclTopoGetSystem;
ncclResult_t ncclTopoGetSystem(struct ncclComm* comm, struct ncclTopoSystem** system, const char* dumpXmlFile) {
  return g_ncclTopoGetSystem(comm, system, dumpXmlFile);
}
ncclResult_t ncclTopoInitTunerConstants(struct ncclComm* comm) { ::abort(); }
ncclResult_t ncclTopoPathAllDirectNVLink(struct ncclTopoSystem* system, bool* allNvlinkConnected) { ::abort(); }
ncclResult_t ncclTopoPathAllNVLink(struct ncclTopoSystem* system, int* allNvLink) { ::abort(); }
extern ncclResult_t g_ncclTopoPrintResult;
ncclResult_t ncclTopoPrint(struct ncclTopoSystem* system) { return g_ncclTopoPrintResult; }
ncclResult_t ncclTopoPrintGraph(struct ncclTopoSystem* system, struct ncclTopoGraph* graph) { ::abort(); }
extern ncclResult_t g_ncclTopoSearchInitResult;
ncclResult_t ncclTopoSearchInit(struct ncclTopoSystem* system) { return g_ncclTopoSearchInitResult; }
extern ncclResult_t g_ncclTopoTrimSystemResult;
ncclResult_t ncclTopoTrimSystem(struct ncclTopoSystem* system, struct ncclComm* comm) { return g_ncclTopoTrimSystemResult; }
ncclResult_t ncclTopoTuneModel(struct ncclComm* comm, int minCompCap, int maxCompCap, struct ncclTopoGraph** graphs) { ::abort(); }
ncclResult_t ncclTopoPostset(struct ncclComm*, int*, int*, struct ncclTopoRanks**, int*, struct ncclTopoGraph**, struct ncclComm*, int) { ::abort(); }
ncclResult_t ncclTopoPreset(struct ncclComm*, struct ncclTopoGraph* (&)[7], struct ncclTopoRanks*) { ::abort(); }
ncclResult_t rcclCheckRomeTopoModelIdxConsensus(int, std::function<int(int)>,
                                                std::function<const char*(int)>,
                                                std::function<unsigned long(int)>) { ::abort(); }
