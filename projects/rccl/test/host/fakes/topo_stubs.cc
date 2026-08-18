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
ncclResult_t ncclTopoCompute(struct ncclTopoSystem* system, struct ncclTopoGraph* graph) { ::abort(); }
ncclResult_t ncclTopoComputeCommCPU(struct ncclComm* comm) { ::abort(); }
ncclResult_t ncclTopoComputeP2pChannels(struct ncclComm* comm) { ::abort(); }
ncclResult_t ncclTopoComputeP2pChannelsPerPeer(struct ncclComm* comm) { ::abort(); }
ncclResult_t ncclTopoComputePaths(struct ncclTopoSystem* system, struct ncclComm* comm) { ::abort(); }
ncclResult_t ncclTopoDumpGraphs(struct ncclTopoSystem* system, int ngraphs, struct ncclTopoGraph** graphs) { ::abort(); }
void ncclTopoFree(struct ncclTopoSystem* system) { ::abort(); }
ncclResult_t ncclTopoGetCpuAffinity(struct ncclTopoSystem* system, int rank, ncclAffinity* affinity) { ::abort(); }
ncclResult_t ncclTopoGetMinNetBw(struct ncclTopoSystem* system, int rank, float* bw) { ::abort(); }
ncclResult_t ncclTopoGetLocalNetCountByBw(struct ncclTopoSystem* system, int gpu, int* count, float* bw) { ::abort(); }
ncclResult_t ncclTopoGetNvbGpus(struct ncclTopoSystem* system, int rank, int* nranks, int** ranks) { ::abort(); }
ncclResult_t ncclTopoGetPxnRanks(struct ncclComm* comm, int** intermediateRanks, int* nranks) { ::abort(); }
ncclResult_t ncclTopoGetSystem(struct ncclComm* comm, struct ncclTopoSystem** system, const char* dumpXmlFile) { ::abort(); }
ncclResult_t ncclTopoInitTunerConstants(struct ncclComm* comm) { ::abort(); }
ncclResult_t ncclTopoPathAllDirectNVLink(struct ncclTopoSystem* system, bool* allNvlinkConnected) { ::abort(); }
ncclResult_t ncclTopoPathAllNVLink(struct ncclTopoSystem* system, int* allNvLink) { ::abort(); }
ncclResult_t ncclTopoPrint(struct ncclTopoSystem* system) { ::abort(); }
ncclResult_t ncclTopoPrintGraph(struct ncclTopoSystem* system, struct ncclTopoGraph* graph) { ::abort(); }
ncclResult_t ncclTopoSearchInit(struct ncclTopoSystem* system) { ::abort(); }
ncclResult_t ncclTopoTrimSystem(struct ncclTopoSystem* system, struct ncclComm* comm) { ::abort(); }
ncclResult_t ncclTopoTuneModel(struct ncclComm* comm, int minCompCap, int maxCompCap, struct ncclTopoGraph** graphs) { ::abort(); }
ncclResult_t ncclTopoPostset(struct ncclComm*, int*, int*, struct ncclTopoRanks**, int*, struct ncclTopoGraph**, struct ncclComm*, int) { ::abort(); }
ncclResult_t ncclTopoPreset(struct ncclComm*, struct ncclTopoGraph* (&)[7], struct ncclTopoRanks*) { ::abort(); }
ncclResult_t rcclCheckRomeTopoModelIdxConsensus(int, std::function<int(int)>,
                                                std::function<const char*(int)>,
                                                std::function<unsigned long(int)>) { ::abort(); }
