/*************************************************************************
 * Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

// Controllable seams for the entry points src/graph/search.cc DEFINES, for
// targets that do not compile that file. See fakes/search_fakes.h for why this
// is split out of topo_stubs.cc.
//
// Mostly _fakes rather than _stubs: every entry here is a driven seam with a
// deterministic default, not a fail-loud floor.

#include <functional>
#include <vector>

#include "fakes/search_fakes.h"
#include "fakes/signature-drift.h"
#include "comm.h"
#include "graph.h"
#include "graph/topo.h"

// Controllable (was fail-loud). init.cc:1982; the value reaches the AllGather3
// payload, so the default is deterministic.
std::function<ncclResult_t(struct ncclComm*, bool*)> g_ncclTopoCheckNicFused =
    [](struct ncclComm*, bool* fused) { *fused = false; return ncclSuccess; };
ncclResult_t ncclTopoCheckNicFused(struct ncclComm* comm, bool* fused) { return g_ncclTopoCheckNicFused(comm, fused); }

ncclResult_t ncclTopoCheckCrossNicSupport(bool* supported) {
  if (supported) *supported = false;
  return ncclSuccess;
}

ncclResult_t g_ncclTopoSearchInitResult = ncclSuccess;
ncclResult_t ncclTopoSearchInit(struct ncclTopoSystem* system) { return g_ncclTopoSearchInitResult; }

ncclResult_t g_ncclTopoComputeCommCPUResult = ncclSuccess;
ncclResult_t ncclTopoComputeCommCPU(struct ncclComm* comm) { return g_ncclTopoComputeCommCPUResult; }

// A std::function rather than a result knob: rung 3 needs it to SUCCEED and write graph->nChannels,
// which init.cc:1671-1672 then read to size the tree graph. Default still fails, so rung 2 is unchanged.
std::function<ncclResult_t(struct ncclTopoSystem*, struct ncclTopoGraph*)> g_ncclTopoCompute =
    [](struct ncclTopoSystem*, struct ncclTopoGraph*) { return ncclRemoteError; };  // rung-2 terminator
int g_ncclTopoComputeCalls = 0;
std::vector<struct ncclTopoGraph*> g_ncclTopoComputeGraphs;
ncclResult_t ncclTopoCompute(struct ncclTopoSystem* system, struct ncclTopoGraph* graph) {
  g_ncclTopoComputeCalls++;
  g_ncclTopoComputeGraphs.push_back(graph);
  return g_ncclTopoCompute(system, graph);
}

// Controllable (was fail-loud). Five call sites (init.cc:1649, :1674, :1691, :1693, :1703), each paired
// with an ncclTopoCompute. Records the graph so a swapped pair -- printing the tree graph after computing
// the ring one -- is visible; a result-only seam would not see it.
ncclResult_t g_ncclTopoPrintGraphResult = ncclSuccess;
std::vector<struct ncclTopoGraph*> g_ncclTopoPrintGraphGraphs;
ncclResult_t ncclTopoPrintGraph(struct ncclTopoSystem* system, struct ncclTopoGraph* graph) {
  g_ncclTopoPrintGraphGraphs.push_back(graph);
  return g_ncclTopoPrintGraphResult;
}

// Controllable (was fail-loud). init.cc:1764, gated on comm->rank == NCCL_GRAPH_DUMP_FILE_RANK. Records
// ngraphs because the call site hardcodes 5 -- a count that must track the five graphs actually computed.
ncclResult_t g_ncclTopoDumpGraphsResult = ncclSuccess;
int g_ncclTopoDumpGraphsCalls = 0;
int g_ncclTopoDumpGraphsNgraphs = -1;
std::vector<struct ncclTopoGraph*> g_ncclTopoDumpGraphsArray;
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

ncclResult_t DefaultTopoGetLinkType(int, int, bool* isXGMI, int) {
  if (isXGMI) *isXGMI = false;
  return ncclSuccess;
}
std::function<ncclResult_t(int, int, bool*, int)> g_ncclTopoGetLinkType = DefaultTopoGetLinkType;
int g_ncclTopoGetLinkTypeCalls = 0;
ncclResult_t ncclTopoGetLinkType(struct ncclTopoSystem* /*system*/, int cudaDev1, int cudaDev2, bool* isXGMI,
                                 int maxInter, int /*nInter*/, int* /*inter*/) {
  g_ncclTopoGetLinkTypeCalls++;
  return g_ncclTopoGetLinkType(cudaDev1, cudaDev2, isXGMI, maxInter);
}

void ResetSearchFakes() {
  g_ncclTopoCheckNicFused = [](struct ncclComm*, bool* fused) { *fused = false; return ncclSuccess; };
  g_ncclTopoSearchInitResult = ncclSuccess;
  g_ncclTopoComputeCommCPUResult = ncclSuccess;
  g_ncclTopoCompute = [](struct ncclTopoSystem*, struct ncclTopoGraph*) { return ncclRemoteError; };
  g_ncclTopoComputeCalls = 0;
  g_ncclTopoComputeGraphs.clear();
  g_ncclTopoPrintGraphResult = ncclSuccess;
  g_ncclTopoPrintGraphGraphs.clear();
  g_ncclTopoDumpGraphsResult = ncclSuccess;
  g_ncclTopoDumpGraphsCalls = 0;
  g_ncclTopoDumpGraphsNgraphs = -1;
  g_ncclTopoDumpGraphsArray.clear();
  g_ncclTopoGetLinkType = DefaultTopoGetLinkType;
  g_ncclTopoGetLinkTypeCalls = 0;
}
