/*************************************************************************
 * Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

// Seams topo_stubs.cc OWNS -- the src/graph/*.cc surface (topo.cc, paths.cc,
// search.cc, connect.cc, rome_topo_consensus.cc) -- declared once so a signature
// change is a compile error rather than a link mismatch.

#ifndef RCCL_TEST_HOST_TOPO_STUBS_H_
#define RCCL_TEST_HOST_TOPO_STUBS_H_

#include <functional>
#include <string>
#include <vector>

#include "nccl.h"
#include "os.h"  // ncclAffinity

struct ncclComm;
struct ncclTopoGraph;
struct ncclTopoRanks;
struct ncclTopoSystem;

// -------------------------------------------------------------------------
// Rung 1 of the initTransportsRank error-injection ladder (init.cc:1386).
// ncclTopoGetSystem stays defaulted to FAILURE on purpose: it is the first call after the
// MNNVL/intra-proc block, so that default is what terminates the ladder and makes :1462-1565
// reachable. Its dumpXmlFile argument passes through so a test can tell :1573 from :1576.
// ncclRemoteError is a SENTINEL -- no init.cc path reachable from initTransportsRank produces it.
// -------------------------------------------------------------------------
extern std::function<ncclResult_t(struct ncclComm*, struct ncclTopoSystem**, const char*)> g_ncclTopoGetSystem;
// src/graph/paths.cc. A std::function, not a result code: initTransportsRank:1506 branches on the WRITTEN
// *level, so a result-only seam could not drive it.
extern std::function<ncclResult_t(int*)> g_ncclGetUserP2pLevel;

// -------------------------------------------------------------------------
// Rung 2: topology detection / CPU affinity (init.cc:1576-1648). All default to success so a test can
// walk the block and inject exactly one failure. ncclTopoCompute is the exception -- it defaults to
// FAILURE because it is the rung-2 terminator, the role ncclTopoGetSystem plays for rung 1.
// ncclTopoComputePaths gets a FailAt index rather than a result because :1591 and :1596 call it twice
// and a single knob cannot separate them.
// -------------------------------------------------------------------------
extern int g_ncclTopoComputePathsCalls;
extern int g_ncclTopoComputePathsFailAt;   // -1 = never fail; 0 = the :1591 call, 1 = the :1596 one
extern ncclResult_t g_ncclTopoTrimSystemResult;
extern ncclResult_t g_ncclTopoSearchInitResult;
extern ncclResult_t g_ncclTopoComputeCommCPUResult;
extern ncclResult_t g_ncclTopoPrintResult;
extern std::function<ncclResult_t(struct ncclTopoSystem*, int, ncclAffinity*)> g_ncclTopoGetCpuAffinity;
extern int g_ncclTopoGetCpuAffinityLastRank;
// A std::function, not a result knob: rung 3 needs it to succeed AND write graph->nChannels, which
// :1671-1672 read back to size the tree graph. Defaults to failing, so it stays the rung-2 terminator.
extern std::function<ncclResult_t(struct ncclTopoSystem*, struct ncclTopoGraph*)> g_ncclTopoCompute;
extern int g_ncclTopoComputeCalls;
// Every ncclTopoGraph* handed to ncclTopoCompute, in call order; [0] is the :1648 ring compute.
extern std::vector<struct ncclTopoGraph*> g_ncclTopoComputeGraphs;

// -------------------------------------------------------------------------
// Rung 3: the graph block (init.cc:1649-1774). ncclTopoComputeP2pChannelsPerPeer terminates this rung
// and deliberately uses a DIFFERENT sentinel (ncclTimeout) from the ncclRemoteError rungs 1 and 2
// share: a rung-3 test that forgot to arm g_ncclTopoCompute would stop at :1648 and return
// ncclRemoteError, which no rung-3 assertion accepts.
// -------------------------------------------------------------------------
extern ncclResult_t g_ncclTopoPrintGraphResult;
extern std::vector<struct ncclTopoGraph*> g_ncclTopoPrintGraphGraphs;  // pairs 1:1 with the computes
extern ncclResult_t g_ncclTopoDumpGraphsResult;
extern int g_ncclTopoDumpGraphsCalls;
// The ngraphs :1764 passed. -1 until the dump runs; assert this rather than the vector length,
// which the fake clamps to the caller's array capacity.
extern int g_ncclTopoDumpGraphsNgraphs;
extern std::vector<struct ncclTopoGraph*> g_ncclTopoDumpGraphsArray;
extern ncclResult_t g_ncclTopoComputeP2pChannelsPerPeerResult;

// -------------------------------------------------------------------------
// Rung 4: AllGather3 (init.cc:1786-2213). ncclTopoPostset ends it with ncclInvalidUsage, not rung 3's
// ncclTimeout. Each default is deterministic: the UUT marshals it into allGather3Data.
// -------------------------------------------------------------------------
extern std::function<ncclResult_t(struct ncclComm*, bool*)> g_ncclTopoCheckNicFused;
extern std::function<ncclResult_t(struct ncclTopoSystem*, int, float*)> g_ncclTopoGetMinNetBw;
extern std::function<ncclResult_t(struct ncclTopoSystem*, int, int*, float*)> g_ncclTopoGetLocalNetCountByBw;
extern std::function<ncclResult_t(struct ncclTopoSystem*, int*)> g_ncclTopoPathAllNVLink;
extern std::function<ncclResult_t(struct ncclComm*, struct ncclTopoRanks*)> g_ncclTopoPreset;
extern ncclResult_t g_rcclCheckRomeTopoModelIdxConsensusResult;
extern int g_rcclCheckRomeTopoModelIdxConsensusCalls;
// What :1999's three lambdas answer for rank 0, i.e. the romeTopoModelIdx and hostname :1974-1975 marshalled.
extern int g_rcclRomeConsensusNranks;
extern int g_rcclRomeConsensusIdx0;
extern std::string g_rcclRomeConsensusHost0;
// src/graph/connect.cc. init.cc:2215, gated on comm->topo->treeDefined. The graph :2215 passed is
// recorded because only the tree graph is correct there and a result-only seam cannot see a swap.
extern ncclResult_t g_ncclTreeBasePostsetResult;
extern int g_ncclTreeBasePostsetCalls;
extern struct ncclTopoGraph* g_ncclTreeBasePostsetGraph;
extern ncclResult_t g_ncclTopoPostsetResult;
extern int g_ncclTopoPostsetCalls;
// The seven graph slots :2213 passed, in order; the aliasing between them is part of the contract.
extern std::vector<struct ncclTopoGraph*> g_ncclTopoPostsetGraphs;
// The `nc` :2213 passed, i.e. the min over every rank's allGather3Data[].nc. -1 until postset runs.
extern int g_ncclTopoPostsetNc;

void ResetTopoStubs();

#endif  // RCCL_TEST_HOST_TOPO_STUBS_H_
