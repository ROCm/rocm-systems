/*************************************************************************
 * Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

// Seams for the entry points that src/graph/search.cc DEFINES, for targets
// that do not compile that file.
//
// Split out of topo_stubs.cc, which is the fakes file for src/graph/ as a
// whole. rccl-UnitTestsMicroSearch compiles search.cc as its unit under test,
// so it links topo_stubs.cc (for search.cc's dependencies) but NOT this file
// (which would duplicate search.cc's own definitions). Every other microtest
// target links both. This mirrors the src/rccl_wrap.cc split described in
// MICROTEST_README.md: wrap_fakes.cc holds the dependencies of the file when
// it is the unit under test, rccl_wrap_fakes.cc holds its own entry points for
// the targets that do not compile it.
//
// Several of these seams play a role in init.cc's error-injection ladder, and
// the "rung" commentary that goes with them is kept here next to the seam it
// describes. See fakes/topo_stubs.h for the rungs' overall structure; the two
// headers are read together by init-test.cc.

#ifndef RCCL_TEST_HOST_FAKES_SEARCH_FAKES_H_
#define RCCL_TEST_HOST_FAKES_SEARCH_FAKES_H_

#include <functional>
#include <vector>

#include "nccl.h"

struct ncclComm;
struct ncclTopoSystem;
struct ncclTopoGraph;

// --- Rung 2: topology detection (init.cc:1576-1648) ------------------------
extern ncclResult_t g_ncclTopoSearchInitResult;
extern ncclResult_t g_ncclTopoComputeCommCPUResult;
// A std::function, not a result knob: rung 3 needs it to succeed AND write
// graph->nChannels, which :1671-1672 read back to size the tree graph.
// Defaults to failing, so it stays the rung-2 terminator.
extern std::function<ncclResult_t(struct ncclTopoSystem*, struct ncclTopoGraph*)> g_ncclTopoCompute;
extern int g_ncclTopoComputeCalls;
// Every ncclTopoGraph* handed to ncclTopoCompute, in call order; [0] is the
// :1648 ring compute.
extern std::vector<struct ncclTopoGraph*> g_ncclTopoComputeGraphs;

// --- Rung 3: the graph block (init.cc:1649-1774) ---------------------------
extern ncclResult_t g_ncclTopoPrintGraphResult;
extern std::vector<struct ncclTopoGraph*> g_ncclTopoPrintGraphGraphs;  // pairs 1:1 with the computes
extern ncclResult_t g_ncclTopoDumpGraphsResult;
extern int g_ncclTopoDumpGraphsCalls;
// The ngraphs :1764 passed. -1 until the dump runs; assert this rather than
// the vector length, which the fake clamps to the caller's array capacity.
extern int g_ncclTopoDumpGraphsNgraphs;
extern std::vector<struct ncclTopoGraph*> g_ncclTopoDumpGraphsArray;

// --- Rung 4: AllGather3 (init.cc:1786-2213) --------------------------------
extern std::function<ncclResult_t(struct ncclComm*, bool*)> g_ncclTopoCheckNicFused;

// --- Not part of the init.cc ladder ----------------------------------------
// Driven by p2p-test.cc and init-test.cc. The two device ids and the isXGMI
// out-parameter; the topo system and the inter-GPU knobs are dropped. maxInter
// is carried because init.cc:1834 passes an explicit 1 where graph.h:90
// defaults to MAX_XGMI_INTER_GPUS.
extern std::function<ncclResult_t(int /*cudaDev1*/, int /*cudaDev2*/, bool* /*isXGMI*/, int /*maxInter*/)>
    g_ncclTopoGetLinkType;
extern int g_ncclTopoGetLinkTypeCalls;
ncclResult_t DefaultTopoGetLinkType(int, int, bool* isXGMI, int);

// Restores every seam above to its default. Chained by neither ResetTopoStubs()
// nor ResetNcclFakes(): a test that drives these calls it directly.
void ResetSearchFakes();

#endif  // RCCL_TEST_HOST_FAKES_SEARCH_FAKES_H_
