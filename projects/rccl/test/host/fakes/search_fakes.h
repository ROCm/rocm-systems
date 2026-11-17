/*************************************************************************
 * Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

#ifndef RCCL_TEST_HOST_FAKES_SEARCH_FAKES_H_
#define RCCL_TEST_HOST_FAKES_SEARCH_FAKES_H_

#include <functional>
#include <vector>

#include "nccl.h"

struct ncclComm;
struct ncclTopoSystem;
struct ncclTopoGraph;

extern ncclResult_t g_ncclTopoSearchInitResult;
extern ncclResult_t g_ncclTopoComputeCommCPUResult;
extern std::function<ncclResult_t(struct ncclTopoSystem*, struct ncclTopoGraph*)> g_ncclTopoCompute;
extern int g_ncclTopoComputeCalls;
extern std::vector<struct ncclTopoGraph*> g_ncclTopoComputeGraphs;

extern ncclResult_t g_ncclTopoPrintGraphResult;
extern std::vector<struct ncclTopoGraph*> g_ncclTopoPrintGraphGraphs;
extern ncclResult_t g_ncclTopoDumpGraphsResult;
extern int g_ncclTopoDumpGraphsCalls;
extern int g_ncclTopoDumpGraphsNgraphs;
extern std::vector<struct ncclTopoGraph*> g_ncclTopoDumpGraphsArray;
extern std::function<ncclResult_t(struct ncclComm*, bool*)> g_ncclTopoCheckNicFused;

extern std::function<ncclResult_t(int /*cudaDev1*/, int /*cudaDev2*/, bool* /*isXGMI*/, int /*maxInter*/)>
    g_ncclTopoGetLinkType;
extern int g_ncclTopoGetLinkTypeCalls;
ncclResult_t DefaultTopoGetLinkType(int, int, bool* isXGMI, int);

void ResetSearchFakes();

#endif  // RCCL_TEST_HOST_FAKES_SEARCH_FAKES_H_
