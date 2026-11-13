/*************************************************************************
 * Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

// Fakes for src/hier_host.cc, for targets that compile rccl_wrap.cc,
// collectives.cc or init.cc but not hier_host.cc.

#ifndef RCCL_TEST_HOST_HIER_HOST_FAKES_H_
#define RCCL_TEST_HOST_HIER_HOST_FAKES_H_

#include <cstddef>
#include <functional>

#include "nccl.h"

struct ncclComm;

// Default false, so the selector never reaches the host-hierarchical branch
// unless a test opts in.
extern std::function<bool(ncclComm*, size_t, ncclDataType_t, ncclRedOp_t)> g_hostHierAllReduceEligible;
// Fail loud until a test installs them.
extern std::function<ncclResult_t(ncclComm*)> g_hostHierEnsure;
extern std::function<ncclResult_t(const void*, void*, size_t, ncclDataType_t, ncclRedOp_t, ncclComm*, hipStream_t)>
    g_hostHierAllReduce;
// Default clears comm->hostHier and succeeds, which is what every comm without
// child comms sees in commFree.
extern std::function<ncclResult_t(ncclComm*)> g_hostHierTeardown;

#endif  // RCCL_TEST_HOST_HIER_HOST_FAKES_H_
