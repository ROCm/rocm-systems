/*************************************************************************
 * Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

// The sched_stubs.cc entries that are controllable seams, not hard aborts.

#ifndef RCCL_TEST_HOST_SCHED_STUBS_H_
#define RCCL_TEST_HOST_SCHED_STUBS_H_

#include <cstddef>
#include <functional>

#include <hip/hip_runtime_api.h>

#include "nccl.h"

struct ncclComm;
struct ncclDevRedOpFull;
struct ncclTaskColl;

extern std::function<void(struct ncclComm*, struct ncclTaskColl*)> g_convertSymTaskDevOp;

// device/onerank.cu. Fails loud by default.
extern std::function<ncclResult_t(void*, void const*, size_t, struct ncclDevRedOpFull, ncclDataType_t, hipStream_t,
                                  hipEvent_t, void const*)>
    g_ncclLaunchOneRank;

void ResetSchedStubs();

#endif  // RCCL_TEST_HOST_SCHED_STUBS_H_
