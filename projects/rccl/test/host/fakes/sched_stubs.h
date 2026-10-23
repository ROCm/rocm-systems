/*************************************************************************
 * Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

// The sched_stubs.cc entries that are controllable seams, not hard aborts.

#ifndef RCCL_TEST_HOST_SCHED_STUBS_H_
#define RCCL_TEST_HOST_SCHED_STUBS_H_

#include <functional>

struct ncclComm;
struct ncclTaskColl;

extern std::function<void(struct ncclComm*, struct ncclTaskColl*)> g_convertSymTaskDevOp;

void ResetSchedStubs();

#endif  // RCCL_TEST_HOST_SCHED_STUBS_H_
