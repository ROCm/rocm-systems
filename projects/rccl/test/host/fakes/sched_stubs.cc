/*************************************************************************
 * Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

// Fail-loud stub floor for the plan schedulers and the deep launch paths a
// host-only microtest never executes: src/scheduler/symmetric_sched.cc,
// src/scheduler/allgatherv_sched.cc, src/rma/rma_proxy_launch.cc,
// src/allocator.cc and src/device/onerank.cu.
//
// Reaching one aborts, which is the point: an unfaked path must be loud, not
// silent. A test that needs one replaces that individual entry with a real fake.

#include "comm.h"
#include "device.h"
#include "info.h"
#include "nccl.h"

#include "fail_loud.h"

namespace {
[[noreturn]] void Unreached(const char* fn) { FailLoud("sched_stubs", fn); }
}  // namespace

// scheduler/symmetric_sched.cc
ncclResult_t ncclMakeSymmetricTaskList(struct ncclComm*, struct ncclTaskColl*,
                                       struct ncclIntruQueue<struct ncclTaskColl,
                                                             &ncclTaskColl::next>*,
                                       struct ncclTaskColl**) {
  Unreached("ncclMakeSymmetricTaskList");
}
ncclResult_t ncclSymmetricTaskScheduler(struct ncclComm*,
                                        struct ncclIntruQueue<struct ncclTaskColl,
                                                              &ncclTaskColl::next>*,
                                        struct ncclKernelPlan*) {
  Unreached("ncclSymmetricTaskScheduler");
}

// scheduler/allgatherv_sched.cc
ncclResult_t ncclScheduleBcastTasksToPlan(struct ncclComm*, struct ncclKernelPlan*,
                                          struct ncclKernelPlanBudget*) {
  Unreached("ncclScheduleBcastTasksToPlan");
}

// rma/rma_proxy_launch.cc
ncclResult_t ncclRmaProxyReclaimPlan(struct ncclComm*, struct ncclKernelPlan*) {
  Unreached("ncclRmaProxyReclaimPlan");
}

// device/onerank.cu
ncclResult_t ncclLaunchOneRank(void*, void const*, size_t, struct ncclDevRedOpFull,
                               ncclDataType_t, hipStream_t, void const*) {
  Unreached("ncclLaunchOneRank");
}

// allocator.cc
ncclResult_t ncclShadowPoolToHost(struct ncclShadowPool*, void*, void**) {
  Unreached("ncclShadowPoolToHost");
}
