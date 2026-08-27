/*************************************************************************
 * Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

// Fail-loud stub floor for enqueue.cc's deep paths (kernel launch, graph
// capture, symmetric/RMA scheduling, buffer registration, profiler).
//
// These satisfy enqueue.cc's link-time symbol closure. The microtests target the
// pure-compute and struct-manipulation helpers and never reach these; reaching
// one aborts, which is the point -- an unfaked path must be loud, not silent.
// A test that needs one replaces that individual entry with a real fake.
//
// Exception: profiler entry points and benign teardown return success, because
// happy-path helpers legitimately call them in passing.

#include <cstdio>
#include <cstdlib>

#include "comm.h"
#include "info.h"
#include "nccl.h"
#include "profiler.h"
#include "proxy.h"

namespace {
[[noreturn]] void Unreached(const char* fn) {
  std::fprintf(stderr, "[enqueue_stubs] unfaked call: %s\n", fn);
  std::fflush(stderr);
  ::abort();
}
}  // namespace

#define FAIL_LOUD(fn) Unreached(fn)

// ---- graph capture / stream ordering -------------------------------------
extern "C" {}
ncclResult_t ncclCudaGetCapturingGraph(struct ncclCudaGraph*, hipStream_t, int) {
  FAIL_LOUD("ncclCudaGetCapturingGraph");
}
ncclResult_t ncclCudaGraphAddDestructor(struct ncclCudaGraph, hipHostFn_t, void*) {
  FAIL_LOUD("ncclCudaGraphAddDestructor");
}
// ncclStreamWaitStream: owned by nccl_fakes.cc:293 (shared). Not redefined here.
ncclResult_t ncclStreamAdvanceToEvent(struct ncclCudaGraph, hipStream_t, hipEvent_t) {
  FAIL_LOUD("ncclStreamAdvanceToEvent");
}
ncclResult_t ncclStrongStreamAcquiredWorkStream(struct ncclCudaGraph, struct ncclStrongStream*,
                                                bool, hipStream_t*) {
  FAIL_LOUD("ncclStrongStreamAcquiredWorkStream");
}

// ---- symmetric / RMA / bcast scheduling ----------------------------------
ncclResult_t ncclMakeSymmetricTaskList(struct ncclComm*, struct ncclTaskColl*,
                                       struct ncclIntruQueue<struct ncclTaskColl,
                                                             &ncclTaskColl::next>*,
                                       struct ncclTaskColl**) {
  FAIL_LOUD("ncclMakeSymmetricTaskList");
}
ncclResult_t ncclSymmetricTaskScheduler(struct ncclComm*,
                                        struct ncclIntruQueue<struct ncclTaskColl,
                                                              &ncclTaskColl::next>*,
                                        struct ncclKernelPlan*) {
  FAIL_LOUD("ncclSymmetricTaskScheduler");
}
ncclResult_t ncclScheduleBcastTasksToPlan(struct ncclComm*, struct ncclKernelPlan*,
                                          struct ncclKernelPlanBudget*) {
  FAIL_LOUD("ncclScheduleBcastTasksToPlan");
}
ncclResult_t ncclRmaProxyReclaimPlan(struct ncclComm*, struct ncclKernelPlan*) {
  FAIL_LOUD("ncclRmaProxyReclaimPlan");
}
ncclResult_t ncclLaunchOneRank(void*, void const*, size_t, struct ncclDevRedOpFull,
                               ncclDataType_t, hipStream_t, void const*) {
  FAIL_LOUD("ncclLaunchOneRank");
}
ncclResult_t ncclShadowPoolToHost(struct ncclShadowPool*, void*, void**) {
  FAIL_LOUD("ncclShadowPoolToHost");
}

// ---- buffer registration --------------------------------------------------
ncclResult_t ncclRegisterCollBuffers(struct ncclComm*, struct ncclTaskColl*, void**, void**,
                                     struct ncclIntruQueue<struct ncclCommCallback,
                                                           &ncclCommCallback::next>*,
                                     bool*) {
  FAIL_LOUD("ncclRegisterCollBuffers");
}
ncclResult_t ncclRegisterCollNvlsBuffers(struct ncclComm*, struct ncclTaskColl*, void**, void**,
                                         struct ncclIntruQueue<struct ncclCommCallback,
                                                               &ncclCommCallback::next>*,
                                         bool*) {
  FAIL_LOUD("ncclRegisterCollNvlsBuffers");
}
ncclResult_t ncclRegisterP2pIpcBuffer(struct ncclComm*, void*, size_t, int, int*, void**,
                                      struct ncclIntruQueue<struct ncclCommCallback,
                                                            &ncclCommCallback::next>*) {
  FAIL_LOUD("ncclRegisterP2pIpcBuffer");
}
ncclResult_t ncclRegisterP2pNetBuffer(struct ncclComm*, void*, size_t, struct ncclConnector*,
                                      int*, void**,
                                      struct ncclIntruQueue<struct ncclCommCallback,
                                                            &ncclCommCallback::next>*) {
  FAIL_LOUD("ncclRegisterP2pNetBuffer");
}

// ---- misc ----------------------------------------------------------------
// ncclGetSymRegType moved to fakes/enqueue_fakes.cc: a fixed success here always
// reported "neither side registered", which actively steers production down one
// arm. It is a configurable, counted seam now rather than a silent default.
ncclResult_t ncclCommSetAsyncError(struct ncclComm*, ncclResult_t) {
  return ncclSuccess;  // benign: error bookkeeping, not behaviour under test
}
