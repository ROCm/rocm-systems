/*************************************************************************
 * Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

// Link-time floor for collectives-test.cc.
//
// collectives-test.cc calls the REAL ncclAlltoAll_impl (from collectives.cc),
// whose fall-through path ends in `return ncclEnqueueCheck(&info)`. The real
// ncclEnqueueCheck (linked from enqueue.cc via enqueue-test.cc) bails at its
// first line -- CommCheck() -- because the test comm carries invalid magic, so
// none of the deep enqueue machinery runs at RUNTIME. But --gc-sections keeps
// that whole static subtree (taskAppend -> collTaskAppend / ceCollTaskAppend and
// the group/profiler entry points) at LINK time, so their leaf externals must be
// satisfied to link.
//
// Two kinds of symbol live here:
//   * rcclParam* : referenced by collectives.cc's gfx1250 branch and by the CE
//     AllReduce arms of the enqueue subtree. Runtime never reaches them on this
//     test's gfx942 + invalid-magic path, but they are ordinary functions the
//     linker needs. Return the production default so a future test that DOES
//     reach one gets defined behaviour rather than a link surprise.
//   * ncclGroup* / ncclProfiler* / ncclCommGetAsyncError : reached only AFTER
//     CommCheck() succeeds, which never happens here. Fail loudly if called so a
//     future test that forms a valid comm cannot silently execute a stub.

#include "fail_loud.h"

#include "nccl.h"
#include "comm.h"        // ncclGroupTaskTypeNum, struct ncclComm
#include "group.h"       // ncclSimInfo_t, thread_local group globals

#include <cstdint>

namespace {
constexpr const char* kTag = "collectives_stubs";
}

// --- rcclParam* leaves (RCCL_PARAM bodies normally in rccl_wrap.cc, faked out).
// Return the production defaults from the RCCL_PARAM(...) declarations.
extern "C++" {
int64_t rcclParamDdaLL() { return 0; }
int64_t rcclParamDdaLLThreshold() { return 0; }
int64_t rcclParamDdaLL128() { return 0; }
int64_t rcclParamDdaLL128Threshold() { return 0; }
int64_t rcclParamCeAllReduce() { return 1; }        // RCCL_CE_ALLREDUCE default 1
int64_t rcclParamForceCeAllReduce() { return 0; }   // RCCL_FORCE_CE_ALLREDUCE default 0
}

// --- Group/launch machinery: reached only past CommCheck(), which this test's
// invalid-magic comm never clears. Fail loud rather than pretend to succeed.
ncclResult_t ncclGroupStartInternal() { FailLoudUnfaked(kTag, "ncclGroupStartInternal"); }
ncclResult_t ncclGroupEndInternal(ncclSimInfo_t*) { FailLoudUnfaked(kTag, "ncclGroupEndInternal"); }
ncclResult_t ncclCommGetAsyncError(ncclComm_t, ncclResult_t*) {
  FailLoudUnfaked(kTag, "ncclCommGetAsyncError");
}

// thread_local group globals referenced by the inline ncclGroupComm* helpers in
// group.h. Never mutated on the covered path; provide real storage so the inline
// bodies link.
thread_local struct ncclComm* ncclGroupCommHead[ncclGroupTaskTypeNum] = {nullptr, nullptr};
thread_local struct ncclComm* ncclGroupCommPreconnectHead = nullptr;
thread_local int ncclGroupBlocking = 0;

// --- Profiler API entry points (enqueue.cc coll/p2p/group event hooks).
ncclResult_t ncclProfilerStartCollApiEvent(struct ncclInfo*, bool) {
  FailLoudUnfaked(kTag, "ncclProfilerStartCollApiEvent");
}
ncclResult_t ncclProfilerStopCollApiEvent() { FailLoudUnfaked(kTag, "ncclProfilerStopCollApiEvent"); }
ncclResult_t ncclProfilerStartGroupApiEvent(struct ncclInfo*, bool) {
  FailLoudUnfaked(kTag, "ncclProfilerStartGroupApiEvent");
}
ncclResult_t ncclProfilerStartP2pApiEvent(struct ncclInfo*, bool) {
  FailLoudUnfaked(kTag, "ncclProfilerStartP2pApiEvent");
}
ncclResult_t ncclProfilerStopP2pApiEvent() { FailLoudUnfaked(kTag, "ncclProfilerStopP2pApiEvent"); }
