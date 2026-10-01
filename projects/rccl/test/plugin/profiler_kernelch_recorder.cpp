/*************************************************************************
 * Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

// Profiler plugin that records Coll, P2p, KernelCh and KernelPhase events for
// ProfilerKernelChMPITests. It builds against the standalone headers an external
// plugin ships with (plugins/profiler/example/nccl), so it sees the same ABI an
// out-of-tree plugin would.
//
// Every handle it hands out stays valid until rcclKchRecorderReset(). That is a
// requirement, not a convenience: RCCL stops a Coll or P2p event in the host task
// that posts its kernel work, before the kernel has run, and the KernelCh events
// that name it as their parent arrive later from the profiler thread.

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/syscall.h>
#include <unistd.h>

#include <deque>
#include <mutex>
#include <set>
#include <unordered_map>

#include "err.h"
#include "profiler.h"
#include "profiler_kernelch_recorder.h"

namespace {

constexpr uint64_t kMagic = 0x6b63685265633031ull;  // "kchRec01"

struct Context {
  uint64_t commId;
  int rank;
};

struct Record {
  uint64_t magic;
  RcclKchRecord view;  // parentIndex is resolved at snapshot time
  const Record* parent;
};

std::mutex gMutex;
std::deque<Record> gRecords;  // deque keeps element addresses stable
// Never cleared: a live communicator keeps the context pointer init() returned,
// and a reset must not invalidate it.
std::deque<Context> gContexts;
uint64_t gOrder;
uint64_t gInitCount;
uint64_t gProxyOpCount;
std::set<long> gProxyTids;
uint64_t gAnomalies;

int activationMask() {
  const char* env = getenv("RCCL_TEST_KCH_EVENT_MASK");
  if (env && env[0] != '\0') return (int)strtol(env, nullptr, 0);
  return ncclProfileColl | ncclProfileP2p | ncclProfileKernelCh | ncclProfileProxyOp;
}

// Caller holds gMutex.
Record* asRecord(void* handle) {
  Record* r = static_cast<Record*>(handle);
  return (r != nullptr && r->magic == kMagic) ? r : nullptr;
}

// Caller holds gMutex.
Record* newRecord(uint64_t type, const Context* ctx, void* parentObj) {
  gRecords.emplace_back();
  Record* r = &gRecords.back();
  memset(&r->view, 0, sizeof(r->view));
  r->magic = kMagic;
  r->parent = asRecord(parentObj);
  r->view.type = type;
  r->view.commId = ctx ? ctx->commId : 0;
  r->view.rank = ctx ? ctx->rank : -1;
  r->view.order = gOrder++;
  r->view.startTid = (long)syscall(SYS_gettid);
  r->view.channelId = -1;
  r->view.phaseId = -1;
  r->view.peer = -1;
  return r;
}

void copyFunc(char* dst, const char* src) {
  if (src == nullptr) return;
  strncpy(dst, src, sizeof(((RcclKchRecord*)nullptr)->func) - 1);
}

ncclResult_t recorderInit(void** context, uint64_t commId, int* eActivationMask, const char* /*commName*/,
                          int /*nNodes*/, int /*nranks*/, int rank, ncclDebugLogger_t /*logfn*/) {
  std::lock_guard<std::mutex> lock(gMutex);
  gContexts.push_back(Context{commId, rank});
  *context = &gContexts.back();
  *eActivationMask = activationMask();
  ++gInitCount;
  return ncclSuccess;
}

ncclResult_t recorderStartEvent(void* context, void** eHandle, ncclProfilerEventDescr_v7_t* eDescr) {
  *eHandle = nullptr;
  const Context* ctx = static_cast<const Context*>(context);
  std::lock_guard<std::mutex> lock(gMutex);
  switch (eDescr->type) {
    case ncclProfileColl: {
      Record* r = newRecord(eDescr->type, ctx, nullptr);
      copyFunc(r->view.func, eDescr->coll.func);
      r->view.seqNumber = eDescr->coll.seqNumber;
      r->view.count = eDescr->coll.count;
      r->view.nChannels = eDescr->coll.nChannels;
      r->view.isSymColl = eDescr->coll.isSymColl ? 1 : 0;
      *eHandle = r;
      break;
    }
    case ncclProfileP2p: {
      Record* r = newRecord(eDescr->type, ctx, nullptr);
      copyFunc(r->view.func, eDescr->p2p.func);
      r->view.count = eDescr->p2p.count;
      r->view.nChannels = eDescr->p2p.nChannels;
      r->view.peer = eDescr->p2p.peer;
      *eHandle = r;
      break;
    }
    case ncclProfileKernelCh: {
      Record* r = newRecord(eDescr->type, ctx, eDescr->parentObj);
      r->view.channelId = eDescr->kernelCh.channelId;
      r->view.startTimer = eDescr->kernelCh.pTimer;
      *eHandle = r;
      break;
    }
    case ncclProfileKernelPhase: {
      Record* r = newRecord(eDescr->type, ctx, eDescr->parentObj);
      r->view.channelId = eDescr->kernelPhase.channelId;
      r->view.phaseId = eDescr->kernelPhase.phaseId;
      r->view.startTimer = eDescr->kernelPhase.pTimer;
      *eHandle = r;
      break;
    }
    case ncclProfileProxyOp:
      // Counted only: a proxy-less path must leave this at zero.
      ++gProxyOpCount;
      gProxyTids.insert((long)syscall(SYS_gettid));
      break;
    default:
      break;
  }
  return ncclSuccess;
}

ncclResult_t recorderStopEvent(void* eHandle) {
  if (eHandle == nullptr) return ncclSuccess;
  std::lock_guard<std::mutex> lock(gMutex);
  Record* r = asRecord(eHandle);
  if (r == nullptr || r->view.stopEvents != 0) {
    ++gAnomalies;
    return ncclSuccess;
  }
  ++r->view.stopEvents;
  return ncclSuccess;
}

ncclResult_t recorderRecordEventState(void* eHandle, ncclProfilerEventState_v7_t eState,
                                      ncclProfilerEventStateArgs_v7_t* eStateArgs) {
  if (eHandle == nullptr) return ncclSuccess;
  if (eState != ncclProfilerKernelChStop && eState != ncclProfilerKernelPhaseStop) return ncclSuccess;
  std::lock_guard<std::mutex> lock(gMutex);
  Record* r = asRecord(eHandle);
  if (r == nullptr || r->view.stopEvents != 0) {
    ++gAnomalies;
    return ncclSuccess;
  }
  ++r->view.stopStates;
  if (eStateArgs) r->view.stopTimer = eStateArgs->kernelCh.pTimer;
  return ncclSuccess;
}

ncclResult_t recorderFinalize(void* /*context*/) { return ncclSuccess; }

}  // namespace

extern "C" {

__attribute__((visibility("default"))) ncclProfiler_v7_t ncclProfiler_v7 = {
  "KernelChRecorder", recorderInit, recorderStartEvent, recorderStopEvent, recorderRecordEventState,
  recorderFinalize,
};

__attribute__((visibility("default"))) void rcclKchRecorderReset(void) {
  std::lock_guard<std::mutex> lock(gMutex);
  gRecords.clear();
  gOrder = 0;
  gInitCount = 0;
  gProxyOpCount = 0;
  gProxyTids.clear();
  gAnomalies = 0;
}

__attribute__((visibility("default"))) size_t rcclKchRecorderSnapshot(RcclKchRecord* out, size_t cap) {
  std::lock_guard<std::mutex> lock(gMutex);
  std::unordered_map<const Record*, int64_t> index;
  index.reserve(gRecords.size());
  int64_t i = 0;
  for (const Record& r : gRecords) index[&r] = i++;
  size_t n = 0;
  for (const Record& r : gRecords) {
    if (n == cap) break;
    out[n] = r.view;
    auto it = r.parent ? index.find(r.parent) : index.end();
    out[n].parentIndex = (it == index.end()) ? -1 : it->second;
    ++n;
  }
  return gRecords.size();
}

__attribute__((visibility("default"))) uint64_t rcclKchRecorderInitCount(void) {
  std::lock_guard<std::mutex> lock(gMutex);
  return gInitCount;
}

__attribute__((visibility("default"))) uint64_t rcclKchRecorderProxyOpCount(void) {
  std::lock_guard<std::mutex> lock(gMutex);
  return gProxyOpCount;
}

__attribute__((visibility("default"))) size_t rcclKchRecorderProxyThreads(long* out, size_t cap) {
  std::lock_guard<std::mutex> lock(gMutex);
  size_t n = 0;
  for (long tid : gProxyTids) {
    if (n == cap) break;
    out[n++] = tid;
  }
  return gProxyTids.size();
}

__attribute__((visibility("default"))) uint64_t rcclKchRecorderAnomalies(void) {
  std::lock_guard<std::mutex> lock(gMutex);
  uint64_t n = gAnomalies;
  for (const Record& r : gRecords) {
    // A device-timed event stopped without the stop state that carries its timer.
    if ((r.view.type == ncclProfileKernelCh || r.view.type == ncclProfileKernelPhase) && r.view.stopEvents != 0 &&
        r.view.stopStates == 0)
      ++n;
  }
  return n;
}

}  // extern "C"
