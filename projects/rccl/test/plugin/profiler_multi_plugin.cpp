/*************************************************************************
 * Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

// Profiler plugin for ProfilerMultiPluginTests. The test loads two copies of this
// library from different files, so each copy keeps its own state: the event mask
// it asks for in init(), how many times init() ran, and the event types it was
// handed. Both copies have to be live at once for the test to see anything.

#include <atomic>
#include <cstdint>

#include "err.h"
#include "profiler.h"

namespace {

std::atomic<int> gMask{0};
std::atomic<int> gInitCount{0};
std::atomic<uint64_t> gSeenTypes{0};

ncclResult_t multiInit(void** context, uint64_t, int* eActivationMask, const char*, int, int, int, ncclDebugLogger_t) {
  *context = &gInitCount;
  *eActivationMask = gMask.load();
  ++gInitCount;
  return ncclSuccess;
}

ncclResult_t multiStartEvent(void* context, void** eHandle, ncclProfilerEventDescr_v7_t* eDescr) {
  gSeenTypes |= eDescr->type;
  *eHandle = context;
  return ncclSuccess;
}

ncclResult_t multiStopEvent(void*) { return ncclSuccess; }

ncclResult_t multiRecordEventState(void*, ncclProfilerEventState_v7_t, ncclProfilerEventStateArgs_v7_t*) {
  return ncclSuccess;
}

ncclResult_t multiFinalize(void*) { return ncclSuccess; }

}  // namespace

extern "C" {

__attribute__((visibility("default"))) ncclProfiler_v7_t ncclProfiler_v7 = {
  "MultiPluginRecorder", multiInit, multiStartEvent, multiStopEvent, multiRecordEventState, multiFinalize,
};

__attribute__((visibility("default"))) void rcclMultiProfSetMask(int mask) { gMask = mask; }
__attribute__((visibility("default"))) int rcclMultiProfInitCount(void) { return gInitCount.load(); }
__attribute__((visibility("default"))) uint64_t rcclMultiProfSeenTypes(void) { return gSeenTypes.load(); }

}  // extern "C"
