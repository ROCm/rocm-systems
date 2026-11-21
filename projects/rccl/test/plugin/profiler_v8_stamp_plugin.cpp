/*************************************************************************
 * Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

// Profiler plugin for ProfilerV8TimestampTests. It exports ncclProfiler_v8 and
// nothing older, opts into NCCL-sourced CLOCK_MONOTONIC timestamps through the
// v8 config handshake, and counts how many host API events arrive stamped.
//
// It spells out only the leading fields of the v8 structs it reads (see
// src/include/plugin/profiler/profiler_v8.h) so it builds without RCCL headers.

#include <limits.h>
#include <stdint.h>

#include <mutex>

#include "profiler_v8_stamp_plugin.h"

namespace {

struct DescrV8 {  // ncclProfilerEventDescr_v8_t up to the union
  uint64_t type;
  void* parentObj;
  int rank;
  uint64_t tsCpu;
  uint64_t tsCpuUncertainty;
};
struct StateArgsV8 {  // ncclProfilerEventStateArgs_v8_t up to the union
  uint64_t tsCpu;
  uint64_t tsCpuUncertainty;
};
struct ConfigV8 {  // ncclProfilerConfig_v8_t
  uint32_t cpuClock;
  uint32_t gpuClockSync;
};
struct ProfilerV8 {  // ncclProfiler_v8_t; results are ncclResult_t, 0 == ncclSuccess
  const char* name;
  int (*init)(void**, uint64_t, int*, const char*, int, int, int, void*, ConfigV8*);
  int (*startEvent)(void*, void**, DescrV8*);
  int (*stopEvent)(void*, DescrV8*);
  int (*recordEventState)(void*, int, StateArgsV8*);
  int (*finalize)(void*);
};

constexpr int kGroupApi = 1 << 8, kCollApi = 1 << 9;  // host-stamped API events only
constexpr uint32_t kCpuClockMonotonic = 1;

std::mutex mu;
RcclProfV8Stats stats = {0, 0, 0, 0, 0, 0, 0, UINT64_MAX, 0};
int contextTag;

// Host-side events carry the clock read at the call site, with no uncertainty.
void note(uint64_t ts, uint64_t uncertainty) {
  if (ts == 0 || uncertainty != 0) ++stats.unstamped;
  if (ts == 0) return;
  if (ts < stats.minTs) stats.minTs = ts;
  if (ts > stats.maxTs) stats.maxTs = ts;
}

int init(void** context, uint64_t, int* eActivationMask, const char*, int, int, int, void*, ConfigV8* config) {
  std::lock_guard<std::mutex> lock(mu);
  ++stats.inits;
  if (config == nullptr) ++stats.initsNoConfig;
  else config->cpuClock = kCpuClockMonotonic;
  *context = &contextTag;
  *eActivationMask = kGroupApi | kCollApi;
  return 0;
}

int startEvent(void*, void** eHandle, DescrV8* eDescr) {
  std::lock_guard<std::mutex> lock(mu);
  ++stats.starts;
  note(eDescr->tsCpu, eDescr->tsCpuUncertainty);
  *eHandle = new uint64_t(eDescr->tsCpu);
  return 0;
}

int stopEvent(void* eHandle, DescrV8* eDescr) {
  std::lock_guard<std::mutex> lock(mu);
  ++stats.stops;
  uint64_t* start = static_cast<uint64_t*>(eHandle);
  if (eDescr == nullptr) {
    ++stats.unstamped;
  } else {
    note(eDescr->tsCpu, eDescr->tsCpuUncertainty);
    if (start != nullptr && eDescr->tsCpu < *start) ++stats.stopBeforeStart;
  }
  delete start;
  return 0;
}

int recordEventState(void*, int, StateArgsV8* eStateArgs) {
  std::lock_guard<std::mutex> lock(mu);
  ++stats.states;
  if (eStateArgs == nullptr) ++stats.unstamped;
  else note(eStateArgs->tsCpu, eStateArgs->tsCpuUncertainty);
  return 0;
}

int finalize(void*) { return 0; }

}  // namespace

extern "C" {

__attribute__((visibility("default"))) ProfilerV8 ncclProfiler_v8 = {
    "V8TimestampCheck", init, startEvent, stopEvent, recordEventState, finalize};

__attribute__((visibility("default"))) void rcclTestProfV8Snapshot(RcclProfV8Stats* out) {
  std::lock_guard<std::mutex> lock(mu);
  *out = stats;
}

}  // extern "C"
