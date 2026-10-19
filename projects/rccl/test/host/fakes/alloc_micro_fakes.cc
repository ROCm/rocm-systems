/*************************************************************************
 * Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

// See alloc_micro_fakes.h.

#include "alloc_micro_fakes.h"

#include <cstdarg>
#include <cstdio>
#include <cstring>

#include "debug.h"

uint32_t ncclDebugLevelMask = NCCL_DEBUG_LEVEL_MASK_UNINITIALIZED;
uint64_t ncclDebugMask = ~uint64_t(0);
thread_local int ncclDebugNoWarn = 0;
struct allocationTracker allocTracker[MAX_ALLOC_TRACK_NGPU] = {};
hipMemAllocationHandleType ncclCuMemHandleType = hipMemHandleTypePosixFileDescriptor;

std::vector<std::string> g_warns;

void ncclDebugLog(ncclDebugLogLevel level, unsigned long, const char*, int, const char* fmt, ...) {
  if (level != NCCL_LOG_WARN) return;
  char buf[512];
  va_list ap;
  va_start(ap, fmt);
  std::vsnprintf(buf, sizeof(buf), fmt, ap);
  va_end(ap);
  g_warns.push_back(buf);
}

bool Warned(const char* needle) {
  for (const std::string& w : g_warns)
    if (w.find(needle) != std::string::npos) return true;
  return false;
}

int g_cuMemEnable = 0;
int ncclCuMemEnable() { return g_cuMemEnable; }

ncclResult_t g_busIdResult = ncclSuccess;
ncclResult_t getBusId(int cudaDev, int64_t* busId) {
  *busId = 0x100 * cudaDev;
  return g_busIdResult;
}

std::vector<TrackCall> g_tracked;
ncclResult_t g_trackResult = ncclSuccess;
ncclMemUntrackInfo g_untrackInfo{};
ncclResult_t g_untrackResult = ncclSuccess;
std::vector<std::pair<void*, size_t>> g_untrackedPersist;

ncclResult_t ncclMemTrack(struct ncclMemManager*, void* ptr, size_t size, hipMemGenericAllocationHandle_t,
                          hipMemAllocationHandleType type, ncclMemType_t memType) {
  g_tracked.push_back({ptr, size, type, memType});
  return g_trackResult;
}
ncclResult_t ncclMemUntrackDynamic(struct ncclMemManager*, void*, struct ncclMemUntrackInfo* info) {
  *info = g_untrackInfo;
  return g_untrackResult;
}
ncclResult_t ncclMemUntrackPersist(struct ncclMemManager*, void* ptr, size_t size) {
  g_untrackedPersist.push_back({ptr, size});
  return ncclSuccess;
}

static int64_t DefaultLoadParam(const char*, int64_t deftVal) { return deftVal; }
std::function<int64_t(const char*, int64_t)> g_loadParam = DefaultLoadParam;

void ResetAllocMicroFakes() {
  g_warns.clear();
  std::memset(allocTracker, 0, sizeof(allocTracker));
  ncclCuMemHandleType = hipMemHandleTypePosixFileDescriptor;
  g_cuMemEnable = 0;
  g_busIdResult = ncclSuccess;
  g_tracked.clear();
  g_trackResult = ncclSuccess;
  g_untrackInfo = {};
  g_untrackResult = ncclSuccess;
  g_untrackedPersist.clear();
  g_loadParam = DefaultLoadParam;
}
