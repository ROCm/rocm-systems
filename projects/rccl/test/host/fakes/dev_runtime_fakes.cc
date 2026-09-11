/*************************************************************************
 * Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

// See dev_runtime_fakes.h. ncclDevrFindWindow is controllable here; only
// ncclDevrIsOneLsaTeam remains in nccl_stubs.cc as a fail-loud floor.

#include "dev_runtime_fakes.h"

static ncclResult_t DefaultDevrFindWindow(struct ncclComm*, void const*, struct ncclDevrWindow** window) {
  if (window) *window = nullptr;
  return ncclSuccess;
}
std::function<ncclResult_t(struct ncclComm*, void const*, struct ncclDevrWindow**)> g_devrFindWindow =
    DefaultDevrFindWindow;
ncclResult_t ncclDevrFindWindow(struct ncclComm* comm, void const* ptr, struct ncclDevrWindow** window) {
  return g_devrFindWindow(comm, ptr, window);
}

bool g_devrWindowIsMultiSegment = false;
bool g_devrWindowHasSysmemSegmentValue = false;
static bool DefaultDevrWindowHasSysmemSegment(struct ncclDevrWindow*) { return g_devrWindowHasSysmemSegmentValue; }
std::function<bool(struct ncclDevrWindow*)> g_devrWindowHasSysmemSegment =
    DefaultDevrWindowHasSysmemSegment;

bool ncclDevrWindowIsMultiSegment(struct ncclDevrWindow*) { return g_devrWindowIsMultiSegment; }
bool ncclDevrWindowHasSysmemSegment(struct ncclDevrWindow* window) {
  return g_devrWindowHasSysmemSegment(window);
}

void ResetDevRuntimeFakes() {
  g_devrFindWindow = DefaultDevrFindWindow;
  g_devrWindowIsMultiSegment = false;
  g_devrWindowHasSysmemSegmentValue = false;
  g_devrWindowHasSysmemSegment = DefaultDevrWindowHasSysmemSegment;
}
