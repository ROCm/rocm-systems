/*************************************************************************
 * Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

// See dev_runtime_fakes.h. ncclDevrFindWindow is controllable here; only
// ncclDevrIsOneLsaTeam remains in nccl_stubs.cc as a fail-loud floor.

#include "dev_runtime_fakes.h"

#include "dev_runtime.h"
#include "signature-drift.h"

ASSERT_HOOK_MATCHES_PROD(g_devrFindWindow, ncclDevrFindWindow);
ASSERT_HOOK_MATCHES_PROD(g_devrWindowIsMultiSegment, ncclDevrWindowIsMultiSegment);
ASSERT_HOOK_MATCHES_PROD(g_devrWindowHasSysmemSegment, ncclDevrWindowHasSysmemSegment);
ASSERT_HOOK_MATCHES_PROD(g_devrInitOnce, ncclDevrInitOnce);
#undef ASSERT_HOOK_MATCHES_PROD

static ncclResult_t DefaultDevrFindWindow(struct ncclComm*, void const*, struct ncclDevrWindow** window) {
  if (window) *window = nullptr;
  return ncclSuccess;
}
std::function<ncclResult_t(struct ncclComm*, void const*, struct ncclDevrWindow**)> g_devrFindWindow =
    DefaultDevrFindWindow;
ncclResult_t ncclDevrFindWindow(struct ncclComm* comm, void const* ptr, struct ncclDevrWindow** window) {
  return g_devrFindWindow(comm, ptr, window);
}

bool g_devrWindowIsMultiSegmentValue = false;
bool g_devrWindowHasSysmemSegmentValue = false;
static bool DefaultDevrWindowIsMultiSegment(struct ncclDevrWindow*) { return g_devrWindowIsMultiSegmentValue; }
static bool DefaultDevrWindowHasSysmemSegment(struct ncclDevrWindow*) { return g_devrWindowHasSysmemSegmentValue; }
std::function<bool(struct ncclDevrWindow*)> g_devrWindowIsMultiSegment =
    DefaultDevrWindowIsMultiSegment;
std::function<bool(struct ncclDevrWindow*)> g_devrWindowHasSysmemSegment =
    DefaultDevrWindowHasSysmemSegment;

bool ncclDevrWindowIsMultiSegment(struct ncclDevrWindow* window) {
  return g_devrWindowIsMultiSegment(window);
}
bool ncclDevrWindowHasSysmemSegment(struct ncclDevrWindow* window) {
  return g_devrWindowHasSysmemSegment(window);
}

static ncclResult_t DefaultDevrInitOnce(struct ncclComm*) { return ncclSuccess; }
std::function<ncclResult_t(struct ncclComm*)> g_devrInitOnce = DefaultDevrInitOnce;
ncclResult_t ncclDevrInitOnce(struct ncclComm* comm) { return g_devrInitOnce(comm); }

void ResetDevRuntimeFakes() {
  g_devrFindWindow = DefaultDevrFindWindow;
  g_devrWindowIsMultiSegmentValue = false;
  g_devrWindowHasSysmemSegmentValue = false;
  g_devrWindowIsMultiSegment = DefaultDevrWindowIsMultiSegment;
  g_devrWindowHasSysmemSegment = DefaultDevrWindowHasSysmemSegment;
  g_devrInitOnce = DefaultDevrInitOnce;
}
